"""
tokenization.py — one tokenizer interface, hand-written, NO `transformers` (shared infra).

Both engines expose:  encode(text) -> list[int] · decode(ids) -> str (skips specials) ·
apply_chat(prompt, enable_thinking=None) -> list[int] · chat_template · eos_ids.

  - `BPETokenizer`  — GPT-2 byte-level BPE (Qwen, Llama-3).
  - `SPMTokenizer`  — SentencePiece score-merge (Gemma, Llama-2, Mistral-SPM).

Each builds from **either** GGUF metadata (`from_gguf`) **or** a Hugging Face safetensors
directory (`from_hf`): `tokenizer.json` / `vocab.json`+`merges.txt` (BPE) or `tokenizer.model`
(SPM SentencePiece protobuf), plus `tokenizer_config.json` (special tokens, eos/bos,
add_bos, chat_template) and an optional standalone `chat_template.jinja`. The safetensors
path no longer uses `transformers.AutoTokenizer` — call `HFTokenizer(path)` (a factory that
sniffs the files and returns the right engine).

(Module is named `tokenization` — NOT `tokenizers` — to avoid shadowing the PyPI
`tokenizers` package.)
"""

from __future__ import annotations

import functools
import heapq
import json
import os
import struct
import re as _plain

try:
    import regex as _re
    _HAVE_REGEX = True
except ImportError:
    import re as _re
    _HAVE_REGEX = False


# ───────────────────────── shared helpers ─────────────────────────

_GPT2_PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
             r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_FALLBACK_PAT = r"'\w+|\w+| ?[^\s\w]+|\s+(?!\S)|\s+"


@functools.lru_cache()
def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _pairs(w):
    return {(w[i], w[i + 1]) for i in range(len(w) - 1)}


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _hf_chat_template(path, cfg):
    """Chat template: a standalone `chat_template.jinja` wins (newer repos), else the
    inline `chat_template` in tokenizer_config.json (which may be a [{name, template}] list)."""
    jinja = os.path.join(path, "chat_template.jinja")
    if os.path.exists(jinja):
        with open(jinja, encoding="utf-8") as f:
            return f.read()
    ct = cfg.get("chat_template")
    if isinstance(ct, list):                      # [{"name": "default", "template": "..."}]
        ct = next((t["template"] for t in ct if t.get("name") == "default"), ct[0]["template"])
    return ct


def _hf_added_tokens(path, cfg):
    """Return {id: (content, is_special)} merging tokenizer_config's `added_tokens_decoder`
    and tokenizer.json's `added_tokens` (the literal-match tokens outside the BPE/SPM vocab)."""
    added = {}
    for sid, info in (cfg.get("added_tokens_decoder") or {}).items():
        added[int(sid)] = (info["content"], bool(info.get("special")))
    tj = os.path.join(path, "tokenizer.json")
    if os.path.exists(tj):
        for a in (_read_json(tj).get("added_tokens") or []):
            added.setdefault(int(a["id"]), (a["content"], bool(a.get("special"))))
    return added


def _tok_id(s, vocab, content_to_id):
    return vocab.get(s, content_to_id.get(s)) if s is not None else None


def _render_chat(chat_template, prompt, bos_token, enable_thinking=None):
    """Render a single user turn with the model's Jinja chat template (add_generation_prompt)."""
    from jinja2 import Environment
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return env.from_string(chat_template).render(
        messages=[{"role": "user", "content": prompt}],
        add_generation_prompt=True, bos_token=bos_token or "", **kw)


# ───────────────────────── byte-level BPE (Qwen, Llama-3) ─────────────────────────

class BPETokenizer:
    """GPT-2 byte-level BPE. Build via `from_gguf` (embedded vocab) or `from_hf` (files)."""

    def _setup(self, tokens, merges, special_map, special_ids, eos_ids, chat_template,
               bos_id, add_bos):
        self.tokens = tokens                                  # id -> token string
        self.vocab = {t: i for i, t in enumerate(tokens) if t is not None}
        self.bpe_ranks = {tuple(m): i for i, m in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.special = special_map                            # literal token -> id
        self.special_ids = set(special_ids)                   # ids to skip on decode
        self.eos_ids = eos_ids
        self.chat_template = chat_template
        self.bos_id, self.add_bos = bos_id, add_bos
        self._special_sorted = sorted(special_map, key=len, reverse=True)
        self._pat = _re.compile(_GPT2_PAT if _HAVE_REGEX else _FALLBACK_PAT)
        return self

    @classmethod
    def from_gguf(cls, meta):
        if meta.get("tokenizer.ggml.model", "gpt2") != "gpt2":
            raise NotImplementedError("only GPT-2 byte-level BPE tokenizers supported")
        self = cls.__new__(cls)
        tokens = list(meta["tokenizer.ggml.tokens"])
        ttype = meta.get("tokenizer.ggml.token_type", [])
        merges = [m.split(" ") for m in meta.get("tokenizer.ggml.merges", [])]
        special_map = {tokens[i]: i for i, t in enumerate(ttype) if t in (3, 4)}
        special_ids = {i for i, t in enumerate(ttype) if t in (3, 4)}
        eos = meta.get("tokenizer.ggml.eos_token_id")
        bos = meta.get("tokenizer.ggml.bos_token_id")
        return self._setup(tokens, merges, special_map, special_ids,
                           [eos] if eos is not None else [], meta.get("tokenizer.chat_template"),
                           bos, bool(meta.get("tokenizer.ggml.add_bos_token", False)))

    @classmethod
    def from_hf(cls, path):
        self = cls.__new__(cls)
        cfg = _read_json(os.path.join(path, "tokenizer_config.json"))
        added = _hf_added_tokens(path, cfg)

        # base vocab + merges: prefer tokenizer.json's model.*, else vocab.json + merges.txt.
        tj = os.path.join(path, "tokenizer.json")
        if os.path.exists(tj):
            model = _read_json(tj)["model"]
            base = model["vocab"]                              # token -> id
            merges = [m.split(" ") if isinstance(m, str) else list(m) for m in model.get("merges", [])]
        else:
            base = _read_json(os.path.join(path, "vocab.json"))
            with open(os.path.join(path, "merges.txt"), encoding="utf-8") as f:
                merges = [ln.split() for ln in f.read().splitlines() if ln and not ln.startswith("#")]

        size = max([*base.values(), *added] or [0]) + 1
        tokens = [None] * size
        for t, i in base.items():
            tokens[i] = t
        for i, (content, _) in added.items():
            tokens[i] = content
        content_to_id = {c: i for i, (c, _) in added.items()}
        special_map = {c: i for i, (c, _) in added.items()}    # all added tokens match literally
        special_ids = {i for i, (_, sp) in added.items() if sp}
        eos = _tok_id(cfg.get("eos_token"), base, content_to_id)
        bos = _tok_id(cfg.get("bos_token"), base, content_to_id)
        return self._setup(tokens, merges, special_map, special_ids,
                           [eos] if eos is not None else [], _hf_chat_template(path, cfg),
                           bos, bool(cfg.get("add_bos_token", False)))

    @functools.lru_cache(maxsize=8192)
    def _bpe(self, token):
        word = tuple(token)
        if len(word) < 2:
            return token
        while True:
            pairs = _pairs(word)
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, 1e10))
            if best not in self.bpe_ranks:
                break
            a, b = best
            new, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new.append(a + b); i += 2
                else:
                    new.append(word[i]); i += 1
            word = tuple(new)
            if len(word) == 1:
                break
        return " ".join(word)

    def _encode_ordinary(self, text):
        ids = []
        for chunk in self._pat.findall(text):
            mapped = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            for piece in self._bpe(mapped).split(" "):
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
        return ids

    def encode(self, text, add_special=True):
        ids = [self.bos_id] if (add_special and self.add_bos and self.bos_id is not None) else []
        if not self._special_sorted:
            ids.extend(self._encode_ordinary(text))
            return ids
        pat = "(" + "|".join(_plain.escape(s) for s in self._special_sorted) + ")"
        for part in _plain.split(pat, text):
            if not part:
                continue
            ids.append(self.special[part]) if part in self.special else ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids):
        out = [self.tokens[i] for i in ids if i not in self.special_ids]
        data = bytearray(self.byte_decoder.get(c, 63) for c in "".join(out))
        return data.decode("utf-8", errors="replace")

    def apply_chat(self, prompt, enable_thinking=None):
        if self.chat_template:
            text = _render_chat(self.chat_template, prompt, None, enable_thinking)
            return self.encode(text, add_special=False)     # template owns specials (incl. bos)
        text = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        return self.encode(text)


# Back-compat alias: the GGUF loader historically used this name for the BPE engine.
GGUFTokenizer = BPETokenizer


# ───────────────────────── SentencePiece (Gemma, Llama-2) ─────────────────────────

def _read_spm_model(blob: bytes):
    """Parse a SentencePiece `tokenizer.model` (ModelProto) → list of (piece, score, type).
    Minimal protobuf reader: top-level field 1 = repeated SentencePiece{piece=1,score=2,type=3}."""
    def varint(buf, i):
        shift = res = 0
        while True:
            b = buf[i]; i += 1
            res |= (b & 0x7F) << shift
            if not b & 0x80:
                return res, i
            shift += 7

    def skip(buf, i, wire):
        if wire == 0:
            return varint(buf, i)[1]
        if wire == 1:
            return i + 8
        if wire == 2:
            n, i = varint(buf, i); return i + n
        if wire == 5:
            return i + 4
        raise ValueError(f"bad wire type {wire}")

    pieces, add_dummy_prefix, i, n = [], True, 0, len(blob)   # SP default add_dummy_prefix=True
    while i < n:
        key, i = varint(blob, i)
        field, wire = key >> 3, key & 7
        if field == 1 and wire == 2:                          # a SentencePiece message
            length, i = varint(blob, i)
            msg, j, end = blob, i, i + length
            i = end
            piece, score, ptype = None, 0.0, 1
            while j < end:
                k, j = varint(msg, j)
                f, w = k >> 3, k & 7
                if f == 1 and w == 2:
                    ln, j = varint(msg, j); piece = msg[j:j + ln].decode("utf-8"); j += ln
                elif f == 2 and w == 5:
                    score = struct.unpack_from("<f", msg, j)[0]; j += 4
                elif f == 3 and w == 0:
                    ptype, j = varint(msg, j)
                else:
                    j = skip(msg, j, w)
            pieces.append((piece, score, ptype))
        elif field == 3 and wire == 2:                        # normalizer_spec message
            length, i = varint(blob, i)
            msg, j, end = blob, i, i + length
            i = end
            while j < end:                                    # field 3 = add_dummy_prefix (bool)
                k, j = varint(msg, j)
                f, w = k >> 3, k & 7
                if f == 3 and w == 0:
                    v, j = varint(msg, j); add_dummy_prefix = bool(v)
                else:
                    j = skip(msg, j, w)
        else:
            i = skip(blob, i, wire)
    return pieces, add_dummy_prefix


class SPMTokenizer:
    """
    SentencePiece tokenizer (Gemma / Llama-2 / Mistral-SPM). Greedy score-based bigram
    merge + byte fallback — matches SentencePiece / llama.cpp. No external deps.

    STUDY NOTE: validate token-for-token vs HF before fully trusting; the fiddly bit is
    the leading-space / add_dummy_prefix handling at segment starts.
    """

    SPACE = "▁"   # ▁ — SentencePiece's visible space marker

    def _setup(self, tokens, scores, special_map, special_ids, eos_ids, chat_template,
               bos_id, add_bos=False, add_dummy_prefix=False):
        self.tokens = tokens
        self.scores = scores
        self.vocab = {t: i for i, t in enumerate(tokens) if t is not None}
        self.special = special_map
        self.special_ids = set(special_ids)
        self.eos_ids = eos_ids
        self.chat_template = chat_template
        self.bos_id, self.add_bos = bos_id, add_bos
        self.add_dummy_prefix = add_dummy_prefix       # SP add_dummy_prefix (Gemma: False)
        self._special_sorted = sorted(special_map, key=len, reverse=True)
        return self

    @classmethod
    def from_gguf(cls, meta):
        self = cls.__new__(cls)
        tokens = list(meta["tokenizer.ggml.tokens"])
        scores = list(meta.get("tokenizer.ggml.scores") or [0.0] * len(tokens))
        ttype = meta.get("tokenizer.ggml.token_type", [])
        special_map = {tokens[i]: i for i, t in enumerate(ttype) if t in (3, 4)}
        special_ids = {i for i, t in enumerate(ttype) if t in (3, 4)}
        eos = meta.get("tokenizer.ggml.eos_token_id")
        eos_ids = [eos] if eos is not None else []
        eot = {t: i for i, t in enumerate(tokens)}.get("<end_of_turn>")
        if eot is not None and eot not in eos_ids:
            eos_ids.append(eot)
        return self._setup(tokens, scores, special_map, special_ids, eos_ids,
                           meta.get("tokenizer.chat_template"),
                           meta.get("tokenizer.ggml.bos_token_id"),
                           bool(meta.get("tokenizer.ggml.add_bos_token", False)))

    @classmethod
    def from_hf(cls, path):
        self = cls.__new__(cls)
        cfg = _read_json(os.path.join(path, "tokenizer_config.json"))
        added = _hf_added_tokens(path, cfg)
        model_path = os.path.join(path, "tokenizer.model")
        if not os.path.exists(model_path):
            raise NotImplementedError(
                "gemma/llama SPM tokenizer needs tokenizer.model (SentencePiece) — not found.")
        with open(model_path, "rb") as f:
            pieces, add_dummy_prefix = _read_spm_model(f.read())
        size = max(len(pieces) - 1, *added) + 1 if added else len(pieces)
        tokens = [None] * size
        scores = [0.0] * size
        for i, (piece, score, _) in enumerate(pieces):
            tokens[i], scores[i] = piece, score
        for i, (content, _) in added.items():                 # added tokens override by id
            tokens[i] = content
        content_to_id = {c: i for i, (c, _) in added.items()}
        # specials = SPM control/user-defined pieces (type 3/4) + any added specials
        special_map = {p: i for i, (p, _, t) in enumerate(pieces) if t in (3, 4)}
        special_ids = {i for i, (_, _, t) in enumerate(pieces) if t in (3, 4)}
        for i, (content, sp) in added.items():
            special_map[content] = i
            if sp:
                special_ids.add(i)
        vocab = {t: i for i, t in enumerate(tokens) if t is not None}
        eos_ids = []
        eos = _tok_id(cfg.get("eos_token"), vocab, content_to_id)
        if eos is not None:
            eos_ids.append(eos)
        eot = vocab.get("<end_of_turn>")             # Gemma stops on <end_of_turn>, not <eos>
        if eot is not None and eot not in eos_ids:
            eos_ids.append(eot)
        bos = _tok_id(cfg.get("bos_token"), vocab, content_to_id)
        return self._setup(tokens, scores, special_map, special_ids, eos_ids,
                           _hf_chat_template(path, cfg), bos,
                           bool(cfg.get("add_bos_token", False)), add_dummy_prefix)

    def _encode_piece(self, text):
        if not text:
            return []
        # SentencePiece add_dummy_prefix (model-dependent: Llama-2 True, Gemma False):
        # when on, prepend a leading ▁ so "hi" tokenizes like " hi".
        s = text.replace(" ", self.SPACE)
        if self.add_dummy_prefix and not s.startswith(self.SPACE):
            s = self.SPACE + s
        sym = list(s)
        n = len(sym)
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1
        heap = []

        def add(l, r):
            if l < 0 or r < 0:
                return
            tid = self.vocab.get(sym[l] + sym[r])
            if tid is not None:
                heapq.heappush(heap, (-self.scores[tid], l, r, len(sym[l]), len(sym[r])))

        for i in range(n - 1):
            add(i, i + 1)
        while heap:
            _, l, r, ll, rl = heapq.heappop(heap)
            if not sym[l] or not sym[r]:
                continue
            if len(sym[l]) != ll or len(sym[r]) != rl or nxt[l] != r:
                continue
            sym[l] += sym[r]; sym[r] = ""
            nxt[l] = nxt[r]
            if nxt[r] != -1:
                prev[nxt[r]] = l
            add(prev[l], l)
            add(l, nxt[l])

        ids, i = [], 0
        while i != -1:
            if sym[i]:
                self._emit(sym[i], ids)
            i = nxt[i]
        return ids

    def _emit(self, piece, ids):
        tid = self.vocab.get(piece)
        if tid is not None:
            ids.append(tid); return
        for b in piece.encode("utf-8"):
            bid = self.vocab.get(f"<0x{b:02X}>")
            if bid is not None:
                ids.append(bid)

    def encode(self, text, add_special=True):
        ids = [self.bos_id] if (add_special and self.add_bos and self.bos_id is not None) else []
        if not self._special_sorted:
            ids.extend(self._encode_piece(text))
            return ids
        pat = "(" + "|".join(_plain.escape(s) for s in self._special_sorted) + ")"
        for part in _plain.split(pat, text):
            if not part:
                continue
            ids.append(self.special[part]) if part in self.special else ids.extend(self._encode_piece(part))
        return ids

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            if i in self.special_ids:
                continue
            tok = self.tokens[i]
            if tok and len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                buf.append(int(tok[3:5], 16))
            elif tok:
                buf.extend(tok.replace(self.SPACE, " ").encode("utf-8"))
        return buf.decode("utf-8", errors="replace")

    def apply_chat(self, prompt, enable_thinking=None):
        bos = self.tokens[self.bos_id] if self.bos_id is not None else ""
        if self.chat_template:
            try:                                          # template emits bos_token itself
                return self.encode(_render_chat(self.chat_template, prompt, bos, enable_thinking),
                                   add_special=False)
            except Exception:
                pass
        return self.encode(f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n")


# ───────────────────────── HF safetensors factory ─────────────────────────

def HFTokenizer(path):
    """Build a tokenizer from a HF safetensors directory — no `transformers`. Sniffs the
    files: `tokenizer.model` → SentencePiece (Gemma/Llama-2); else byte-level BPE."""
    if os.path.exists(os.path.join(path, "tokenizer.model")):
        return SPMTokenizer.from_hf(path)
    if (os.path.exists(os.path.join(path, "tokenizer.json"))
            or os.path.exists(os.path.join(path, "vocab.json"))):
        return BPETokenizer.from_hf(path)
    raise NotImplementedError(
        f"no recognized tokenizer files in {path} "
        f"(need tokenizer.model, tokenizer.json, or vocab.json+merges.txt)")
