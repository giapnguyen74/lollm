"""
tokenization.py — one tokenizer interface, two implementations (shared infra).

Both expose:  encode(text) -> list[int] · decode(ids) -> str (skips special) ·
apply_chat(prompt) -> list[int] · chat_template (truthy if present) · eos_ids.

`HFTokenizer` wraps transformers (safetensors path). `GGUFTokenizer` is built from
the GGUF embedded vocab (GPT-2 byte-level BPE — Qwen/Llama-3).
(Module is named `tokenization` — NOT `tokenizers` — to avoid shadowing the PyPI
`tokenizers` package that transformers imports.)
"""

from __future__ import annotations

import functools
import heapq
import re as _plain

try:
    import regex as _re
    _HAVE_REGEX = True
except ImportError:
    import re as _re
    _HAVE_REGEX = False


# ───────────────────────── HF (safetensors) ─────────────────────────

class HFTokenizer:
    def __init__(self, path: str):
        # intentionally lazy: keeps `transformers` off the GGUF path, since GGUFTokenizer /
        # SPMTokenizer below are hand-written with no external deps.
        from transformers import AutoTokenizer
        self._wrap(AutoTokenizer.from_pretrained(path))

    @classmethod
    def from_tokenizer(cls, tk):
        """Wrap an already-built transformers tokenizer (e.g. reconstructed from a GGUF)."""
        self = cls.__new__(cls)
        self._wrap(tk)
        return self

    def _wrap(self, tk):
        self.tk = tk
        self.chat_template = tk.chat_template
        e = tk.eos_token_id
        self.eos_ids = [e] if e is not None else []

    @staticmethod
    def _as_ids(x):
        # Normalize whatever transformers hands back → flat list[int].
        # (`tokenize()`/`apply_chat_template()` may return a list, a tensor, or a
        # BatchEncoding/dict — list(BatchEncoding) gives KEYS, not ids.)
        if hasattr(x, "input_ids"):          # BatchEncoding
            x = x.input_ids
        elif isinstance(x, dict):
            x = x["input_ids"]
        if hasattr(x, "tolist"):             # tensor
            x = x.tolist()
        if x and isinstance(x[0], list):     # batched [[...]] → first row
            x = x[0]
        return list(x)

    def encode(self, text):
        return self._as_ids(self.tk(text))

    def apply_chat(self, prompt, enable_thinking=None):
        # Qwen3/Qwen3.5 templates branch on `enable_thinking`: undefined/true opens a
        # `<think>\n` block (model reasons first), false injects an empty `<think>\n\n</think>`
        # so it answers directly. Forward it only when set, so other templates are unaffected.
        kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        return self._as_ids(self.tk.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True, **kw))

    def decode(self, ids):
        return self.tk.decode(ids, skip_special_tokens=True)


# ───────────────────────── GGUF embedded BPE ─────────────────────────

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


class GGUFTokenizer:
    def __init__(self, meta: dict):
        if meta.get("tokenizer.ggml.model", "gpt2") != "gpt2":
            raise NotImplementedError("only GPT-2 byte-level BPE tokenizers supported")
        self.tokens = meta["tokenizer.ggml.tokens"]
        self.token_type = meta.get("tokenizer.ggml.token_type", [])
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.bpe_ranks = {tuple(m.split(" ")): i
                          for i, m in enumerate(meta.get("tokenizer.ggml.merges", []))}
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.chat_template = meta.get("tokenizer.chat_template")
        eos = meta.get("tokenizer.ggml.eos_token_id")
        self.eos_ids = [eos] if eos is not None else []
        self.special = {self.tokens[i]: i for i, t in enumerate(self.token_type) if t in (3, 4)}
        self._special_sorted = sorted(self.special, key=len, reverse=True)
        self._pat = _re.compile(_GPT2_PAT if _HAVE_REGEX else _FALLBACK_PAT)

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

    def encode(self, text):
        if not self._special_sorted:
            return self._encode_ordinary(text)
        pat = "(" + "|".join(_plain.escape(s) for s in self._special_sorted) + ")"
        ids = []
        for part in _plain.split(pat, text):
            if not part:
                continue
            ids.append(self.special[part]) if part in self.special else ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids):
        out = [self.tokens[i] for i in ids
               if not (i < len(self.token_type) and self.token_type[i] in (3, 4))]
        data = bytearray(self.byte_decoder.get(c, 63) for c in "".join(out))
        return data.decode("utf-8", errors="replace")

    def apply_chat(self, prompt, system="You are a helpful assistant."):
        text = (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        return self.encode(text)


# ───────────────────────── GGUF embedded SentencePiece ─────────────────────────

class SPMTokenizer:
    """
    SentencePiece tokenizer built from GGUF metadata (`tokenizer.ggml.model ==
    "llama"`) — used by Gemma / Llama-2 / Mistral-SPM. Implements llama.cpp's greedy
    score-based bigram merge + byte fallback. No external deps.

    STUDY NOTE: validate token-for-token vs llama.cpp / HF before fully trusting.
    The fiddly bit is the leading-space / dummy-prefix handling at segment starts.
    """

    SPACE = "▁"   # ▁ — SentencePiece's visible space marker

    def __init__(self, meta: dict):
        self.tokens = meta["tokenizer.ggml.tokens"]
        self.scores = meta.get("tokenizer.ggml.scores") or [0.0] * len(self.tokens)
        self.token_type = meta.get("tokenizer.ggml.token_type", [])
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.bos_id = meta.get("tokenizer.ggml.bos_token_id")
        self.chat_template = meta.get("tokenizer.chat_template")

        eos = meta.get("tokenizer.ggml.eos_token_id")
        self.eos_ids = [eos] if eos is not None else []
        # Gemma chat stops on <end_of_turn>, not the default <eos> — add it.
        eot = self.vocab.get("<end_of_turn>")
        if eot is not None and eot not in self.eos_ids:
            self.eos_ids.append(eot)

        # control (3) / user-defined (4) tokens are matched literally during encode.
        self.special = {self.tokens[i]: i for i, t in enumerate(self.token_type) if t in (3, 4)}
        self._special_sorted = sorted(self.special, key=len, reverse=True)

    def _encode_piece(self, text):
        if not text:
            return []
        sym = list(text.replace(" ", self.SPACE))     # one merged-string slot per char
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
                # max score first → negate; greedy SentencePiece merge.
                heapq.heappush(heap, (-self.scores[tid], l, r, len(sym[l]), len(sym[r])))

        for i in range(n - 1):
            add(i, i + 1)
        while heap:
            _, l, r, ll, rl = heapq.heappop(heap)
            if not sym[l] or not sym[r]:
                continue
            if len(sym[l]) != ll or len(sym[r]) != rl or nxt[l] != r:   # stale entry
                continue
            sym[l] += sym[r]
            sym[r] = ""
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
            ids.append(tid)
            return
        for b in piece.encode("utf-8"):                # byte fallback → <0xXX>
            bid = self.vocab.get(f"<0x{b:02X}>")
            if bid is not None:
                ids.append(bid)

    def encode(self, text):
        if not self._special_sorted:
            return self._encode_piece(text)
        pat = "(" + "|".join(_plain.escape(s) for s in self._special_sorted) + ")"
        ids = []
        for part in _plain.split(pat, text):
            if not part:
                continue
            ids.append(self.special[part]) if part in self.special else ids.extend(self._encode_piece(part))
        return ids

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            if i < len(self.token_type) and self.token_type[i] in (3, 4):
                continue                                # skip control / special
            tok = self.tokens[i]
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                buf.append(int(tok[3:5], 16))           # byte token
            else:
                buf.extend(tok.replace(self.SPACE, " ").encode("utf-8"))
        return buf.decode("utf-8", errors="replace")

    def apply_chat(self, prompt):
        return self.encode(self._render_chat(prompt))

    def _render_chat(self, prompt):
        if self.chat_template:
            try:
                from jinja2 import Environment
                env = Environment(trim_blocks=True, lstrip_blocks=True)
                env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
                bos = self.tokens[self.bos_id] if self.bos_id is not None else ""
                return env.from_string(self.chat_template).render(
                    messages=[{"role": "user", "content": prompt}],
                    add_generation_prompt=True, bos_token=bos)
            except Exception:
                pass
        # fallback: Gemma's turn format
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
