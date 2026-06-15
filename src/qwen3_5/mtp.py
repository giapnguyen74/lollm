"""
qwen3_5/mtp.py — the multi-token-prediction (MTP) head (step 6).

The Qwen3.5 checkpoint ships an Eagle/MTP-style speculative head (`mtp_num_hidden_layers:
1`). The base CausalLM *ignores* it (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`),
so it is **not needed for correct next-token generation** — it only accelerates decoding
by proposing token t+2. This module is therefore **additive and opt-in**: the base model
loads and passes parity without it.

To predict token t+2 from the main model's hidden state at position t and the embedding of
token t+1 (Eagle recipe, per docs/qwen3_5-architecture.md):

    h = pre_fc_norm_hidden(hidden_t)          # norm the previous hidden state
    e = pre_fc_norm_embedding(emb(token_{t+1}))   # norm the next-token embedding
    x = fc([e ; h])                           # combine 2·hidden → hidden
    x = decoder_layer(x)                      # one full-attention block (gated, like the main stack)
    x = norm(x)
    logits = lm_head(x)                       # SHARED head + SHARED embeddings

Caveat: no `transformers` class implements MTP (all list `mtp.*` as ignored), so unlike the
base LM there is **no reference to numerically parity-check against**. We match the
checkpoint's tensor layout exactly and follow the documented Eagle forward; the validation
here is structural (loads, runs, right shapes), not a logit comparison. The two details
that a reference would pin down — the `fc` concat order `[emb ; hidden]` and the position
offset of the MTP block — follow the doc and the standard Eagle convention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Qwen3_5Config
from .blocks import RMSNorm, RoPE
from .modeling_qwen3_5 import DecoderLayer


class MTP(nn.Module):
    """One-layer Eagle/MTP head. Reuses the family's full-attention decoder block; the
    shared `embed_tokens` / `lm_head` live on the main model and are applied by `speculate`."""

    def __init__(self, cfg: Qwen3_5Config):
        super().__init__()
        self.cfg = cfg
        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.pre_fc_norm_embedding = RMSNorm(h, eps)        # norm the next-token embedding
        self.pre_fc_norm_hidden = RMSNorm(h, eps)           # norm the previous hidden state
        self.fc = nn.Linear(2 * h, h, bias=False)           # combine [emb ; hidden] → hidden
        self.layers = nn.ModuleList(                        # one forced full-attention block
            [DecoderLayer(cfg, 0, layer_type="full_attention")])
        self.norm = RMSNorm(h, eps)                         # final norm before the shared lm_head
        self._rope = None

    def _rope_for(self, device):
        if self._rope is None:
            self._rope = RoPE(self.cfg.rotary_dim, self.cfg.rope_theta, device)
        return self._rope

    def forward(self, hidden, next_emb, positions=None):
        """`hidden`,`next_emb`: (B, L, H), already aligned (next_emb[:, i] = emb of the token
        following position i). Returns the MTP hidden state (B, L, H); apply the shared
        lm_head for logits. `positions` defaults to the t+1 alignment (i+1)."""
        b, ell, _ = hidden.shape
        if positions is None:
            positions = torch.arange(1, ell + 1, device=hidden.device)
        cos, sin = self._rope_for(hidden.device).cos_sin(positions, hidden.dtype)
        x = self.fc(torch.cat(
            [self.pre_fc_norm_embedding(next_emb), self.pre_fc_norm_hidden(hidden)], dim=-1))
        for layer in self.layers:
            x = layer(x, cos, sin, None, use_cache=False)   # cache=None → one-shot, no KV store
        return self.norm(x)


@torch.no_grad()
def speculate(model, mtp: MTP, input_ids):
    """Run the main model + MTP head to propose token t+2 for each context.

    Returns logits (B, L, vocab) with L = T-1: row i scores the token at absolute position
    i+2, given the context `input_ids[:, :i+1]` and the next token `input_ids[:, i+1]`.
    """
    _, _, hidden = model(input_ids, return_hidden=True)     # (B, T, H), pre-final-norm
    emb = model.model.embed_tokens(input_ids)               # (B, T, H)
    # align: hidden at position i pairs with the embedding of token i+1
    h = hidden[:, :-1]
    next_emb = emb[:, 1:]
    x = mtp(h, next_emb)
    return model.lm_head(x)


@torch.no_grad()
def generate_mtp(model, mtp: MTP, ids, decode, stop_ids, device, *, max_new_tokens=128,
                 temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.1):
    """
    Self-speculative decoding with the MTP head — a drop-in for `generate.generate` used by
    `run.py --mtp`. Each step the MTP head cheaply drafts token t+2, then ONE batched main
    forward over [pending, draft] verifies it: accept → two tokens committed for ~one main
    pass; reject → restore the pre-verify cache and re-run the single confirmed token.

    Output equals greedy autoregressive decoding exactly (the draft is only ever *accepted*
    when it matches the main model's own next token). With sampling (`temperature > 0`) it is
    a close approximation, not a formal lossless rejection sampler. The MTP block's internal
    position offset is unverifiable but irrelevant to correctness here: a wrong draft is
    simply rejected, so it can only cost speed, never accuracy.
    """
    from generate import sample_next

    eos = set(stop_ids)
    context = list(ids)
    generated, prev = [], ""

    def flush():
        nonlocal prev
        text = decode(generated)
        if len(text) > len(prev):
            piece, prev = text[len(prev):], text
            return piece
        return None

    # prefill → first pending token (not yet in the cache)
    logits, past, hidden = model(torch.tensor([ids], device=device), return_hidden=True)
    hidden_last = hidden[:, -1:]                                   # main hidden at last position
    pending = sample_next(logits[0, -1], temperature, top_k, top_p, repetition_penalty, context)

    produced = 0
    while produced < max_new_tokens:
        if pending in eos:
            break
        # emit `pending` (always a confirmed token)
        generated.append(pending); context.append(pending); produced += 1
        piece = flush()
        if piece:
            yield piece
        if produced >= max_new_tokens:
            break

        # 1. DRAFT t+2 cheaply: MTP(hidden_{t}, emb(pending)) → argmax
        emb = model.model.embed_tokens(torch.tensor([[pending]], device=device))
        draft = int(model.lm_head(mtp(hidden_last, emb))[0, -1].argmax())

        # 2. VERIFY: snapshot, then one batched main pass over [pending, draft]
        snap = past.clone()
        two = torch.tensor([[pending, draft]], device=device)
        logits2, past, hidden2 = model(two, past, return_hidden=True)
        true_next = sample_next(logits2[0, 0], temperature, top_k, top_p, repetition_penalty, context)

        # 3. ACCEPT / REJECT
        if draft == true_next:
            if draft in eos:
                break                                              # next real token is eos → stop
            generated.append(draft); context.append(draft); produced += 1
            piece = flush()
            if piece:
                yield piece
            pending = sample_next(logits2[0, 1], temperature, top_k, top_p, repetition_penalty, context)
            hidden_last = hidden2[:, 1:2]
        else:
            # draft wrong: roll back the speculative pass, commit only `pending`
            past = snap
            one = torch.tensor([[pending]], device=device)
            _, past, hidden1 = model(one, past, return_hidden=True)
            pending = true_next
            hidden_last = hidden1[:, -1:]


def _mtp_raw(canonical: str, fmt: str, to_raw):
    """MTP params live top-level as `mtp.*` in the checkpoint (the CausalLM's ignore regex
    is `^mtp.*`), so prefix our module-local names and reuse the family name map."""
    return to_raw("mtp." + canonical, fmt)


@torch.no_grad()
def load_mtp(model, weights: dict, fmt: str, device="cpu", dtype=None) -> MTP:
    """Build the MTP head and stream its `mtp.*` weights (same shape-checked load as the
    base family). Reuses the already-loaded `embed_tokens` / `lm_head` on `model`."""
    from .weights import to_raw, _set_param

    cfg = model.cfg
    with torch.device("meta"):
        mtp = MTP(cfg)
    for name, mp in list(mtp.named_parameters()):
        raw = _mtp_raw(name, fmt, to_raw)
        if raw not in weights:
            raise RuntimeError(f"qwen3_5 MTP: missing tensor for {name} (raw {raw})")
        t = weights.pop(raw)
        if tuple(t.shape) != tuple(mp.shape):
            raise RuntimeError(
                f"qwen3_5 MTP: shape mismatch {name}: {tuple(t.shape)} vs {tuple(mp.shape)}")
        if dtype is not None:
            t = t.to(dtype)
        _set_param(mtp, name, t.to(device))
    return mtp.eval()
