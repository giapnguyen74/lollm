"""
qwen3_5_moe/mtp.py — the multi-token-prediction (MTP) head (opt-in, additive).

Same Eagle/MTP head as the dense qwen3_5 family — it reuses one full-attention DecoderLayer,
so on this family that block's FFN follows the same MoE-vs-dense rule as the main stack
(decided by `cfg.is_moe_layer`). The base CausalLM ignores `mtp.*` and passes parity without
it; this head only accelerates decoding by drafting token t+2. There is no `transformers`
reference for MTP, so validation here is structural (loads, runs, right shapes) — the strict
shape-checked load is what catches a wrong FFN kind for the MTP block.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from generate import sample_next
from .config import Qwen3_5MoeConfig
from .blocks import RMSNorm, RoPE
from .modeling_qwen3_5_moe import DecoderLayer
from .weights import to_raw, _set_param


class MTP(nn.Module):
    """One-layer Eagle/MTP head. Reuses the family's full-attention decoder block; the
    shared `embed_tokens` / `lm_head` live on the main model and are applied by `speculate`."""

    def __init__(self, cfg: Qwen3_5MoeConfig):
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
    Self-speculative decoding with the MTP head — a drop-in for `generate.generate`. Each
    step the MTP head cheaply drafts token t+2, then ONE batched main forward over
    [pending, draft] verifies it: accept → two tokens for ~one main pass; reject → restore
    the pre-verify cache and re-run the single confirmed token.
    """
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

    logits, past, hidden = model(torch.tensor([ids], device=device), return_hidden=True)
    hidden_last = hidden[:, -1:]
    pending = sample_next(logits[0, -1], temperature, top_k, top_p, repetition_penalty, context)

    produced = 0
    while produced < max_new_tokens:
        if pending in eos:
            break
        generated.append(pending); context.append(pending); produced += 1
        piece = flush()
        if piece:
            yield piece
        if produced >= max_new_tokens:
            break

        # 1. DRAFT t+2 cheaply
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
                break
            generated.append(draft); context.append(draft); produced += 1
            piece = flush()
            if piece:
                yield piece
            pending = sample_next(logits2[0, 1], temperature, top_k, top_p, repetition_penalty, context)
            hidden_last = hidden2[:, 1:2]
        else:
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
    cfg = model.cfg
    with torch.device("meta"):
        mtp = MTP(cfg)
    for name, mp in list(mtp.named_parameters()):
        raw = _mtp_raw(name, fmt, to_raw)
        if raw not in weights:
            raise RuntimeError(f"qwen3_5_moe MTP: missing tensor for {name} (raw {raw})")
        t = weights.pop(raw)
        if tuple(t.shape) != tuple(mp.shape):
            raise RuntimeError(
                f"qwen3_5_moe MTP: shape mismatch {name}: {tuple(t.shape)} vs {tuple(mp.shape)}")
        if dtype is not None:
            t = t.to(dtype)
        _set_param(mtp, name, t.to(device))
    return mtp.eval()
