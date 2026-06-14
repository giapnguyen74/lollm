"""
generate.py — shared infra. The one generation loop + sampler.

Prefill → decode → stop is architecture-agnostic, so it lives here once. It calls
the family model's `forward(input_ids, past) -> (logits, past)`, samples, threads
the opaque `past` back in, and stops on the family's eos ids. A family provides the
forward, never its own loop. (If a family one day needs a different loop, we change
*this* — not the family.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_next(logits, temperature, top_k, top_p, repetition_penalty=1.0, prev_ids=None):
    if repetition_penalty != 1.0 and prev_ids:
        idx = torch.tensor(sorted(set(prev_ids)), device=logits.device, dtype=torch.long)
        logits = logits.clone()
        s = logits[idx]
        logits[idx] = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)
    if temperature <= 0.0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p and 0.0 < top_p < 1.0:
        sl, si = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
        rm = cum > top_p
        rm[..., 1:] = rm[..., :-1].clone()
        rm[..., 0] = False
        sl[rm] = float("-inf")
        logits = torch.empty_like(logits).scatter_(0, si, sl)
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1))


@torch.no_grad()
def generate(model, ids, decode, stop_ids, device, *, max_new_tokens=128,
             temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.1):
    """`decode(list[int]) -> str`. Yields incremental text pieces."""
    input_ids = torch.tensor([ids], device=device)
    context = list(ids)
    eos = set(stop_ids)

    logits, past = model(input_ids)
    nxt = sample_next(logits[0, -1], temperature, top_k, top_p, repetition_penalty, context)

    generated, prev = [], ""
    for _ in range(max_new_tokens):
        if nxt in eos:
            break
        generated.append(nxt)
        context.append(nxt)
        text = decode(generated)
        if len(text) > len(prev):
            yield text[len(prev):]
            prev = text
        logits, past = model(torch.tensor([[nxt]], device=device), past)
        nxt = sample_next(logits[0, -1], temperature, top_k, top_p, repetition_penalty, context)
