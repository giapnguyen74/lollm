"""
gemma4_debug.py — localize where our Gemma4 forward diverges from transformers.

compare_logits only checks the final logits; when they're wrong it doesn't say *where*.
This runs our model and the `transformers` reference on the same ids (fp32, CPU) and
prints the cosine similarity of the hidden state after EACH decoder layer, plus the
embedding. The first layer whose cosine falls off a cliff is where the bug lives:

  - embedding cos < 1            → embed lookup / ×√hidden scale
  - drops at layer 0             → attention / MLP / PLE / RoPE in the block
  - cliff exactly at layer 15    → shared-KV reuse (first shared layer on E2B)
  - slow steady decline          → a per-layer issue (PLE, a norm, scale=1.0)

    python src/gemma4_debug.py
    python src/gemma4_debug.py --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import transformers

import loader
import router


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-e2b-it")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=1,
                    help="layer to break down submodule-by-submodule")
    args = ap.parse_args()

    path = loader.resolve(args.model)
    L = loader.load(args.model)
    cfg_text = L.raw_config.get("text_config", L.raw_config)
    ids = L.tokenizer.encode(args.prompt)
    ids_t = torch.tensor([ids])
    print(f"ids = {ids}")

    # ── ours (fp32 CPU), capture per-layer hidden via forward hooks ──
    ours = router.route(L.model_type).load(L.raw_config, L.weights, L.fmt, "cpu", torch.float32)
    our_hidden = []
    for layer in ours.model.layers:
        layer.register_forward_hook(lambda m, i, o: our_hidden.append(o.detach()))
    with torch.no_grad():
        our_logits, _ = ours(ids_t)
    our_embed = (ours.model.embed_tokens(ids_t).float()
                 * (cfg_text["hidden_size"] ** 0.5))

    # ── reference text model with hidden states ──
    arch = (json.load(open(os.path.join(path, "config.json")))["architectures"] or [None])[0]
    ref_cls = getattr(transformers, arch, transformers.AutoModelForCausalLM)
    ref = ref_cls.from_pretrained(path, dtype=torch.float32).eval()
    lm = ref.model.language_model if hasattr(ref.model, "language_model") else ref.model
    with torch.no_grad():
        out = lm(ids_t, output_hidden_states=True)
    ref_hidden = out.hidden_states            # [0]=embeddings, [i+1]=after layer i

    # ── report ──
    layer_types = cfg_text["layer_types"]
    first_shared = cfg_text["num_hidden_layers"] - cfg_text.get("num_kv_shared_layers", 0)
    print(f"\nembedding cos = {_cos(our_embed, ref_hidden[0]):.6f}")
    print("layer  type               shared  cos(our, ref)")
    for i in range(len(our_hidden)):
        tag = "shared" if i >= first_shared else ""
        kind = "full" if layer_types[i] == "full_attention" else "sliding"
        print(f"{i:>4}  {kind:<9}         {tag:<6}  {_cos(our_hidden[i], ref_hidden[i + 1]):.6f}")
    # NB: the LAST layer's cos can read low even when correct — the reference's final
    # hidden_states entry is post-final-norm while our hook is pre-norm. Trust the
    # final-logits cos below as the verdict.
    print(f"\nfinal logits cos = {_cos(our_logits[0, -1], ref(ids_t).logits[0, -1]):.6f}")

    # ── PLE tensor check: ours vs the reference's get/project methods ──
    print("\n=== PLE (per-layer inputs) ===")
    try:
        ref_embed = lm.embed_tokens(ids_t)
        ref_ple_id = lm.get_per_layer_inputs(ids_t, ref_embed)
        ref_ple = lm.project_per_layer_inputs(ref_embed, ref_ple_id)     # (B,T,L,ple)
        our_ple = ours._per_layer_inputs(ids_t, our_embed)
        print(f"PLE whole-tensor cos = {_cos(our_ple, ref_ple):.6f}")
        for i in (0, args.layer, ref_ple.shape[2] - 1):
            print(f"  PLE[layer {i}] cos = {_cos(our_ple[:, :, i, :], ref_ple[:, :, i, :]):.6f}")
    except Exception as e:
        print(f"  (skipped: {type(e).__name__}: {e})")

    # ── submodule breakdown on the first broken layer ──
    n = args.layer
    print(f"\n=== submodule outputs on layer {n} ({layer_types[n]}) ===")
    grab = {}

    def cap(tag):
        def h(m, i, o):
            grab[tag] = (o[0] if isinstance(o, tuple) else o).detach()
        return h

    ol, rl = ours.model.layers[n], lm.layers[n]
    names = ["input_layernorm", "self_attn", "post_attention_layernorm",
             "pre_feedforward_layernorm", "mlp", "post_feedforward_layernorm",
             "per_layer_input_gate", "per_layer_projection", "post_per_layer_input_norm"]
    handles = []
    for nm in names:
        if hasattr(ol, nm) and hasattr(rl, nm):
            handles.append(getattr(ol, nm).register_forward_hook(cap("our_" + nm)))
            handles.append(getattr(rl, nm).register_forward_hook(cap("ref_" + nm)))
    with torch.no_grad():
        ours(ids_t)
        lm(ids_t)
    for nm in names:
        if "our_" + nm in grab and "ref_" + nm in grab:
            print(f"  {nm:<28} cos = {_cos(grab['our_' + nm], grab['ref_' + nm]):.6f}")
    for hd in handles:
        hd.remove()


if __name__ == "__main__":
    main()
