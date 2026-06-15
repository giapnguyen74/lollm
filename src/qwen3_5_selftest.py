"""
qwen3_5_selftest.py — STEP 1 + STEP 2 checks (no download).

STEP 1 (load skeleton): build a tiny model on meta, fabricate fake weights keyed by the
REAL HF tensor names (`model.language_model.*`, four GDN `in_proj_*`), run `qwen3_5.load`,
and verify the name map, shapes, tied embeddings, and fail-loud paths.

STEP 2 (full-attention forward): run one full-attention DecoderLayer (prefill + a cached
decode step) and check output/KV shapes; verify partial RoPE matches the transformers
reference numerically.

STEP 3 (Gated DeltaNet): run GatedDeltaNet directly — check output/cache shapes, conv
causality, that the recurrent (token-by-token) path equals the chunked prefill path, and
prefill-then-decode consistency.

STEP 4 (family cache + wired forward): run the whole hybrid model and verify a full
prefill matches token-by-token incremental decode (both mixers' caches + the position
counter), plus per-layer cache state shapes.

STEP 6 (MTP head): load the `mtp.*` head (name map + shapes) and run the speculative
forward predicting token t+2. Structural only — no transformers class implements MTP, so
there is no reference to numerically compare against (unlike step-5 parity).

Run in your venv (needs torch):  python src/qwen3_5_selftest.py
"""

from __future__ import annotations

import torch

from qwen3_5.config import Qwen3_5Config
from qwen3_5.modeling_qwen3_5 import Qwen3_5Model, DecoderLayer, Qwen3_5Cache
from qwen3_5.blocks import RoPE, GatedDeltaNet
from qwen3_5 import weights as W


def tiny_raw(tie=True):
    return {
        "model_type": "qwen3_5",
        "tie_word_embeddings": tie,
        "text_config": {
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25},
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
            "full_attention_interval": 4, "attn_output_gate": True,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4, "hidden_act": "silu", "tie_word_embeddings": tie,
        },
    }


def expected_params(cfg):
    with torch.device("meta"):
        skel = Qwen3_5Model(cfg)
    return {n: tuple(p.shape) for n, p in skel.named_parameters()}


def fake_weights(names, drop=None, mangle=None):
    """Keyed by the RAW (checkpoint) name = to_raw(canonical)."""
    w = {}
    for n, shape in names.items():
        if n == "lm_head.weight":          # tied → absent from checkpoint
            continue
        if n == drop:
            continue
        s = (shape[0] + 1,) + shape[1:] if n == mangle else shape
        w[W.to_raw(n, "hf")] = torch.zeros(s)
    return w


def check_step1():
    raw = tiny_raw(tie=True)
    cfg = Qwen3_5Config.from_hf(raw)
    names = expected_params(cfg)

    # name map: text params get the language_model prefix; lm_head stays top-level
    assert W.to_raw("model.embed_tokens.weight", "hf") == "model.language_model.embed_tokens.weight"
    assert W.to_raw("model.layers.0.linear_attn.in_proj_qkv.weight", "hf") == \
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    assert W.to_raw("lm_head.weight", "hf") == "lm_head.weight"

    # key shapes (full-attn gate ×2; the four GDN projections)
    hd, nh, nkv, nv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.linear_num_value_heads
    checks = {
        "model.layers.3.self_attn.q_proj.weight": (nh * hd * 2, cfg.hidden_size),
        "model.layers.3.self_attn.k_proj.weight": (nkv * hd, cfg.hidden_size),
        "model.layers.3.self_attn.q_norm.weight": (hd,),
        "model.layers.0.linear_attn.in_proj_qkv.weight": (cfg.conv_dim, cfg.hidden_size),
        "model.layers.0.linear_attn.in_proj_z.weight": (cfg.value_dim, cfg.hidden_size),
        "model.layers.0.linear_attn.in_proj_b.weight": (nv, cfg.hidden_size),
        "model.layers.0.linear_attn.in_proj_a.weight": (nv, cfg.hidden_size),
        "model.layers.0.linear_attn.conv1d.weight": (cfg.conv_dim, 1, cfg.linear_conv_kernel_dim),
        "model.layers.0.linear_attn.A_log": (nv,),
        "model.layers.0.linear_attn.dt_bias": (nv,),
        "model.layers.0.linear_attn.norm.weight": (cfg.linear_value_head_dim,),
        "model.layers.0.linear_attn.out_proj.weight": (cfg.hidden_size, cfg.value_dim),
    }
    for n, shp in checks.items():
        assert names.get(n) == shp, f"{n}: {names.get(n)} != {shp}"

    # load + tie
    model = W.load(raw, fake_weights(names), "hf", "cpu", torch.float32)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    # fail-loud: missing + shape mismatch
    for kw, msg in [(dict(drop="model.norm.weight"), "missing tensor"),
                    (dict(mangle="model.embed_tokens.weight"), "shape mismatch")]:
        try:
            W.load(raw, fake_weights(names, **kw), "hf", "cpu", torch.float32)
            raise AssertionError(f"expected RuntimeError ({msg})")
        except RuntimeError as e:
            assert msg in str(e), e

    n_full = sum(1 for t in cfg.layer_types if t == "full_attention")
    print(f"step1: params={len(names)} layers={cfg.num_hidden_layers} "
          f"({n_full} full, {cfg.num_hidden_layers - n_full} linear) — name map + shapes + tie OK")


def _ref_apply_rope(q, k, cos, sin):
    """transformers reference partial RoPE (cos/sin shape (B,T,rd), unsqueeze head dim)."""
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rd = cos.shape[-1]
    qr, qp = q[..., :rd], q[..., rd:]
    kr, kp = k[..., :rd], k[..., rd:]
    qe = torch.cat([(qr * cos) + (rotate_half(qr) * sin), qp], dim=-1)
    ke = torch.cat([(kr * cos) + (rotate_half(kr) * sin), kp], dim=-1)
    return qe, ke


def check_step2():
    torch.manual_seed(0)
    cfg = Qwen3_5Config.from_hf(tiny_raw())

    # partial RoPE parity vs reference
    B, T, nh, hd, rd = 1, 5, cfg.num_attention_heads, cfg.head_dim, cfg.rotary_dim
    rope = RoPE(rd, cfg.rope_theta, "cpu")
    cos, sin = rope.cos_sin(torch.arange(T), torch.float32)
    q, k = torch.randn(B, nh, T, hd), torch.randn(B, nh, T, hd)
    o_q, o_k = RoPE.apply(q, k, cos, sin)
    r_q, r_k = _ref_apply_rope(q, k, cos[None].expand(B, -1, -1), sin[None].expand(B, -1, -1))
    assert torch.allclose(o_q, r_q, atol=1e-6) and torch.allclose(o_k, r_k, atol=1e-6), "RoPE mismatch"
    assert torch.allclose(o_q[..., rd:], q[..., rd:]), "pass-through dims must be unrotated"

    # one full-attention layer: prefill then cached decode, KV via the cache methods
    layer = DecoderLayer(cfg, 3).eval()       # idx 3 → full_attention
    cache = Qwen3_5Cache(cfg.num_hidden_layers)
    x = torch.randn(B, T, cfg.hidden_size)
    y = layer(x, cos, sin, cache, 3, use_cache=True)
    assert y.shape == x.shape
    assert cache.read_kv(3)[0].shape == (B, cfg.num_key_value_heads, T, hd)
    x1 = torch.randn(B, 1, cfg.hidden_size)
    c1, s1 = rope.cos_sin(torch.arange(T, T + 1), torch.float32)
    y1 = layer(x1, c1, s1, cache, 3, use_cache=True)
    assert y1.shape == x1.shape
    assert cache.read_kv(3)[0].shape == (B, cfg.num_key_value_heads, T + 1, hd)
    print("step2: partial-RoPE parity + gated-attention forward (prefill+decode) OK")


def check_step3():
    torch.manual_seed(0)
    cfg = Qwen3_5Config.from_hf(tiny_raw())
    gdn = GatedDeltaNet(cfg).eval()
    B, T = 1, 9
    cd, K = cfg.conv_dim, cfg.linear_conv_kernel_dim
    rec_shape = (B, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim)
    x = torch.randn(B, T, cfg.hidden_size)

    # prefill: output shape + cache shapes (conv_state width = kernel-1, recurrent state)
    out, conv_s, rec_s = gdn(x, use_cache=True)
    assert out.shape == (B, T, cfg.hidden_size), out.shape
    assert conv_s.shape == (B, cd, K - 1), conv_s.shape
    assert rec_s.shape == rec_shape, rec_s.shape
    out_nocache, c0, r0 = gdn(x)               # use_cache=False → no state returned
    assert c0 is None and r0 is None
    assert torch.allclose(out, out_nocache, atol=1e-6)

    # causality: perturbing token t leaves outputs < t unchanged
    t = 5
    x2 = x.clone(); x2[:, t] += 3.0
    o2 = gdn(x2)[0]
    assert torch.allclose(o2[:, :t], out_nocache[:, :t], atol=1e-5), "GDN not causal"
    assert not torch.allclose(o2[:, t], out_nocache[:, t], atol=1e-5), "perturbation had no effect"

    # recurrent decode (token-by-token, carrying state) must equal chunked prefill
    seq, cs, rs = [], None, None
    for i in range(T):
        oi, cs, rs = gdn(x[:, i:i + 1], conv_state=cs, recurrent_state=rs, use_cache=True)
        seq.append(oi)
    seq = torch.cat(seq, dim=1)
    assert torch.allclose(seq, out_nocache, atol=1e-4), \
        f"recurrent vs chunked mismatch (max {(seq - out_nocache).abs().max():.2e})"

    # prefill-then-decode: chunked over first T-1, recurrent for the last, matches full prefill
    _, cp, rp = gdn(x[:, :T - 1], use_cache=True)
    o_last = gdn(x[:, T - 1:T], conv_state=cp, recurrent_state=rp, use_cache=True)[0]
    assert torch.allclose(o_last, out_nocache[:, T - 1:T], atol=1e-4), "prefill+decode mismatch"
    print("step3: GDN forward — shapes, causality, recurrent==chunked, prefill+decode OK")


def check_step4():
    torch.manual_seed(0)
    cfg = Qwen3_5Config.from_hf(tiny_raw())
    model = Qwen3_5Model(cfg).eval()           # random init — consistency holds regardless
    B, T = 1, 9
    ids = torch.randint(0, cfg.vocab_size, (B, T))

    # full prefill: logits at every position + a returned family cache
    full, cache = model(ids)
    assert full.shape == (B, T, cfg.vocab_size), full.shape
    assert isinstance(cache, Qwen3_5Cache) and cache.seen_tokens == T
    # cache holds the right state type per layer (KV via the policy vs fixed linear slot)
    for i, layer in enumerate(model.model.layers):
        if layer.layer_type == "full_attention":
            K, _ = cache.read_kv(i)
            assert K.shape == (B, cfg.num_key_value_heads, T, cfg.head_dim), (i, K.shape)
        else:
            conv, rec = cache.linear_state(i)
            assert conv.shape == (B, cfg.conv_dim, cfg.linear_conv_kernel_dim - 1), (i, conv.shape)
            assert rec.shape == (B, cfg.linear_num_value_heads,
                                 cfg.linear_key_head_dim, cfg.linear_value_head_dim), (i, rec.shape)

    # incremental decode (token-by-token, threading the cache) must match full prefill —
    # exercises BOTH mixers' caches through the whole hybrid stack + the position counter
    logits0, past = model(ids[:, :1])
    outs = [logits0[:, -1]]
    for i in range(1, T):
        li, past = model(ids[:, i:i + 1], past)
        assert past.seen_tokens == i + 1
        outs.append(li[:, -1])
    inc = torch.stack(outs, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), \
        f"prefill vs incremental decode mismatch (max {(full - inc).abs().max():.2e})"

    # two-token cached continuation (the --mtp verify path: GDN chunked WITH prior state)
    # must equal feeding the same two tokens one at a time.
    _, base = model(ids[:, :T])
    nxt = torch.randint(0, cfg.vocab_size, (B, 2))
    l_chunk, _ = model(nxt, base.clone())                    # [a,b] in one cached pass
    s0, p = model(nxt[:, :1], base.clone())
    s1, _ = model(nxt[:, 1:2], p)                            # a then b
    l_step = torch.cat([s0, s1], dim=1)
    assert torch.allclose(l_chunk, l_step, atol=1e-4), \
        f"2-token cached chunk vs step mismatch (max {(l_chunk - l_step).abs().max():.2e})"
    print("step4: family cache + wired forward — prefill==incremental, 2-token cached==stepwise OK")


def check_step6():
    torch.manual_seed(0)
    from qwen3_5.mtp import MTP, load_mtp, speculate
    cfg = Qwen3_5Config.from_hf(tiny_raw())
    model = Qwen3_5Model(cfg).eval()              # random init

    # expected MTP params → fake checkpoint weights keyed by the RAW (mtp.*) names
    with torch.device("meta"):
        skel = MTP(cfg)
    names = {n: tuple(p.shape) for n, p in skel.named_parameters()}
    raw = {W.to_raw("mtp." + n, "hf"): torch.randn(*s) for n, s in names.items()}
    # names land top-level as mtp.* (matches the checkpoint's `^mtp.*` ignore regex)
    assert W.to_raw("mtp.fc.weight", "hf") == "mtp.fc.weight"
    assert names["fc.weight"] == (cfg.hidden_size, 2 * cfg.hidden_size)
    assert names["layers.0.self_attn.q_proj.weight"] == (cfg.num_attention_heads * cfg.head_dim * 2,
                                                         cfg.hidden_size)

    mtp = load_mtp(model, dict(raw), "hf", "cpu", torch.float32)

    # speculative forward: predict token t+2 → logits (B, T-1, vocab)
    B, T = 1, 9
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits = speculate(model, mtp, ids)
    assert logits.shape == (B, T - 1, cfg.vocab_size), logits.shape
    assert torch.isfinite(logits).all()

    # fail-loud on a missing MTP tensor
    bad = dict(raw); bad.pop(W.to_raw("mtp.norm.weight", "hf"))
    try:
        load_mtp(model, bad, "hf", "cpu", torch.float32)
        raise AssertionError("expected RuntimeError (missing tensor)")
    except RuntimeError as e:
        assert "missing tensor" in str(e), e
    print("step6: MTP head — mtp.* name map + shapes + speculative t+2 forward OK "
          "(structural; no transformers reference exists)")


def main():
    check_step1()
    check_step2()
    check_step3()
    check_step4()
    check_step6()
    print("QWEN3_5 SELFTEST: PASS ✅")


if __name__ == "__main__":
    main()
