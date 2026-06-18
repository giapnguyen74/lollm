# TODOs — near-term, actionable

Things we intend to fix **soon** (concrete, scoped). Longer-term / untimed ideas live in
[ROADMAP.md](./ROADMAP.md); resolved gotchas become lessons in
[LESSONS.md](./LESSONS.md).

**IDs are stable** — `T-#` never renumbers or gets reused. When an item is done, flip its
status to ✅ (leave it in place for one cycle, then move the lesson to LESSONS.md and
delete the row). Status: 🔴 open · 🟡 in progress · ✅ done.

| ID  | status | area    | title |
|-----|--------|---------|-------|
| T-1 | 🔴 open | mps / dtype | Default Gemma to bf16 on MPS (fp16 overflow risk the gate can't see) |
| T-2 | 🔴 open | parity      | Add a >sliding_window prompt to the parity gate (window is untested) |
| T-3 | 🔴 open | docs        | Stale reference: `qwen3_5_selftest.py` is cited but doesn't exist |

---

## T-1 · Default Gemma to bf16 on MPS

`run.py::pick_dtype` returns **float16 on MPS**, but the parity gate runs **fp32 on CPU**,
so fp16-only overflow is invisible to it. Gemma is the worst case (×√hidden embedding,
wide GeGLU, historical fp16 `inf`s). Softmax/RMSNorm upcast to fp32, but the q·kᵀ matmul
and GeGLU run in fp16 on MPS.

**Do:** prefer **bfloat16 on MPS** where the torch build supports it (fall back to fp16
only if unavailable); or add an fp16 path to `compare_logits` and assert cosine ≈ 1. At
minimum, document that fp16 + Gemma on MPS is unvalidated.
**Where:** `src/run.py::pick_dtype`, `src/compare_logits.py`.

## T-2 · Parity gate must exercise the sliding window

`compare_logits` uses a ~6-token prompt; Gemma's window is 512, so local layers behave
like full causal and the band mask is **never tested** by the gate (only the offline
`*_selftest.py` cover it). A correct and a broken sliding implementation pass identically.

**Do:** add a long-prompt (>512 token) case to `compare_logits` / `sanity_test` for
gemma3 + gemma4, and ideally a prefill==incremental-decode assertion in the gate itself.
**Where:** `src/compare_logits.py`, `src/sanity_test.py`.

## T-3 · Stale `qwen3_5_selftest.py` reference

Docs reference `src/qwen3_5_selftest.py` (conv causality, recurrent==chunked, prefill==
decode, MTP shapes) but the file isn't in the repo. Either restore it (use
`gemma3_selftest.py` / `gemma4_selftest.py` as the pattern) or scrub the references.
**Where:** `docs/architecture.md`, `src/`.
