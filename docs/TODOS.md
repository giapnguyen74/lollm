# TODOs — near-term, actionable

Things we intend to fix **soon** (concrete, scoped). Longer-term / untimed ideas live in
[ROADMAP.md](./ROADMAP.md); resolved gotchas become lessons in
[LESSONS.md](./LESSONS.md).

**IDs are stable** — `T-#` never renumbers or gets reused. When an item is done, flip its
status to ✅ (leave it in place for one cycle, then move the lesson to LESSONS.md and
delete the row). Status: 🔴 open · 🟡 in progress · ✅ done · 📌 acknowledged (accepted, no fix planned).

| ID  | status | area    | title |
|-----|--------|---------|-------|
| T-1 | 📌 acknowledged | mps / dtype | fp16-on-MPS overflow risk the fp32 gate can't see (accepted) |
| T-2 | 🔴 open | parity      | Add a >sliding_window prompt to the parity gate (window is untested) |
| T-3 | ✅ done | docs        | Removed stale `qwen3_5_selftest.py` doc references |

---

## T-1 · fp16-on-MPS overflow risk (acknowledged) 📌

Real inference runs **fp16 on MPS**, but the parity gate runs **fp32 on CPU**, so a
fp16-only overflow would be invisible to the gate. Gemma has the largest activations
(×√hidden embedding, wide GeGLU).

**Decision: accept the risk, no code change.** In practice fp16 is near-identical to
fp32 here — the overflow-prone reductions (attention softmax, every RMSNorm) already
upcast to fp32 internally, so plain fp16 stays well within range. bf16-on-MPS was tried
and rejected (~3× slower, memory-heavy — see L-5). The escape hatch if a future model
*does* overflow is `--dtype fp32`. The general guard is the checklist's **real-device run**
(CONVENTIONS §5): the fp32 gate proves the math; a real fp16 run on the device confirms
the runtime path. Kept here as an acknowledged risk, not an action.

## T-2 · Parity gate must exercise the sliding window

`compare_logits` uses a ~6-token prompt; Gemma's window is 512, so local layers behave
like full causal and the band mask is **never tested** by the gate. A correct and a
broken sliding implementation pass identically.

**Do:** add a long-prompt (>512 token) case to `compare_logits` / `sanity_test` for
gemma3 + gemma4, and ideally a prefill==incremental-decode assertion in the gate itself.
**Where:** `src/compare_logits.py`, `src/sanity_test.py`.

## T-3 · Stale `qwen3_5_selftest.py` reference ✅

Docs referenced `src/qwen3_5_selftest.py`, which doesn't exist. Per preference — **don't
cite ad-hoc test/debug scripts in the docs** — the references were removed (from
`architecture.md`, `gemma4-architecture.md`, `LESSONS.md`, `ROADMAP.md`, `README.md`,
`CONVENTIONS.md`) rather than recreating the script. The canonical `compare_logits` /
`sanity_test` / `run.py` workflow stays.
