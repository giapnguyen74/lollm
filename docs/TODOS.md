# TODOs — near-term, actionable

Things we intend to fix **soon** (concrete, scoped). Longer-term / untimed ideas live in
[ROADMAP.md](./ROADMAP.md); resolved gotchas become lessons in
[LESSONS.md](./LESSONS.md).

**IDs are stable** — `T-#` never renumbers or gets reused. When an item is done, flip its
status to ✅ (leave it in place for one cycle, then move the lesson to LESSONS.md and
delete the row). Status: 🔴 open · 🟡 in progress · ✅ done · 📌 acknowledged (accepted, no fix planned).

| ID  | status | area    | title |
|-----|--------|---------|-------|
| T-1 | 📌 acknowledged | mps / dtype | fp16-on-MPS: fp16 default kept (bf16 rejected, L-5); `--dtype` override **not shipped** |
| T-2 | 🔴 open | parity      | Add a >sliding_window prompt to the parity gate (window is untested) |

---

## T-1 · fp16-on-MPS dtype 📌

Real inference runs **fp16 on MPS**, but the parity gate runs **fp32 on CPU**, so a
fp16-only overflow would be invisible to the gate. Gemma has the largest activations
(×√hidden embedding, wide GeGLU).

**Decision (made):** keep **fp16 as the MPS default** — in practice it's near-identical
to fp32 because the overflow-prone reductions (attention softmax, every RMSNorm) already
upcast to fp32. bf16-on-MPS was tried and rejected (~3× slower, memory-heavy — see L-5).


## T-2 · Parity gate must exercise the sliding window

`compare_logits` uses a ~6-token prompt; Gemma's window is 512, so local layers behave
like full causal and the band mask is **never tested** by the gate. A correct and a
broken sliding implementation pass identically.

**Do:** add a long-prompt (>512 token) case to `compare_logits` / `sanity_test` for
gemma3 + gemma4, and ideally a prefill==incremental-decode assertion in the gate itself.
**Where:** `src/compare_logits.py`, `src/sanity_test.py`.
