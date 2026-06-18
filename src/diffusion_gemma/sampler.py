"""
sampler.py — diffusion_gemma denoising sampler (Entropy-Bound + linear temperature).

Ported 1:1 from transformers `generation_diffusion_gemma.py` (EntropyBoundSampler +
LinearTemperatureScheduleLogitsProcessor) so a learner can read the diffusion sampling rule
next to the model. The denoise *loop* (rungs 4–5) drives these; here are the per-step pieces.

The mechanism ("Uniform State Diffusion"):
  • canvas starts as RANDOM tokens (not a mask token);
  • each step the denoiser predicts a distribution per position;
  • accept the lowest-entropy positions whose mutual-information bound stays ≤ `entropy_bound`
    (they take the freshly sampled token); RE-RANDOMISE the rest for the next step.
"""
from __future__ import annotations

import torch


def linear_temperature(scores: torch.Tensor, cur_step: int, t_min: float, t_max: float,
                       max_denoising_steps: int) -> torch.Tensor:
    """`scores / t`, where t = t_min + (t_max-t_min)·(cur_step/N). cur_step counts DOWN
    (steps remaining), so t runs t_max → t_min across the schedule."""
    t = t_min + (t_max - t_min) * (cur_step / max_denoising_steps)
    return scores / t


class EntropyBoundSampler:
    """Accept low-entropy tokens up to a mutual-information bound; re-randomise the rest."""

    def __init__(self, entropy_bound: float, canvas_length: int, vocab_size: int):
        self.entropy_bound = entropy_bound
        self.canvas_length = canvas_length
        self.vocab_size = vocab_size
        self.accepted_token_mask = None        # set by accept_canvas, read by renoise_canvas

    def initialize_canvas(self, batch_size: int, device) -> torch.Tensor:
        """A fresh canvas = uniform-random token ids (Uniform State Diffusion)."""
        return torch.randint(0, self.vocab_size, (batch_size, self.canvas_length), device=device)

    def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step=None):
        # per-position entropy of the predicted distribution
        token_entropy = torch.distributions.Categorical(logits=logits).entropy()   # (B, L)
        sorted_e, sorted_idx = torch.sort(token_entropy, dim=-1, descending=False)
        cumulative = torch.cumsum(sorted_e, dim=-1)
        # accept while (sum of strictly-smaller entropies) ≤ bound → approx-independent tokens
        sorted_sel = (cumulative - sorted_e) <= self.entropy_bound
        self.accepted_token_mask = torch.scatter(
            torch.zeros_like(sorted_sel), dim=-1, index=sorted_idx, src=sorted_sel)
        # accepted positions take the new sampled token; others keep the current canvas
        return torch.where(self.accepted_token_mask, denoiser_canvas, current_canvas)

    def renoise_canvas(self, accepted_canvas, cur_step=None):
        random_canvas = self.initialize_canvas(accepted_canvas.shape[0], accepted_canvas.device)
        return torch.where(~self.accepted_token_mask, random_canvas, accepted_canvas)


class StableAndConfidentStopping:
    """Adaptive stop: finished when the argmax canvas is unchanged for `stability_threshold`
    steps AND the mean per-position entropy is below `confidence_threshold`. Ported 1:1."""

    def __init__(self, stability_threshold: int, confidence_threshold: float):
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.argmax_canvas_history = None

    def reset(self):
        self.argmax_canvas_history = None

    def __call__(self, argmax_canvas, logits):
        # stability — compare against the last `stability_threshold` argmax canvases
        if self.stability_threshold == 0:
            stable = torch.ones(logits.shape[0], device=logits.device, dtype=torch.bool)
        else:
            if self.argmax_canvas_history is None:
                self.argmax_canvas_history = torch.full(
                    (self.stability_threshold, *argmax_canvas.shape), -1,
                    dtype=argmax_canvas.dtype, device=argmax_canvas.device)
            stable = (self.argmax_canvas_history == argmax_canvas[None]).all(dim=-1).all(dim=0)
            self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, -1, dims=0)
            self.argmax_canvas_history[-1] = argmax_canvas
        # confidence — mean entropy of the (temperature-scaled) logits
        entropy = torch.distributions.Categorical(logits=logits).entropy()
        confident = entropy.mean(dim=-1) < self.confidence_threshold
        return stable & confident


def denoise_block(forward_logits, sampler: EntropyBoundSampler, stop, *, max_denoising_steps,
                  t_min, t_max, batch_size, device, sample=True):
    """One block = inner denoise loop. `forward_logits(canvas, self_cond) -> softcapped logits`.
    Returns the finished block (the last step's argmax canvas). Mirrors transformers `generate`'s
    inner loop: cur_step = N..1, accept → renoise, carry processed logits as next self-cond, stop
    early when all batch items are finished. `sample=False` = greedy denoiser (deterministic)."""
    current = sampler.initialize_canvas(batch_size, device)
    argmax = current
    self_cond = None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if stop is not None:
        stop.reset()
    for cur_step in reversed(range(1, max_denoising_steps + 1)):
        raw = forward_logits(current, self_cond)
        proc = linear_temperature(raw, cur_step, t_min, t_max, max_denoising_steps)
        B, L, V = proc.shape
        if sample:
            probs = torch.softmax(proc, dim=-1, dtype=torch.float32)
            denoiser = torch.multinomial(probs.view(-1, V), 1).squeeze(-1).view(B, L)
        else:
            denoiser = proc.argmax(dim=-1)
        argmax = proc.argmax(dim=-1)
        accepted = sampler.accept_canvas(current, denoiser, proc, cur_step)
        current = sampler.renoise_canvas(accepted, cur_step)
        if stop is not None:
            finished = finished | stop(argmax, proc)
            if torch.all(finished):
                break
        self_cond = proc
    return argmax
