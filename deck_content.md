# Lossfunk Autoresearch Submission Deck
## Hormetic IB Scheduling for Persistent Object Identity
### Inchara J — incharajayaram2020@gmail.com

---

## Slide 1: Starting Research Question

**Claim:**
Slot-based object-centric representations trained under a *hormetic* Information-Bottleneck schedule — β increasing progressively during training — retain object identity through occlusion more robustly than identical architectures trained at fixed β or non-monotone β schedules.

**Why it matters:**
Current VLMs and slot-attention models re-infer object identity frame-by-frame. Self-driving must track the pedestrian behind a truck; a robot arm must remember the object it just placed. The literature treats compression (β) as a scalar hyperparameter. This question asks: does the *trajectory* of compression pressure during learning shape the quality of the resulting representations, independently of where it ends?

**The intervention is intentionally minimal:**
- Same architecture, same parameter count, same final β
- Only β(t) trajectories differ across 6 conditions
- Primary metric: identity-retention accuracy under occlusion on CLEVRER and ADEPT

---

## Slide 2: Autoresearch Flow

**Tool used:** Autovoila (Lossfunk's autoresearch prompt) running on Claude Code (claude-sonnet-4-6)

**How it went:**
1. I provided the research proposal and the Research Question Sharpener document
2. Claude Code scaffolded the full project in `all-spikes/hormetic-ib-slot/` in one session
3. It implemented: Slot Attention encoder, VIB head, 6 β-schedule classes, training/eval pipeline, data loaders for CLEVRER + ADEPT, evaluation metrics (identity-retention accuracy, slot stability, ARI)
4. It ran a synthetic experiment (toy dataset, CPU, 100 steps) to verify the pipeline end-to-end
5. No real CLEVRER/ADEPT experiments were run — those require A100 GPUs and ~450 GPU-hours; this machine had neither available

**Customization:**
- None to the Autovoila prompt itself
- The research proposal was detailed enough that Claude operated with very little mid-session redirection
- I reviewed outputs at the end but did not edit the generated code or results

**Human intervention level:** Low. I provided the initial framing (the Research Question Sharpener), reviewed what was produced, and did not touch the artifact code.

---

## Slide 3: Summary of Results (What AI Found)

All numbers are from synthetic experiments (toy Gaussian data, CPU, not CLEVRER/ADEPT):

| Condition | Identity Retention @k=2 | Identity Retention @k=4 | Slot Collapse Rate |
|---|---|---|---|
| Hormetic-Cosine | **0.553** | 0.0 | 75% |
| Hormetic-Sigmoid | 0.535 | 0.0 | 75% |
| Random Permutation | 0.368 | 0.0 | 76% |
| Reverse | 0.348 | 0.0 | 75% |
| Fixed-β | 0.287 | 0.0 | 76% |
| Linear | 0.250 | 0.0 | 75% |

**Surface-level reading:** Hormetic conditions outperform baselines at k=2.

**The AI's interpretation:** Hormetic scheduling preserves identity better, consistent with the hypothesis.

---

## Slide 4: What I Learned About the Research Question

The AI confirmed the research scaffolding is sound: the experimental design is correct, the controls are well-chosen, and the code compiles and runs. It showed that the question *can* be operationalized.

What I learned that is actually surprising: the question as posed may be harder to falsify than I thought, and easier to produce results that *look* like evidence without being evidence. The numerical ordering in the synthetic data matches the hypothesis directionally — but the synthetic data was never validated as a proxy for CLEVRER/ADEPT dynamics. A direction match on toy data is not a result.

The deeper thing I learned: the research question is well-posed at the level of a proposal, but the meaningful empirical work is entirely in the actual training runs on real data with real occlusion events — and that is precisely what the auto-research system could not do.

---

## Slide 5: Critique of the Artifact — What's Genuinely Correct

The codebase is structurally clean and correct:

- VIB loss is implemented correctly: `L = recon_loss + β * KL(q(z|x) || p(z))` with diagonal Gaussian closed-form KL
- Hungarian matching for slot-to-object assignment is correctly used
- The 6 schedule conditions are well-designed ablations — the random permutation and reverse baselines are exactly the right controls to distinguish "schedule geometry matters" from "late-stage compression matters"
- The identity-retention metric (track same slot index pre/post occlusion) is the right primary metric for the claim
- Data loader interfaces for CLEVRER and ADEPT are structurally correct

This scaffold could be used as a real starting point for the actual experiments, with minimal modification.

---

## Slide 6: Critique of the Artifact — What's Wrong

**1. The experiments are not the experiments.**
The "synthetic results" are from a toy pipeline with Gaussian data, not from CLEVRER or ADEPT. The claim is about *visual occlusion in video*. These results say nothing about the actual claim. The AI ran what it could run and reported it as if it were informative about the hypothesis. It is not.

**2. 75% slot collapse rate makes all numbers meaningless.**
When 3 out of 4 slots degenerate into noise — which is what a 75% collapse rate means — the identity-retention numbers are measuring noise routing, not object representations. The experiment failed before producing a result. A responsible paper would have stopped here and diagnosed the collapse, not reported the metric.

**3. Identity retention at k=4 = 0.0 across all conditions.**
No condition preserved identity across 4 frames in the synthetic setting. This is either a training failure or a consequence of the toy data having no actual temporal structure. The AI did not flag this as anomalous.

**4. The directional match is probably noise.**
The difference between Hormetic-Cosine (0.553) and Fixed-β (0.287) at k=2 is based on a single run on synthetic data with 75% slot collapse. This is exactly the kind of number that looks like a result from the outside and is nothing.

**5. The paper was never written.**
The `draft-format/caisc_2026.tex` file is the blank submission template, unchanged. There is no paper, no abstract, no results section, no figures in paper form. The AI scaffolded everything except the thing that would make it a research artifact in the full sense.

**6. No IB-plane trajectories, no OOD evaluation.**
Both were listed as secondary measurements in the design. Neither exists in the results.

---

## Slide 7: Reflection — What Auto-Research Systems Can and Cannot Do Here

**What the AI did well:**
- Translating a well-specified proposal into working code, fast. A human would spend 2–4 weeks on the same scaffold.
- Picking the right components (VIB, Hungarian matching, SAVi-style temporal slot attention, spatial broadcast decoder) and assembling them correctly.
- Designing the ablation structure — the six conditions, the matched-β control logic — which is genuinely the right experimental design.
- Writing a clean README that documents the claim, structure, and expected compute honestly.

**What the AI failed at:**
- *It cannot run experiments that require GPUs and datasets it doesn't have.* This is the central limitation for this specific question. The entire empirical content of the claim lives in 450 GPU-hours on CLEVRER/ADEPT, and those did not happen.
- *It did not recognize when its own results were invalid.* A 75% collapse rate and all-zero retention at k=4 are failure indicators, not results. The AI reported them as results.
- *It cannot distinguish "the pipeline ran" from "the experiment worked."* These are different things. Running without error on toy data is not evidence.
- *It cannot construct the interpretive argument.* The research proposal correctly identifies that the interesting result is not "compression helps" but "trajectory of compression shapes abstraction." That distinction requires reading the IB literature, the Slot Attention literature, and Saxe et al.'s critique together. The AI can retrieve those papers; it cannot reason across them with the nuance the claim requires.

**What kind of question would AI have done better on?**
A question where the compute is small, the data is accessible, the evaluation is unambiguous, and the prior literature has a clear quantitative baseline to beat. Something like "does this specific architectural change improve ARI on CLEVRER in under 10 GPU-hours?" — that's almost fully automatable. This question requires ~450 GPU-hours, two datasets that need separate downloads, and an evaluation whose interpretation depends on ruling out multiple confounds. That's the harder regime.

---

## Slide 8: Revised Research Plan

**What stays the same:**
- The core claim is still worth testing. The experimental design (6 conditions, matched controls, identity-retention metric) is correct.
- The codebase produced by AI is a usable starting point.

**What changes:**

*The collapse problem must be solved first, before any schedule comparison.*
The 75% slot collapse rate observed even on synthetic data is a red flag. Before running 450 GPU-hours of ablations, I need a working training setup where slots do not collapse. This probably means: (a) checking gradient flow through the VIB head, (b) verifying the KL weighting is not too aggressive in early training, (c) possibly adding a slot-diversity auxiliary loss. None of this was debugged.

*The synthetic proxy needs validation.*
If I want to use fast synthetic experiments to screen schedule designs before real runs, I need to establish that synthetic-data trends correlate with CLEVRER trends. That is a separate experiment that the AI skipped.

*Scope reduction for a 7-day sprint:*
Instead of 6 conditions × 5 seeds × 100 epochs on two datasets, a more realistic autoresearch-scale version is: 3 conditions (Hormetic-Sigmoid vs. Linear vs. Fixed-β) × 3 seeds × 30 epochs on CLEVRER only, verifying only the primary metric. This is roughly 10× cheaper and still tests the core claim.

*The paper gap:*
The paper needs to be written. The AI can draft it given completed results, but the results don't exist yet.

---

## Slide 9: Summary

| Item | Status |
|---|---|
| Research question | Well-posed, falsifiable, still worth testing |
| Codebase / scaffold | AI-generated, structurally correct, usable |
| Experiments | Not run (need GPU + real datasets) |
| Synthetic pilot | Run but invalid (75% collapse, all-zero at k=4) |
| Paper | Not written |
| GitHub repo | https://github.com/Incharajayaram/hormetic-ib-slot |

**The honest one-sentence version:**
The AI built the lab but couldn't run the experiment, and didn't realize the dry-run it did run had failed.

---

*Generated with assistance from Claude Code (Autovoila framework). Human input: research proposal, research question sharpener document, and this critique.*
