# Hormetic IB Slot — Codebase

Empirical study of whether the **schedule of β compression pressure** during training affects object-permanence in slot-based object-centric representations.

**Claim:** Slot attention models trained under a hormetic (progressively increasing) β-IB schedule retain object identity through occlusion more robustly than models trained at fixed β or under non-monotone β schedules.

---

## Quick start

```bash
# Install
pip install -e .
# or
pip install -r requirements.txt

# Check schedule shapes before running anything
python -c "from hormetic_ib_slot.schedules.beta_schedules import plot_schedules; plot_schedules(save_path='schedules.png')"

# Debug run (2 epochs, 20 videos, CPU)
python scripts/train.py \
    --config configs/base.yaml \
    --experiment configs/experiments/hormetic_sigmoid.yaml \
    --debug

# Full training run (requires GPU, CLEVRER data)
export CLEVRER_ROOT=/data/clevrer
export ADEPT_ROOT=/data/adept

python scripts/train.py \
    --config configs/base.yaml \
    --experiment configs/experiments/hormetic_sigmoid.yaml \
    --seed 0 \
    --device cuda:0

# Run full ablation (all 6 schedules × 5 seeds)
chmod +x scripts/run_ablation.sh
DEVICE=cuda:0 CLEVRER_ROOT=/data/clevrer ADEPT_ROOT=/data/adept bash scripts/run_ablation.sh
```

---

## Project structure

```
hormetic-ib-slot/
├── src/hormetic_ib_slot/
│   ├── schedules/
│   │   └── beta_schedules.py     # 6 β schedule classes + registry
│   ├── models/
│   │   ├── backbone.py           # SmallCNN, ResNetBackbone, PositionEmbedding
│   │   ├── slot_attention.py     # SlotAttention, TemporalSlotAttention (SAVi-style)
│   │   ├── vib.py                # VIBHead, VIBPredictionHead
│   │   ├── decoder.py            # SpatialBroadcastDecoder
│   │   └── model.py              # HormeticIBSlot (full model)
│   ├── training/
│   │   ├── losses.py             # VIB loss, Hungarian matching, identity loss
│   │   └── trainer.py            # Trainer class (CSV logging, checkpointing)
│   ├── data/
│   │   ├── clevrer.py            # CLEVRER video dataset + loader
│   │   └── adept.py              # ADEPT object permanence dataset + loader
│   ├── evaluation/
│   │   ├── identity_retention.py # Primary metric: identity retention accuracy
│   │   └── slot_stability.py     # ARI, cosine stability, collapse rate
│   └── utils/
│       ├── logging.py            # CSVLogger, setup_experiment_dir
│       ├── visualization.py      # Slot decomposition plots
│       └── checkpoint.py         # save/load checkpoint
├── configs/
│   ├── base.yaml                 # Default hyperparameters
│   └── experiments/              # One file per schedule condition
│       ├── hormetic_sigmoid.yaml
│       ├── hormetic_cosine.yaml
│       ├── linear.yaml
│       ├── reverse.yaml
│       ├── random_permutation.yaml
│       └── fixed_beta.yaml
└── scripts/
    ├── train.py                  # Training entry point
    ├── evaluate.py               # Evaluation entry point
    ├── run_ablation.sh           # Full ablation sweep
    └── analyze_results.py        # Aggregate results + figures
```

---

## Experimental conditions

| Condition | Schedule | Role |
|---|---|---|
| **Hormetic-Sigmoid** | β(t) = β_max · σ(k(t−t₀)), normalised | Primary intervention |
| **Hormetic-Cosine** | β(t) = β_max · 0.5 · (1−cos(πt)) | Secondary hormetic control |
| **Linear** | β(t) = β_max · t | Monotone, non-sigmoidal control |
| **Reverse** | β(t) = β_max · (1−t) | Anti-hormetic (starts compressed) |
| **Random Perm.** | Same values as Linear, shuffled | Tests whether order matters |
| **Fixed-β** | β(t) = β_max | Standard VIB/β-VAE baseline |

All conditions share:
- Identical architecture
- Identical parameter count
- Same β_max at training end
- Same random seeds (5 seeds per condition)

---

## Data setup

### CLEVRER
```
clevrer_root/
  videos/train/video_XXXXX-XXXXX/video_XXXXX.mp4
  videos/val/...
  annotations/train/annotation_XXXXX.json
  annotations/val/...
```
Download: [http://clevrer.csail.mit.edu/](http://clevrer.csail.mit.edu/)

### ADEPT
```
adept_root/
  sequences/seq_XXXXXX/
    frames/frame_XXXXXX.png
    metadata.json
```
Download: [https://github.com/google-deepmind/adept](https://github.com/google-deepmind/adept)

Update `configs/base.yaml` paths or set env vars:
```bash
export CLEVRER_ROOT=/path/to/clevrer
export ADEPT_ROOT=/path/to/adept
```

---

## Primary metric: Identity-Retention Accuracy

For each video with annotated object tracks:
1. Before occlusion (frame t): assign slots to ground-truth objects via Hungarian matching on slot attention masks.
2. Run model through k frames of occlusion.
3. After re-emergence (frame t+k): re-assign slots to objects.
4. Accuracy = fraction of objects where the same slot index corresponds to the same object before and after.

Measured at occlusion durations k ∈ {4, 8, 16, 32} frames.

---

## Expected compute

| Component | Count | Est. time (A100) |
|---|---|---|
| Conditions | 6 | — |
| Seeds per condition | 5 | — |
| Training per run | 100 epochs | ~15h |
| **Total training** | 30 runs | **~450 GPU-hours** |
| Evaluation | all runs | ~5h |

Estimated cost at A100 spot: ~$60–90.

---

## Outputs

Each run writes to `results/<schedule_name>/seed_<N>/`:
- `config.yaml` — merged config
- `logs.csv` — per-step: step, epoch, beta, total_loss, recon_loss, kl_loss, identity_loss
- `checkpoint_best.pt` — best validation checkpoint
- `checkpoint_latest.pt` — latest checkpoint

After evaluation:
- `results/ablation_results.json` — all metrics
- `results/figures/identity_retention_by_duration.pdf`
- `results/figures/mean_accuracy_bar.pdf`
- `results/figures/summary_table.md`

---

## Citation context

This codebase accompanies the research proposal:

> "Hormetic Information-Bottleneck Training Improves Persistent Object Identity
> in Slot-Based Object-Centric Representations"

Key prior work:
- Locatello et al. (2020) — Object-Centric Learning with Slot Attention (NeurIPS)
- Kipf et al. (2021) — Conditional Object-Centric Learning from Video (ICLR) [SAVi]
- Alemi et al. (2017) — Deep Variational Information Bottleneck (ICLR)
- Chen et al. (2020) — CLEVRER: CoLlision Events for Video REpresentation (ICLR)
- Kortmann et al. (2022) — ADEPT: A Dataset for Object Permanence (NeurIPS)
