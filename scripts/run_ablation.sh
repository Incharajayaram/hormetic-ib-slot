#!/bin/bash
# Run full schedule ablation across 5 seeds
# Expected total: ~450 GPU-hours on A100 (all 6 conditions × 5 seeds × ~15h each)
# Estimated cost: ~$60-90 at typical A100 spot rates

set -e

DEVICE=${DEVICE:-"cuda:0"}
BASE_CFG="configs/base.yaml"
EXP_DIR="configs/experiments"
RESULTS="results"
NUM_SEEDS=5

SCHEDULES=(
    "hormetic_sigmoid"
    "hormetic_cosine"
    "linear"
    "reverse"
    "random_permutation"
    "fixed_beta"
)

mkdir -p "$RESULTS"

echo "=== Hormetic IB Slot Schedule Ablation ==="
echo "Device: $DEVICE"
echo "Seeds: 0 to $((NUM_SEEDS-1))"
echo "Schedules: ${SCHEDULES[*]}"
echo ""

for schedule in "${SCHEDULES[@]}"; do
    for seed in $(seq 0 $((NUM_SEEDS-1))); do
        echo ">>> Running: $schedule | seed=$seed"
        python scripts/train.py \
            --config "$BASE_CFG" \
            --experiment "$EXP_DIR/${schedule}.yaml" \
            --seed "$seed" \
            --device "$DEVICE" \
            2>&1 | tee -a "$RESULTS/${schedule}_seed${seed}.log"
        echo "<<< Done: $schedule | seed=$seed"
        echo ""
    done
done

echo "=== All training runs complete. Running evaluation... ==="

python scripts/evaluate.py \
    --results_dir "$RESULTS" \
    --device "$DEVICE" \
    --clevrer "${CLEVRER_ROOT:-/data/clevrer}" \
    --adept "${ADEPT_ROOT:-/data/adept}" \
    --output "$RESULTS/ablation_results.json"

echo "=== Evaluation complete. Results in $RESULTS/ablation_results.json ==="

# Generate summary table and plots
python scripts/analyze_results.py --results "$RESULTS/ablation_results.json"
