#!/bin/bash
# Run few-shot experiments with different k values

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=========================================="
echo "Few-Shot Tweet Evaluation Experiments"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$PROJECT_ROOT/data/evaluations"

# Run experiments for k=1, 4, 8, 16
# for k in 1 4 8 16; do
for k in 16; do
    echo "=========================================="
    echo "Running ${k}-shot evaluation..."
    echo "=========================================="

    python "$PROJECT_ROOT/src/evaluate_tweets_gemma_fewshot.py" \
        --k-shots $k \
        --input data/survey_data_ps_deduped.csv \
        --output "$PROJECT_ROOT/data/evaluations/gemma_k${k}_$(date +%Y%m%d_%H%M%S).json" \
        --gpu 1

    echo ""
    echo "Completed ${k}-shot evaluation"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in $PROJECT_ROOT/data/evaluations/"
echo "=========================================="
