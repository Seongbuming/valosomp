#!/bin/bash
# Run full experiments on MTurk dataset for both zero-shot and few-shot

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Parse start step argument
START_STEP=${1:-1}

echo "=========================================="
echo "Full Dataset Experiment"
echo "MTurk Survey Data Evaluation"
echo "=========================================="
echo ""
echo "Available steps:"
echo "  1. Zero-Shot Evaluation (k=0)"
echo "  2. Few-Shot Evaluations (k=1,4,8,16)"
echo "  3. Human-LLM Agreement Analysis"
echo ""
echo "Starting from step: $START_STEP"
echo ""

# MTurk data has 250 tweets
MTURK_DATA="$PROJECT_ROOT/data/mturk_survey_data_deduped.csv"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Check if MTurk data exists
if [ ! -f "$MTURK_DATA" ]; then
    echo "Error: MTurk data not found at $MTURK_DATA"
    exit 1
fi

echo "Input: $MTURK_DATA"
echo "Timestamp: $TIMESTAMP"
echo ""

# ==========================================
# 1. Zero-Shot Evaluation
# ==========================================
if [ $START_STEP -le 1 ]; then
    echo "=========================================="
    echo "1. Zero-Shot Evaluation (k=0)"
    echo "=========================================="

    python "$PROJECT_ROOT/zero_shot/src/evaluate_tweets_gemma.py" \
        --input "$MTURK_DATA" \
        --output "$PROJECT_ROOT/zero_shot/data/evaluations/mturk_full_zeroshot_${TIMESTAMP}.json"

    echo "✅ Zero-shot evaluation complete"
    echo ""
else
    echo "Skipping step 1: Zero-Shot Evaluation"
    echo ""
fi

# ==========================================
# 2. Few-Shot Evaluations (k=1,4,8,16)
# ==========================================
if [ $START_STEP -le 2 ]; then
    echo "=========================================="
    echo "2. Few-Shot Evaluations"
    echo "=========================================="

    for k in 1 4 8 16; do
        echo "Running ${k}-shot evaluation..."

        python "$PROJECT_ROOT/few_shot/src/evaluate_tweets_gemma_fewshot.py" \
            --k-shots $k \
            --input "$MTURK_DATA" \
            --output "$PROJECT_ROOT/few_shot/data/evaluations/mturk_full_k${k}_${TIMESTAMP}.json"

        echo "✅ ${k}-shot complete"
        echo ""
    done
else
    echo "Skipping step 2: Few-Shot Evaluations"
    echo ""
fi

# ==========================================
# 3. Human-LLM Agreement Analysis
# ==========================================
if [ $START_STEP -le 3 ]; then
    echo "=========================================="
    echo "3. Human-LLM Agreement Analysis"
    echo "=========================================="

    # Find the most recent evaluation files
    ZEROSHOT_CSV=$(ls -t "$PROJECT_ROOT/zero_shot/data/evaluations/mturk_full_zeroshot_"*.csv 2>/dev/null | head -1)
    K1_CSV=$(ls -t "$PROJECT_ROOT/few_shot/data/evaluations/mturk_full_k1_"*.csv 2>/dev/null | head -1)
    K4_CSV=$(ls -t "$PROJECT_ROOT/few_shot/data/evaluations/mturk_full_k4_"*.csv 2>/dev/null | head -1)
    K8_CSV=$(ls -t "$PROJECT_ROOT/few_shot/data/evaluations/mturk_full_k8_"*.csv 2>/dev/null | head -1)
    K16_CSV=$(ls -t "$PROJECT_ROOT/few_shot/data/evaluations/mturk_full_k16_"*.csv 2>/dev/null | head -1)

    echo "Using evaluation files:"
    echo "  Zero-shot: $ZEROSHOT_CSV"
    echo "  k=1: $K1_CSV"
    echo "  k=4: $K4_CSV"
    echo "  k=8: $K8_CSV"
    echo "  k=16: $K16_CSV"
    echo ""

    echo "Analyzing zero-shot..."
    python "$PROJECT_ROOT/zero_shot/src/analyze_human_llm_agreement.py" \
        --mturk-data "$MTURK_DATA" \
        --llm-results "$ZEROSHOT_CSV" \
        --k-values 0 \
        --output-dir "$PROJECT_ROOT/zero_shot/figures"

    echo "Analyzing few-shot (all k values)..."
    python "$PROJECT_ROOT/few_shot/src/analyze_human_llm_agreement.py" \
        --mturk-data "$MTURK_DATA" \
        --llm-results "$K1_CSV" "$K4_CSV" "$K8_CSV" "$K16_CSV" \
        --k-values 1 4 8 16 \
        --output-dir "$PROJECT_ROOT/few_shot/figures"

    echo "✅ Agreement analysis complete"
    echo ""
else
    echo "Skipping step 3: Human-LLM Agreement Analysis"
    echo ""
fi

# ==========================================
# 4. Summary
# ==========================================
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved:"
echo "  Zero-shot: $PROJECT_ROOT/zero_shot/data/evaluations/"
echo "  Few-shot:  $PROJECT_ROOT/few_shot/data/evaluations/"
echo ""
echo "Figures saved:"
echo "  Zero-shot: $PROJECT_ROOT/zero_shot/figures/"
echo "  Few-shot:  $PROJECT_ROOT/few_shot/figures/"
echo ""
echo "Next steps:"
echo "  1. Review scatter_human_vs_llm_k*.png for each k"
echo "  2. Compare correlation_by_k.png"
echo "  3. Check human_llm_agreement.csv for metrics"
echo "=========================================="
