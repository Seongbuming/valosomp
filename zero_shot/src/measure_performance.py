#!/usr/bin/env python3
"""
Performance measurement script for zero-shot tweet evaluation.
Compares model predictions with human annotations from MTurk.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'zero_shot', 'src'))

from evaluate_tweets_gemma import TweetEvaluator


def measure_performance_on_mturk(
    mturk_data_path="data/survey_data_ps_deduped.csv",
    sample_size=None,
    gpu_id=None
):
    """
    Measure model performance on MTurk data

    Args:
        mturk_data_path: Path to MTurk data
        sample_size: Number of samples to evaluate (None = all)
        gpu_id: GPU to use (None = default)

    Returns:
        metrics: Performance metrics
    """
    print("="*70)
    print("ZERO-SHOT PERFORMANCE MEASUREMENT")
    print("="*70)
    print(f"MTurk data: {mturk_data_path}")
    print()

    # Initialize evaluator
    evaluator = TweetEvaluator(gpu_id=gpu_id)

    # Load MTurk data
    eval_df = pd.read_csv(mturk_data_path)
    print(f"Loaded {len(eval_df)} examples")

    # Sample if requested
    if sample_size and sample_size < len(eval_df):
        eval_df = eval_df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} examples for evaluation")

    print()

    # Evaluate on dataset
    val_results = evaluator.evaluate_on_validation(eval_df)

    # Compute metrics
    metrics = evaluator.compute_metrics(val_results)

    # Print results
    avg_mae, avg_pearson, avg_within_1, avg_level = evaluator.print_metrics(metrics)

    # Save detailed results including raw scores
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"zero_shot/data/performance_{timestamp}.json"

    # Extract sample-level data for analysis
    detailed_predictions = []
    for result in val_results:
        detailed_predictions.append({
            'tweet_id': result['tweet_id'],
            'tweet_text': result['tweet_text'],
            'human_scores': result['human_scores'],
            'model_scores': result['model_scores']
        })

    results = {
        'model_params': {'method': 'zero-shot'},
        'metrics': {
            dim: {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                  for k, v in m.items()}
            for dim, m in metrics.items()
        },
        'summary': {
            'avg_mae': float(avg_mae),
            'avg_pearson': float(avg_pearson),
            'avg_within_1': float(avg_within_1),
            'avg_level_acc': float(avg_level)
        },
        'num_samples': len(eval_df),
        'timestamp': datetime.now().isoformat(),
        'detailed_predictions': detailed_predictions  # Include raw scores
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Detailed results saved to: {output_file}")

    return metrics, results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Measure zero-shot tweet evaluation performance")
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU to use: specific GPU ID like '0', '1' (default: uses default GPU)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of samples to evaluate (default: all)"
    )

    args = parser.parse_args()

    measure_performance_on_mturk(
        sample_size=args.sample_size,
        gpu_id=args.gpu
    )


if __name__ == "__main__":
    main()
