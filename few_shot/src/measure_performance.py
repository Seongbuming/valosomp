#!/usr/bin/env python3
"""
Performance measurement script for tweet evaluation models.
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
sys.path.insert(0, os.path.join(project_root, 'few_shot', 'src'))

from evaluate_tweets_gemma_fewshot import FewShotTweetEvaluator


def measure_performance_on_mturk(
    model_params=None,
    mturk_data_path="data/survey_data_ps_deduped.csv",
    sample_size=None,
    use_train_val_split=True
):
    """
    Measure model performance on MTurk data

    Args:
        model_params: Dict with k_shots, temperature, use_dimension_specific, etc.
        mturk_data_path: Path to MTurk data
        sample_size: Number of samples to evaluate (None = all)
        use_train_val_split: If True, use validation set only

    Returns:
        metrics: Performance metrics
    """
    if model_params is None:
        model_params = {
            'k_shots': 4,
            'temperature': 0.1,
            'use_dimension_specific': False
        }

    print("="*70)
    print("PERFORMANCE MEASUREMENT")
    print("="*70)
    print(f"Model parameters: {model_params}")
    print(f"MTurk data: {mturk_data_path}")
    print()

    # Initialize evaluator
    evaluator = FewShotTweetEvaluator(
        k_shots=model_params.get('k_shots', 4),
        temperature=model_params.get('temperature', 0.1),
        top_p=model_params.get('top_p', 1.0),
        use_dimension_specific=model_params.get('use_dimension_specific', False),
        gpu_id=model_params.get('gpu_id', 'auto')
    )

    # Get evaluation data
    if use_train_val_split:
        print("Creating train/val split to prevent data leakage...")
        train_df, val_df = evaluator.create_train_val_split(val_ratio=0.2)

        # Update evaluator to only use training data for few-shot examples
        evaluator.mturk_df = train_df
        evaluator.mturk_tweets = train_df['Tweet Text'].tolist()
        evaluator.mturk_embeddings = evaluator.embedding_model.encode(
            evaluator.mturk_tweets,
            show_progress_bar=True,
            convert_to_tensor=True
        )

        eval_df = val_df
        print(f"Using validation set: {len(eval_df)} examples")
    else:
        eval_df = pd.read_csv(mturk_data_path)
        print(f"Using full dataset: {len(eval_df)} examples")

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
    output_file = f"few_shot/data/performance_{timestamp}.json"

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
        'model_params': model_params,
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
        'use_train_val_split': use_train_val_split,
        'timestamp': datetime.now().isoformat(),
        'detailed_predictions': detailed_predictions  # Include raw scores
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Detailed results saved to: {output_file}")

    return metrics, results


def compare_configurations(configs, sample_size=50):
    """
    Compare multiple model configurations

    Args:
        configs: List of model parameter dicts
        sample_size: Number of samples for each evaluation
    """
    print("="*70)
    print("CONFIGURATION COMPARISON")
    print("="*70)
    print(f"Comparing {len(configs)} configurations on {sample_size} samples each")
    print()

    all_results = []

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Configuration: {config}")
        print("-"*70)

        metrics, results = measure_performance_on_mturk(
            model_params=config,
            sample_size=sample_size,
            use_train_val_split=True
        )

        all_results.append(results)

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Config':<40} {'MAE':<8} {'Pearson':<8} {'Within-1':<10}")
    print("-"*70)

    for result in all_results:
        config_str = f"k={result['model_params']['k_shots']}, " \
                    f"T={result['model_params']['temperature']}, " \
                    f"DS={result['model_params']['use_dimension_specific']}"
        mae = result['summary']['avg_mae']
        pearson = result['summary']['avg_pearson']
        within_1 = result['summary']['avg_within_1']

        print(f"{config_str:<40} {mae:<8.3f} {pearson:<8.3f} {within_1:<10.1%}")

    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"few_shot/data/comparison_{timestamp}.json"

    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nComparison saved to: {comparison_file}")
    print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Measure tweet evaluation performance")
    parser.add_argument(
        "--k-shots", type=int, default=4,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--dimension-specific", action="store_true",
        help="Use dimension-specific example selection"
    )
    parser.add_argument(
        "--gpu", type=str, default="auto",
        help="GPU to use: 'auto' (default), or specific GPU ID like '0', '1'"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of samples to evaluate (default: all validation set)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare multiple configurations"
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Don't use train/val split (may cause data leakage)"
    )

    args = parser.parse_args()

    if args.compare:
        # Compare multiple configurations
        configs = [
            {'k_shots': 2, 'temperature': 0.1, 'use_dimension_specific': False},
            {'k_shots': 4, 'temperature': 0.1, 'use_dimension_specific': False},
            {'k_shots': 4, 'temperature': 0.1, 'use_dimension_specific': True},
            {'k_shots': 8, 'temperature': 0.1, 'use_dimension_specific': False},
            {'k_shots': 8, 'temperature': 0.1, 'use_dimension_specific': True},
        ]
        compare_configurations(configs, sample_size=args.sample_size or 50)
    else:
        # Single configuration
        model_params = {
            'k_shots': args.k_shots,
            'temperature': args.temperature,
            'use_dimension_specific': args.dimension_specific,
            'gpu_id': args.gpu
        }

        measure_performance_on_mturk(
            model_params=model_params,
            sample_size=args.sample_size,
            use_train_val_split=not args.no_split
        )


if __name__ == "__main__":
    main()
