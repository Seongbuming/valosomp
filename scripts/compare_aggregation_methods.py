#!/usr/bin/env python3
"""
Compare different aggregation methods for combining 3 annotator scores.
Methods: median (current), mean, max, majority voting
"""

import json
import sys
import numpy as np
from collections import Counter

def score_to_2level(score):
    """Convert score (1-7) to 2-level (low/high)"""
    if score <= 4:
        return 'low'
    else:  # 5-7
        return 'high'

def aggregate_median(scores):
    """Current method: median of 3 scores"""
    return np.median(scores)

def aggregate_mean(scores):
    """Mean of 3 scores (rounded)"""
    return round(np.mean(scores))

def aggregate_max(scores):
    """Max of 3 scores (most sensitive to high)"""
    return max(scores)

def aggregate_majority_voting(scores):
    """Majority voting at level, then use median score within winning level"""
    # Convert to levels
    levels = [score_to_2level(s) for s in scores]

    # Count votes
    level_counts = Counter(levels)

    # Get majority level
    majority_level = level_counts.most_common(1)[0][0]

    # If tie (shouldn't happen with 3 annotators), use median
    if len(level_counts) == 2 and level_counts['low'] == level_counts['high']:
        return np.median(scores)

    # Return median score from the majority level
    scores_in_majority = [s for s in scores if score_to_2level(s) == majority_level]
    return np.median(scores_in_majority)

def analyze_with_aggregation(json_path, aggregation_fn, method_name):
    """Calculate 2-level accuracy with given aggregation method"""

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Collect predictions by dimension
    results_by_dim = {
        'I-Involvement': {'correct': 0, 'total': 0},
        'You-Involvement': {'correct': 0, 'total': 0},
        'We-Involvement': {'correct': 0, 'total': 0}
    }

    confusion = {'low': {'low': 0, 'high': 0}, 'high': {'low': 0, 'high': 0}}

    for pred in data['detailed_predictions']:
        for dim in ['I-Involvement', 'You-Involvement', 'We-Involvement']:
            human = pred['human_scores'][dim]
            model = pred['model_scores'][dim]

            # Apply aggregation
            human_agg = aggregation_fn(human)
            model_agg = aggregation_fn(model)

            human_level = score_to_2level(human_agg)
            model_level = score_to_2level(model_agg)

            # Track accuracy
            results_by_dim[dim]['total'] += 1
            if human_level == model_level:
                results_by_dim[dim]['correct'] += 1

            # Track confusion
            confusion[human_level][model_level] += 1

    # Calculate overall accuracy
    total_correct = sum(d['correct'] for d in results_by_dim.values())
    total = sum(d['total'] for d in results_by_dim.values())
    overall_acc = total_correct / total if total > 0 else 0

    return {
        'method': method_name,
        'overall_accuracy': overall_acc,
        'by_dimension': {
            dim: results_by_dim[dim]['correct'] / results_by_dim[dim]['total']
            for dim in results_by_dim
        },
        'confusion': confusion,
        'total_correct': total_correct,
        'total': total
    }

def compare_all_methods(json_path):
    """Compare all aggregation methods on a single results file"""

    with open(json_path, 'r') as f:
        data = json.load(f)

    print("="*70)
    print("AGGREGATION METHOD COMPARISON")
    print("="*70)
    print(f"File: {json_path}")
    print(f"Model: {data.get('model_params', 'N/A')}")
    print()

    # Test all methods
    methods = [
        (aggregate_median, "Median (current)"),
        (aggregate_mean, "Mean"),
        (aggregate_max, "Max"),
        (aggregate_majority_voting, "Majority Voting")
    ]

    results = []
    for agg_fn, method_name in methods:
        result = analyze_with_aggregation(json_path, agg_fn, method_name)
        results.append(result)

    # Print comparison table
    print("📊 OVERALL ACCURACY BY METHOD")
    print("-"*70)
    print(f"{'Method':<20} {'Accuracy':>10} {'Improvement':>12}")
    print("-"*70)

    baseline_acc = results[0]['overall_accuracy']  # median

    for result in results:
        acc = result['overall_accuracy']
        improvement = acc - baseline_acc
        marker = " ⭐" if acc == max(r['overall_accuracy'] for r in results) else ""

        print(f"{result['method']:<20} {acc:>9.1%} {improvement:>+11.1%}{marker}")

    print()

    # Print per-dimension breakdown for best method
    best_result = max(results, key=lambda r: r['overall_accuracy'])

    print(f"📊 BEST METHOD: {best_result['method']} ({best_result['overall_accuracy']:.1%})")
    print("-"*70)
    print("Per-dimension accuracy:")
    for dim in ['I-Involvement', 'You-Involvement', 'We-Involvement']:
        acc = best_result['by_dimension'][dim]
        print(f"  {dim:20} {acc:.1%}")

    print()
    print("Confusion Matrix:")
    conf = best_result['confusion']
    print(f"{'':>10} | {'low':>8} | {'high':>8}")
    print("-"*35)
    for true_level in ['low', 'high']:
        low_pred = conf[true_level]['low']
        high_pred = conf[true_level]['high']
        total = low_pred + high_pred
        acc = conf[true_level][true_level] / total if total > 0 else 0
        print(f"{true_level:>10} | {low_pred:>8} | {high_pred:>8} (acc: {acc:.1%})")

    print()
    print("="*70)

    return results

def compare_all_experiments():
    """Compare aggregation methods across all experiments"""

    files = [
        ("Few-shot (NO calibration)", "few_shot/data/performance_20251224_170955.json"),
        ("Few-shot (WITH calibration)", "few_shot/data/performance_20251224_173554.json"),
        ("Zero-shot", "zero_shot/data/performance_20251224_175736.json")
    ]

    all_results = {}

    for name, path in files:
        print(f"\n{'='*70}")
        print(f"📁 {name}")
        print()
        try:
            results = compare_all_methods(path)
            all_results[name] = results
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue

    # Final summary
    print("\n" + "="*70)
    print("📊 SUMMARY: BEST ACCURACY BY EXPERIMENT AND METHOD")
    print("="*70)
    print(f"{'Experiment':<35} {'Median':>8} {'Mean':>8} {'Max':>8} {'Majority':>8} {'Best':>8}")
    print("-"*70)

    for name, results in all_results.items():
        accs = [r['overall_accuracy'] for r in results]
        best_acc = max(accs)

        row = f"{name:<35}"
        for acc in accs:
            marker = "⭐" if acc == best_acc else ""
            row += f" {acc:>6.1%}{marker:<1}"
        row += f" {best_acc:>7.1%}"
        print(row)

    print("="*70)

    # Find overall best
    best_overall = None
    best_overall_acc = 0

    for exp_name, results in all_results.items():
        for result in results:
            if result['overall_accuracy'] > best_overall_acc:
                best_overall_acc = result['overall_accuracy']
                best_overall = (exp_name, result['method'])

    if best_overall:
        print(f"\n🏆 OVERALL BEST: {best_overall[0]} + {best_overall[1]} = {best_overall_acc:.1%}")
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single file analysis
        json_path = sys.argv[1]
        compare_all_methods(json_path)
    else:
        # Compare all experiments
        compare_all_experiments()
