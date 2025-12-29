#!/usr/bin/env python3
"""
Calculate 2-Level Accuracy (Low vs High) from saved performance JSON files.
Low: 1-4, High: 5-7
"""

import json
import sys
from collections import defaultdict
import numpy as np

def score_to_2level(score):
    """Convert score (1-7) to 2-level (low/high)"""
    if score <= 4:
        return 'low'
    else:  # 5-7
        return 'high'

def analyze_2level_accuracy(json_path):
    """Calculate 2-level accuracy from a performance JSON file"""

    with open(json_path, 'r') as f:
        data = json.load(f)

    print("="*70)
    print("2-LEVEL ACCURACY ANALYSIS (Low: 1-4, High: 5-7)")
    print("="*70)
    print(f"File: {json_path}")
    print(f"Model params: {data.get('model_params', 'N/A')}")
    print()

    # Collect predictions by dimension
    results_by_dim = {
        'I-Involvement': {'correct': 0, 'total': 0},
        'You-Involvement': {'correct': 0, 'total': 0},
        'We-Involvement': {'correct': 0, 'total': 0}
    }

    # Confusion matrix for overall
    confusion = defaultdict(lambda: defaultdict(int))

    for pred in data['detailed_predictions']:
        for dim in ['I-Involvement', 'You-Involvement', 'We-Involvement']:
            human = pred['human_scores'][dim]
            model = pred['model_scores'][dim]

            # Use median for multi-annotator scores
            human_median = np.median(human)
            model_median = np.median(model)

            human_level = score_to_2level(human_median)
            model_level = score_to_2level(model_median)

            # Track per-dimension accuracy
            results_by_dim[dim]['total'] += 1
            if human_level == model_level:
                results_by_dim[dim]['correct'] += 1

            # Track confusion
            confusion[human_level][model_level] += 1

    # Print per-dimension results
    print("📊 PER-DIMENSION 2-LEVEL ACCURACY")
    print("-"*70)

    all_correct = 0
    all_total = 0

    for dim in ['I-Involvement', 'You-Involvement', 'We-Involvement']:
        correct = results_by_dim[dim]['correct']
        total = results_by_dim[dim]['total']
        acc = correct / total if total > 0 else 0

        all_correct += correct
        all_total += total

        print(f"{dim:20} {correct:3}/{total:3} = {acc:6.1%}")

    avg_acc = all_correct / all_total if all_total > 0 else 0
    print("-"*70)
    print(f"{'AVERAGE':20} {all_correct:3}/{all_total:3} = {avg_acc:6.1%} ⭐")
    print()

    # Print confusion matrix
    print("📊 2-LEVEL CONFUSION MATRIX (rows=human, cols=model)")
    print("-"*70)
    print(f"{'':>10} | {'low':>8} | {'high':>8} | Total | Accuracy")
    print("-"*70)

    for human_level in ['low', 'high']:
        total = sum(confusion[human_level].values())
        correct = confusion[human_level][human_level]
        acc = correct / total if total > 0 else 0

        print(f"{human_level:>10} | {confusion[human_level]['low']:>8} | "
              f"{confusion[human_level]['high']:>8} | {total:>5} | {acc:>7.1%}")

    print("-"*70)
    print(f"Overall: {avg_acc:.1%}")
    print()

    # Compare with 3-level accuracy from file
    if 'summary' in data and 'avg_level_acc' in data['summary']:
        old_acc = data['summary']['avg_level_acc']
        improvement = avg_acc - old_acc
        print(f"📈 IMPROVEMENT OVER 3-LEVEL")
        print("-"*70)
        print(f"3-level accuracy: {old_acc:.1%}")
        print(f"2-level accuracy: {avg_acc:.1%}")
        print(f"Improvement:      +{improvement:.1%} ({improvement*100:.1f} points)")
        print()

    print("="*70)

    return {
        'accuracy': avg_acc,
        'by_dimension': {
            dim: results_by_dim[dim]['correct'] / results_by_dim[dim]['total']
            for dim in results_by_dim
        },
        'confusion_matrix': dict(confusion)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        analyze_2level_accuracy(json_path)
    else:
        print("Analyzing all recent experiment results...")
        print()

        files = [
            ("Few-shot (NO calibration)", "few_shot/data/performance_20251224_170955.json"),
            ("Few-shot (WITH calibration)", "few_shot/data/performance_20251224_173554.json"),
            ("Zero-shot", "zero_shot/data/performance_20251224_175736.json")
        ]

        results = []
        for name, path in files:
            print(f"\n{'='*70}")
            print(f"📁 {name}")
            try:
                result = analyze_2level_accuracy(path)
                results.append((name, result['accuracy']))
            except FileNotFoundError:
                print(f"File not found: {path}")
                continue

        # Summary comparison
        if results:
            print("\n" + "="*70)
            print("📊 SUMMARY COMPARISON")
            print("="*70)
            for name, acc in results:
                print(f"{name:35} {acc:.1%}")
            print("="*70)
