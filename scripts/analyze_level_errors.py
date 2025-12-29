#!/usr/bin/env python3
"""
Analyze Level Accuracy errors to understand performance gaps.
"""

import json
import sys
from collections import Counter, defaultdict
import numpy as np

def score_to_level(score):
    """Convert score (1-7) to level (low/middle/high)"""
    if score <= 3:
        return 'low'
    elif score == 4:
        return 'middle'
    else:  # 5-7
        return 'high'

def analyze_predictions(json_path):
    """Analyze prediction patterns from performance JSON"""

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Analyzing: {json_path}")
    print(f"Model params: {data.get('model_params', 'N/A')}")
    print("="*70)

    # Collect all predictions
    all_errors = []
    level_confusion = defaultdict(lambda: defaultdict(int))
    model_variance = []  # Track variance in model's 3 predictions
    human_variance = []  # Track variance in human's 3 annotations

    for pred in data['detailed_predictions']:
        for dim in ['I-Involvement', 'You-Involvement', 'We-Involvement']:
            human = pred['human_scores'][dim]
            model = pred['model_scores'][dim]

            # Convert to levels (use median for multi-annotator)
            human_median = np.median(human)
            model_median = np.median(model)

            human_level = score_to_level(human_median)
            model_level = score_to_level(model_median)

            # Track confusion
            level_confusion[human_level][model_level] += 1

            # Track variance
            model_variance.append(np.var(model))
            human_variance.append(np.var(human))

            if human_level != model_level:
                all_errors.append({
                    'dimension': dim,
                    'tweet': pred['tweet_text'][:50] + '...',
                    'human_scores': human,
                    'model_scores': model,
                    'human_level': human_level,
                    'model_level': model_level
                })

    # Print confusion matrix
    print("\n📊 LEVEL CONFUSION MATRIX (rows=human, cols=model)")
    print("-"*70)
    levels = ['low', 'middle', 'high']
    print(f"{'':>10} | {'low':>8} | {'middle':>8} | {'high':>8} | Total")
    print("-"*70)

    for human_level in levels:
        total = sum(level_confusion[human_level].values())
        correct = level_confusion[human_level][human_level]
        print(f"{human_level:>10} | {level_confusion[human_level]['low']:>8} | "
              f"{level_confusion[human_level]['middle']:>8} | "
              f"{level_confusion[human_level]['high']:>8} | {total:>5} "
              f"(acc: {100*correct/total if total > 0 else 0:.1f}%)")

    # Calculate overall accuracy
    total_predictions = sum(sum(v.values()) for v in level_confusion.values())
    correct_predictions = sum(level_confusion[lvl][lvl] for lvl in levels)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print("-"*70)
    print(f"Overall Level Accuracy: {accuracy:.1%}")
    print()

    # Analyze model variance
    print("📈 PREDICTION VARIANCE ANALYSIS")
    print("-"*70)
    print(f"Model variance (avg): {np.mean(model_variance):.3f}")
    print(f"Human variance (avg): {np.mean(human_variance):.3f}")
    print()
    print("High model variance = model gives inconsistent scores")
    print("(e.g., predicting [1, 5, 7] instead of [5, 5, 5])")
    print()

    # Show most common error patterns
    print("🔍 MOST COMMON ERRORS (Top 10)")
    print("-"*70)
    error_patterns = Counter()
    for err in all_errors:
        pattern = f"{err['human_level']} → {err['model_level']} ({err['dimension']})"
        error_patterns[pattern] += 1

    for pattern, count in error_patterns.most_common(10):
        pct = 100 * count / len(all_errors)
        print(f"{pattern:40} {count:3} ({pct:.1f}%)")

    # Show sample errors
    print("\n📝 SAMPLE ERRORS (first 5)")
    print("-"*70)
    for i, err in enumerate(all_errors[:5]):
        print(f"\n{i+1}. {err['dimension']}")
        print(f"   Tweet: {err['tweet']}")
        print(f"   Human: {err['human_scores']} (level: {err['human_level']})")
        print(f"   Model: {err['model_scores']} (level: {err['model_level']})")

    print("\n" + "="*70)

    return {
        'confusion_matrix': dict(level_confusion),
        'accuracy': accuracy,
        'model_variance': np.mean(model_variance),
        'human_variance': np.mean(human_variance),
        'error_count': len(all_errors),
        'total_predictions': total_predictions
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # Default to latest zero-shot
        json_path = "zero_shot/data/performance_20251224_175736.json"

    analyze_predictions(json_path)
