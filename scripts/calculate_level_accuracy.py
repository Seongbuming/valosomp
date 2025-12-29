#!/usr/bin/env python3
"""
Calculate Level Accuracy from saved performance JSON files.
This is needed because earlier experiments didn't have Level Acc in the code.
"""

import json
import sys

def score_to_level(score):
    """Convert score (1-7) to level (low/middle/high)"""
    if score <= 3:
        return 'low'
    elif score == 4:
        return 'middle'
    else:  # 5-7
        return 'high'

def calculate_level_accuracy_from_json(json_path):
    """Calculate level accuracy from a performance JSON file"""

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Analyzing: {json_path}")
    print(f"Model params: {data.get('model_params', 'N/A')}")
    print()

    # This file format has human_scores and model_scores
    if 'human_scores' in data and 'model_scores' in data:
        human_scores = data['human_scores']
        model_scores = data['model_scores']

        for dimension in ['I-Involvement', 'You-Involvement', 'We-Involvement']:
            if dimension in human_scores and dimension in model_scores:
                human = human_scores[dimension]
                model = model_scores[dimension]

                human_levels = [score_to_level(s) for s in human]
                model_levels = [score_to_level(s) for s in model]

                level_acc = sum(h == m for h, m in zip(human_levels, model_levels)) / len(human_levels)

                print(f"{dimension}: Level Acc = {level_acc:.1%}")

        # Calculate average
        all_human = []
        all_model = []
        for dimension in ['I-Involvement', 'You-Involvement', 'We-Involvement']:
            if dimension in human_scores and dimension in model_scores:
                all_human.extend(human_scores[dimension])
                all_model.extend(model_scores[dimension])

        human_levels_all = [score_to_level(s) for s in all_human]
        model_levels_all = [score_to_level(s) for s in all_model]
        avg_level_acc = sum(h == m for h, m in zip(human_levels_all, model_levels_all)) / len(human_levels_all)

        print(f"\nAVERAGE Level Acc: {avg_level_acc:.1%}")
        print("="*60)
        return avg_level_acc
    else:
        print("Error: JSON file format not recognized")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        calculate_level_accuracy_from_json(json_path)
    else:
        print("Usage: python calculate_level_accuracy.py <performance_json_file>")
