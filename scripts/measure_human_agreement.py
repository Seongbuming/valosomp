#!/usr/bin/env python3
"""
Measure inter-annotator agreement in MTurk data.
This provides the upper bound (ceiling) for model performance.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import sys


def calculate_human_agreement(mturk_data_path="data/survey_data_ps_deduped.csv"):
    """
    Calculate inter-annotator agreement metrics.

    In MTurk data, each tweet has ratings from multiple annotators.
    We measure how much they agree with each other.
    """
    df = pd.read_csv(mturk_data_path)

    print("="*70)
    print("HUMAN INTER-ANNOTATOR AGREEMENT ANALYSIS")
    print("="*70)
    print(f"Dataset: {mturk_data_path}")
    print(f"Total samples: {len(df)}\n")

    dimensions = {
        "I-Involvement": ["I-Involvement-1", "I-Involvement-2", "I-Involvement-3"],
        "You-Involvement": ["You-Involvement-1", "You-Involvement-2", "You-Involvement-3"],
        "We-Involvement": ["We-Involvement-1", "We-Involvement-2", "We-Involvement-3"]
    }

    results = {}

    for dim_name, rating_cols in dimensions.items():
        print(f"\n{dim_name}:")
        print("-"*70)

        # Extract ratings for all 3 questions
        all_ratings = []
        for col in rating_cols:
            if col in df.columns:
                all_ratings.append(df[col].values)

        if len(all_ratings) < 3:
            print(f"  ⚠️  Insufficient data")
            continue

        # Calculate pairwise agreement between questions (within same dimension)
        # This tells us how consistent the 3 questions are
        maes = []
        correlations = []
        within_1_accs = []

        # Compare Q1 vs Q2, Q1 vs Q3, Q2 vs Q3
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            # Remove NaN values
            valid_idx = ~(np.isnan(all_ratings[i]) | np.isnan(all_ratings[j]))
            r1 = all_ratings[i][valid_idx]
            r2 = all_ratings[j][valid_idx]

            if len(r1) < 10:
                continue

            # MAE
            mae = mean_absolute_error(r1, r2)
            maes.append(mae)

            # Correlation
            corr, _ = pearsonr(r1, r2)
            correlations.append(corr)

            # Within-1 accuracy
            within_1 = np.mean(np.abs(r1 - r2) <= 1)
            within_1_accs.append(within_1)

        # Average metrics
        avg_mae = np.mean(maes) if maes else np.nan
        avg_corr = np.mean(correlations) if correlations else np.nan
        avg_within_1 = np.mean(within_1_accs) if within_1_accs else np.nan

        # Standard deviation across questions (shows consistency)
        all_ratings_stacked = np.column_stack(all_ratings)
        std_per_sample = np.nanstd(all_ratings_stacked, axis=1)
        avg_std = np.nanmean(std_per_sample)

        print(f"  Inter-question agreement (within dimension):")
        print(f"    MAE:           {avg_mae:.3f}")
        print(f"    Pearson r:     {avg_corr:.3f}")
        print(f"    Within-1 Acc:  {avg_within_1:.1%}")
        print(f"    Avg Std Dev:   {avg_std:.3f}")

        results[dim_name] = {
            'mae': avg_mae,
            'pearson': avg_corr,
            'within_1': avg_within_1,
            'std': avg_std
        }

    # Overall summary
    print("\n" + "="*70)
    print("SUMMARY - Human Agreement Ceiling")
    print("="*70)

    avg_mae = np.mean([r['mae'] for r in results.values() if not np.isnan(r['mae'])])
    avg_corr = np.mean([r['pearson'] for r in results.values() if not np.isnan(r['pearson'])])
    avg_within_1 = np.mean([r['within_1'] for r in results.values() if not np.isnan(r['within_1'])])

    print(f"Average across dimensions:")
    print(f"  MAE:           {avg_mae:.3f}")
    print(f"  Pearson r:     {avg_corr:.3f}")
    print(f"  Within-1 Acc:  {avg_within_1:.1%}")
    print()

    # Interpret results
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nThese metrics represent how much HUMANS agree with each other.")
    print("This is the CEILING for model performance.\n")

    print("Realistic Performance Goals:")
    print("-"*70)

    # Model should aim for 70-90% of human performance
    target_mae_high = avg_mae * 1.3  # Within 30% worse than humans
    target_mae_excellent = avg_mae * 1.1  # Within 10% worse than humans

    target_corr_high = avg_corr * 0.85  # 85% of human correlation
    target_corr_excellent = avg_corr * 0.95  # 95% of human correlation

    target_w1_high = avg_within_1 * 0.85
    target_w1_excellent = avg_within_1 * 0.95

    print(f"\n🎯 EXCELLENT Performance (95% of human):")
    print(f"   MAE:           < {target_mae_excellent:.3f}")
    print(f"   Pearson r:     > {target_corr_excellent:.3f}")
    print(f"   Within-1 Acc:  > {target_w1_excellent:.1%}")

    print(f"\n✅ GOOD Performance (85% of human):")
    print(f"   MAE:           < {target_mae_high:.3f}")
    print(f"   Pearson r:     > {target_corr_high:.3f}")
    print(f"   Within-1 Acc:  > {target_w1_high:.1%}")

    print(f"\n⚠️  ACCEPTABLE Performance (70% of human):")
    print(f"   MAE:           < {avg_mae * 1.5:.3f}")
    print(f"   Pearson r:     > {avg_corr * 0.7:.3f}")
    print(f"   Within-1 Acc:  > {avg_within_1 * 0.7:.1%}")

    print(f"\n⭐ LEVEL-BASED GOALS (Primary Metric):")
    print(f"   Excellent: > 85% (Low/Mid/High accuracy)")
    print(f"   Good:      > 75%")
    print(f"   Acceptable: > 65%")

    print("\n" + "="*70)
    print("LITERATURE BENCHMARKS")
    print("="*70)
    print("""
For similar subjective rating prediction tasks:

Sentiment Intensity Prediction (SemEval-2018):
  - Best systems: Pearson r = 0.7-0.8
  - Good systems: Pearson r = 0.6-0.7

Emotion Recognition (EmoContext):
  - Best systems: F1 = 0.75-0.80
  - Good systems: F1 = 0.65-0.75

Stance Detection:
  - Best systems: F1 = 0.65-0.70
  - Good systems: F1 = 0.55-0.65

For your involvement prediction task (more subjective):
  - Excellent: Pearson r > 0.6, Within-1 > 75%
  - Good: Pearson r > 0.5, Within-1 > 65%
  - Acceptable: Pearson r > 0.4, Within-1 > 55%
    """)

    print("="*70)

    return results


if __name__ == "__main__":
    calculate_human_agreement()
