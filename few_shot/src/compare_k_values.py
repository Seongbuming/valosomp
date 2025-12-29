#!/usr/bin/env python3
"""
Compare results across different k-shot values
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

def load_results(k_value, results_dir="few_shot/results"):
    """Load results for a specific k value"""
    # Try to find the most recent file for this k value
    pattern = f"*k{k_value}*.csv"
    files = list(Path(results_dir).glob(pattern))

    if not files:
        # Try in current directory
        files = list(Path("few_shot").glob(f"test_targeted_k{k_value}.csv"))

    if not files:
        print(f"Warning: No results found for k={k_value}")
        return None

    # Use the most recent file
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    print(f"Loading k={k_value} from: {latest_file}")

    df = pd.read_csv(latest_file)
    df['k'] = k_value
    return df


def compare_k_values(k_values=[1, 4, 8, 16], results_dir="few_shot/results"):
    """Compare results across different k values"""

    # Load all results
    dfs = []
    for k in k_values:
        df = load_results(k, results_dir)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("No results found!")
        return

    # Combine all dataframes
    all_results = pd.concat(dfs, ignore_index=True)

    print(f"\nLoaded {len(dfs)} k-value settings")
    print(f"Total evaluations: {len(all_results)}")

    # Create comparison plots
    dimensions = ['I-Involvement', 'You-Involvement', 'We-Involvement']

    # 1. Distribution of average scores by k
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, dim in enumerate(dimensions):
        col = f'{dim}_avg'
        if col in all_results.columns:
            all_results.boxplot(column=col, by='k', ax=axes[i])
            axes[i].set_title(f'{dim} Score Distribution by k')
            axes[i].set_xlabel('k (number of examples)')
            axes[i].set_ylabel('Average Score')
            axes[i].get_figure().suptitle('')  # Remove default title

    plt.tight_layout()
    plt.savefig('few_shot/k_comparison_distributions.png', dpi=150)
    print("\nSaved: few_shot/k_comparison_distributions.png")

    # 2. Mean scores by k
    fig, ax = plt.subplots(figsize=(10, 6))

    mean_scores = []
    for dim in dimensions:
        col = f'{dim}_avg'
        if col in all_results.columns:
            means = all_results.groupby('k')[col].mean()
            mean_scores.append(means)
            ax.plot(means.index, means.values, marker='o', label=dim, linewidth=2)

    ax.set_xlabel('k (number of examples)', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('Mean Involvement Scores by k-shot Setting', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('few_shot/k_comparison_means.png', dpi=150)
    print("Saved: few_shot/k_comparison_means.png")

    # 3. Statistical summary
    print("\n" + "="*60)
    print("Statistical Summary by k-value")
    print("="*60)

    for dim in dimensions:
        col = f'{dim}_avg'
        if col in all_results.columns:
            print(f"\n{dim}:")
            summary = all_results.groupby('k')[col].agg(['mean', 'std', 'min', 'max', 'count'])
            print(summary.to_string())

    # 4. Variance analysis
    print("\n" + "="*60)
    print("Score Variance by k-value")
    print("="*60)
    print("(Lower variance may indicate more consistent evaluations)")

    for dim in dimensions:
        col = f'{dim}_avg'
        if col in all_results.columns:
            variance = all_results.groupby('k')[col].var()
            print(f"\n{dim}:")
            print(variance.to_string())

    # 5. If we have multiple k values on same tweets, compute correlations
    if len(dfs) > 1:
        print("\n" + "="*60)
        print("Score Correlations Between k-values")
        print("="*60)

        # Pivot to get k values as columns
        for dim in dimensions:
            col = f'{dim}_avg'
            if col not in all_results.columns:
                continue

            pivot = all_results.pivot_table(
                index='tweet_id',
                columns='k',
                values=col
            )

            if len(pivot.columns) > 1:
                corr = pivot.corr()
                print(f"\n{dim} Correlation Matrix:")
                print(corr.to_string())

    # Save summary table
    summary_data = []
    for k in sorted(all_results['k'].unique()):
        k_data = all_results[all_results['k'] == k]
        row = {'k': k, 'n_tweets': len(k_data)}

        for dim in dimensions:
            col = f'{dim}_avg'
            if col in k_data.columns:
                row[f'{dim}_mean'] = k_data[col].mean()
                row[f'{dim}_std'] = k_data[col].std()

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('few_shot/k_comparison_summary.csv', index=False)
    print("\n" + "="*60)
    print("Saved summary table: few_shot/k_comparison_summary.csv")
    print("="*60)

    return all_results


def compare_with_zeroshot(fewshot_k4_path, zeroshot_path):
    """Compare few-shot (k=4) with zero-shot results"""

    print("\n" + "="*60)
    print("Comparing Few-Shot (k=4) vs Zero-Shot")
    print("="*60)

    try:
        fewshot = pd.read_csv(fewshot_k4_path)
        zeroshot = pd.read_csv(zeroshot_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return

    # Merge on tweet_id
    merged = fewshot.merge(
        zeroshot,
        on='tweet_id',
        suffixes=('_few', '_zero')
    )

    print(f"\nMatched {len(merged)} tweets between datasets")

    dimensions = ['I-Involvement', 'You-Involvement', 'We-Involvement']

    # Create scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, dim in enumerate(dimensions):
        few_col = f'{dim}_avg_few'
        zero_col = f'{dim}_avg_zero'

        if few_col in merged.columns and zero_col in merged.columns:
            axes[i].scatter(merged[zero_col], merged[few_col], alpha=0.6)
            axes[i].plot([1, 7], [1, 7], 'r--', label='y=x')
            axes[i].set_xlabel('Zero-Shot Score')
            axes[i].set_ylabel('Few-Shot (k=4) Score')
            axes[i].set_title(dim)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            # Compute correlation
            corr = merged[[zero_col, few_col]].corr().iloc[0, 1]
            axes[i].text(0.05, 0.95, f'r={corr:.3f}',
                        transform=axes[i].transAxes,
                        verticalalignment='top')

    plt.tight_layout()
    plt.savefig('few_shot/fewshot_vs_zeroshot.png', dpi=150)
    print("\nSaved: few_shot/fewshot_vs_zeroshot.png")

    # Statistical comparison
    print("\n" + "="*60)
    print("Mean Score Comparison")
    print("="*60)

    for dim in dimensions:
        few_col = f'{dim}_avg_few'
        zero_col = f'{dim}_avg_zero'

        if few_col in merged.columns and zero_col in merged.columns:
            few_mean = merged[few_col].mean()
            zero_mean = merged[zero_col].mean()
            diff = few_mean - zero_mean

            print(f"\n{dim}:")
            print(f"  Zero-Shot: {zero_mean:.3f}")
            print(f"  Few-Shot:  {few_mean:.3f}")
            print(f"  Difference: {diff:+.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare few-shot results across k values")
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        default=[1, 4],
        help='k values to compare (default: 1 4)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='few_shot/results',
        help='Directory containing results files'
    )
    parser.add_argument(
        '--compare-zeroshot',
        action='store_true',
        help='Also compare with zero-shot results'
    )
    parser.add_argument(
        '--zeroshot-path',
        type=str,
        default='zero_shot/results/gemma_evaluations.csv',
        help='Path to zero-shot results CSV'
    )

    args = parser.parse_args()

    # Compare k values
    results = compare_k_values(args.k_values, args.results_dir)

    # Optionally compare with zero-shot
    if args.compare_zeroshot and results is not None:
        # Find k=4 results
        k4_file = None
        for k in [4]:  # Look for k=4 specifically
            df = load_results(k, args.results_dir)
            if df is not None:
                k4_file = f"few_shot/test_targeted_k4.csv"  # Adjust as needed
                break

        if k4_file and os.path.exists(args.zeroshot_path):
            compare_with_zeroshot(k4_file, args.zeroshot_path)
        else:
            print("\nSkipping zero-shot comparison (files not found)")


if __name__ == "__main__":
    main()
