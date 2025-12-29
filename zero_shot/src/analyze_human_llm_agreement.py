#!/usr/bin/env python3
"""
Human vs LLM Agreement Analysis

사람(MTurk)의 평가와 LLM의 평가(few-shot)를 비교 분석합니다.
k 값에 따라 사람과의 일치도가 어떻게 달라지는지 확인합니다.

분석 항목:
1. Correlation (Pearson, Spearman) - rating 유사도
2. MAE/RMSE - rating 정확도
3. Agreement rate - rating이 ±1 범위 내
4. Comment similarity (optional) - semantic similarity
5. k 값에 따른 변화 추이
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from pathlib import Path
import json
import sys

# Sentence transformers for comment similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Comment similarity analysis will be skipped.")


class HumanLLMAgreementAnalyzer:
    def __init__(self, mturk_path="data/mturk_survey_data_deduped.csv"):
        """Initialize analyzer with MTurk data as ground truth"""
        self.mturk_path = mturk_path
        self.mturk_df = None
        self.results = {}

        # Load embedding model for comment similarity
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Loading embedding model for comment similarity...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.embedding_model = None

        self.dimensions = ['I-Involvement', 'You-Involvement', 'We-Involvement']

    def load_mturk_data(self):
        """Load MTurk data as ground truth"""
        print(f"Loading MTurk data from {self.mturk_path}")
        self.mturk_df = pd.read_csv(self.mturk_path)

        # Calculate average ratings for each dimension
        for dim in self.dimensions:
            cols = [f'{dim}-1', f'{dim}-2', f'{dim}-3']
            self.mturk_df[f'{dim}_avg_human'] = self.mturk_df[cols].mean(axis=1)

        print(f"Loaded {len(self.mturk_df)} MTurk annotations")

    def load_llm_results(self, llm_csv_path, k_value):
        """Load LLM evaluation results"""
        print(f"\nLoading LLM results (k={k_value}) from {llm_csv_path}")
        llm_df = pd.read_csv(llm_csv_path)
        print(f"Loaded {len(llm_df)} LLM evaluations")
        return llm_df

    def merge_human_llm(self, llm_df):
        """Merge MTurk and LLM results on Tweet Text (because Tweet IDs may differ)"""
        # Merge on Tweet Text since Tweet IDs may be slightly different
        if 'tweet_text' in llm_df.columns and 'Tweet Text' in self.mturk_df.columns:
            merged = self.mturk_df.merge(
                llm_df,
                left_on='Tweet Text',
                right_on='tweet_text',
                how='inner',
                suffixes=('_human', '_llm')
            )
        elif 'Tweet Text' in llm_df.columns and 'Tweet Text' in self.mturk_df.columns:
            merged = self.mturk_df.merge(
                llm_df,
                on='Tweet Text',
                how='inner',
                suffixes=('_human', '_llm')
            )
        else:
            print(f"Error: Could not find matching columns")
            print(f"MTurk columns: {self.mturk_df.columns.tolist()[:10]}")
            print(f"LLM columns: {llm_df.columns.tolist()[:10]}")
            return pd.DataFrame()

        print(f"Matched {len(merged)} tweets between MTurk and LLM")

        if len(merged) == 0:
            print("Warning: No matching tweets found!")
            if 'Tweet ID' in self.mturk_df.columns:
                print(f"MTurk Tweet IDs sample: {self.mturk_df['Tweet ID'].head().tolist()}")
            if 'tweet_id' in llm_df.columns:
                print(f"LLM tweet_ids sample: {llm_df['tweet_id'].head().tolist()}")

        return merged

    def compute_rating_agreement(self, merged_df, k_value):
        """Compute agreement metrics for ratings"""
        metrics = {
            'k': k_value,
            'n_tweets': len(merged_df)
        }

        for dim in self.dimensions:
            human_col = f'{dim}_avg_human'
            llm_col = f'{dim}_avg'

            if human_col not in merged_df.columns or llm_col not in merged_df.columns:
                print(f"Warning: Missing columns for {dim}")
                continue

            human_scores = merged_df[human_col].values
            llm_scores = merged_df[llm_col].values

            # Remove NaN values
            valid_mask = ~(np.isnan(human_scores) | np.isnan(llm_scores))
            human_scores = human_scores[valid_mask]
            llm_scores = llm_scores[valid_mask]

            if len(human_scores) == 0:
                print(f"Warning: No valid scores for {dim}")
                continue

            # Correlation
            pearson_r, pearson_p = pearsonr(human_scores, llm_scores)
            spearman_r, spearman_p = spearmanr(human_scores, llm_scores)

            # Error metrics
            mae = mean_absolute_error(human_scores, llm_scores)
            rmse = np.sqrt(mean_squared_error(human_scores, llm_scores))

            # Agreement within ±1
            agreement_1 = np.mean(np.abs(human_scores - llm_scores) <= 1.0)
            agreement_0_5 = np.mean(np.abs(human_scores - llm_scores) <= 0.5)

            # Cohen's Kappa (for categorical agreement)
            # Convert to discrete categories (1-7)
            human_discrete = np.round(human_scores).astype(int)
            llm_discrete = np.round(llm_scores).astype(int)
            kappa = cohen_kappa_score(human_discrete, llm_discrete, weights='linear')

            # Bias (systematic over/under estimation)
            bias = np.mean(llm_scores - human_scores)

            metrics[dim] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'mae': mae,
                'rmse': rmse,
                'agreement_1': agreement_1,
                'agreement_0_5': agreement_0_5,
                'cohen_kappa': kappa,
                'bias': bias,
                'human_mean': np.mean(human_scores),
                'llm_mean': np.mean(llm_scores),
                'human_std': np.std(human_scores),
                'llm_std': np.std(llm_scores)
            }

        return metrics

    def compute_comment_similarity(self, merged_df, k_value):
        """Compute semantic similarity between human and LLM comments"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.embedding_model is None:
            return {}

        similarities = {}

        for dim in self.dimensions:
            human_col = f'{dim} Comment'
            llm_col = f'{dim}_comment'

            if human_col not in merged_df.columns or llm_col not in merged_df.columns:
                continue

            # Filter out NaN comments
            valid_comments = merged_df[[human_col, llm_col]].dropna()

            if len(valid_comments) == 0:
                continue

            human_comments = valid_comments[human_col].tolist()
            llm_comments = valid_comments[llm_col].tolist()

            # Compute embeddings
            print(f"Computing comment embeddings for {dim}...")
            human_embeddings = self.embedding_model.encode(human_comments)
            llm_embeddings = self.embedding_model.encode(llm_comments)

            # Compute cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            sims = []
            for h_emb, l_emb in zip(human_embeddings, llm_embeddings):
                sim = cosine_similarity([h_emb], [l_emb])[0][0]
                sims.append(sim)

            similarities[dim] = {
                'mean_similarity': np.mean(sims),
                'median_similarity': np.median(sims),
                'std_similarity': np.std(sims),
                'n_comments': len(sims),
                'similarities': sims  # Store individual similarities for plotting
            }

        return similarities

    def analyze_k_value(self, llm_csv_path, k_value):
        """Analyze agreement for a specific k value"""
        print(f"\n{'='*70}")
        print(f"Analyzing k={k_value}")
        print(f"{'='*70}")

        # Load LLM results
        llm_df = self.load_llm_results(llm_csv_path, k_value)

        # Merge with MTurk
        merged_df = self.merge_human_llm(llm_df)

        if len(merged_df) == 0:
            print(f"Skipping k={k_value} - no matching tweets")
            return None

        # Compute metrics
        rating_metrics = self.compute_rating_agreement(merged_df, k_value)
        comment_similarities = self.compute_comment_similarity(merged_df, k_value)

        result = {
            'k': k_value,
            'rating_metrics': rating_metrics,
            'comment_similarities': comment_similarities,
            'merged_df': merged_df
        }

        self.results[k_value] = result
        return result

    def print_summary(self, result):
        """Print summary for a k value"""
        k = result['k']
        metrics = result['rating_metrics']

        print(f"\n{'='*70}")
        print(f"Summary for k={k} (n={metrics['n_tweets']} tweets)")
        print(f"{'='*70}")

        for dim in self.dimensions:
            if dim not in metrics:
                continue

            m = metrics[dim]
            print(f"\n{dim}:")
            print(f"  Correlation:")
            print(f"    Pearson r:  {m['pearson_r']:.3f} (p={m['pearson_p']:.4f})")
            print(f"    Spearman r: {m['spearman_r']:.3f} (p={m['spearman_p']:.4f})")
            print(f"  Error:")
            print(f"    MAE:  {m['mae']:.3f}")
            print(f"    RMSE: {m['rmse']:.3f}")
            print(f"    Bias: {m['bias']:+.3f} (LLM - Human)")
            print(f"  Agreement:")
            print(f"    Within ±0.5: {m['agreement_0_5']*100:.1f}%")
            print(f"    Within ±1.0: {m['agreement_1']*100:.1f}%")
            print(f"    Cohen's κ:   {m['cohen_kappa']:.3f}")
            print(f"  Means:")
            print(f"    Human: {m['human_mean']:.2f} ± {m['human_std']:.2f}")
            print(f"    LLM:   {m['llm_mean']:.2f} ± {m['llm_std']:.2f}")

        # Comment similarities
        if result['comment_similarities']:
            print(f"\nComment Semantic Similarity:")
            for dim, sim in result['comment_similarities'].items():
                print(f"  {dim}: {sim['mean_similarity']:.3f} (n={sim['n_comments']})")

    def compare_k_values(self, k_values=[1, 4, 8, 16]):
        """Compare agreement across different k values"""
        print(f"\n{'='*70}")
        print(f"Comparing k values: {k_values}")
        print(f"{'='*70}")

        comparison = []

        for k in k_values:
            if k not in self.results:
                print(f"Warning: No results for k={k}")
                continue

            result = self.results[k]
            metrics = result['rating_metrics']

            row = {'k': k, 'n_tweets': metrics['n_tweets']}

            for dim in self.dimensions:
                if dim not in metrics:
                    continue

                m = metrics[dim]
                row[f'{dim}_pearson'] = m['pearson_r']
                row[f'{dim}_spearman'] = m['spearman_r']
                row[f'{dim}_mae'] = m['mae']
                row[f'{dim}_rmse'] = m['rmse']
                row[f'{dim}_agreement_1'] = m['agreement_1']
                row[f'{dim}_kappa'] = m['cohen_kappa']
                row[f'{dim}_bias'] = m['bias']

            comparison.append(row)

        comparison_df = pd.DataFrame(comparison)
        return comparison_df

    def plot_agreement_by_k(self, comparison_df, output_dir="few_shot/figures"):
        """Plot agreement metrics by k value"""
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        # 1. Correlation by k
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, dim in enumerate(self.dimensions):
            pearson_col = f'{dim}_pearson'
            spearman_col = f'{dim}_spearman'

            if pearson_col in comparison_df.columns:
                axes[i].plot(comparison_df['k'], comparison_df[pearson_col],
                           marker='o', label='Pearson', linewidth=2)
            if spearman_col in comparison_df.columns:
                axes[i].plot(comparison_df['k'], comparison_df[spearman_col],
                           marker='s', label='Spearman', linewidth=2)

            axes[i].set_xlabel('k (number of examples)', fontsize=12)
            axes[i].set_ylabel('Correlation Coefficient', fontsize=12)
            axes[i].set_title(dim, fontsize=14)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_by_k.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_dir}/correlation_by_k.png")
        plt.close()

        # 2. Error metrics by k
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, dim in enumerate(self.dimensions):
            mae_col = f'{dim}_mae'
            rmse_col = f'{dim}_rmse'

            if mae_col in comparison_df.columns:
                axes[i].plot(comparison_df['k'], comparison_df[mae_col],
                           marker='o', label='MAE', linewidth=2)
            if rmse_col in comparison_df.columns:
                axes[i].plot(comparison_df['k'], comparison_df[rmse_col],
                           marker='s', label='RMSE', linewidth=2)

            axes[i].set_xlabel('k (number of examples)', fontsize=12)
            axes[i].set_ylabel('Error', fontsize=12)
            axes[i].set_title(dim, fontsize=14)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_by_k.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/error_by_k.png")
        plt.close()

        # 3. Agreement rate by k
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, dim in enumerate(self.dimensions):
            agreement_col = f'{dim}_agreement_1'

            if agreement_col in comparison_df.columns:
                axes[i].plot(comparison_df['k'], comparison_df[agreement_col] * 100,
                           marker='o', linewidth=2, color='green')

            axes[i].set_xlabel('k (number of examples)', fontsize=12)
            axes[i].set_ylabel('Agreement Rate (%)', fontsize=12)
            axes[i].set_title(f'{dim}\n(within ±1 point)', fontsize=14)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/agreement_by_k.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/agreement_by_k.png")
        plt.close()

    def plot_rating_scatter(self, k_value, output_dir="few_shot/figures"):
        """Plot scatter plots of human vs LLM ratings for a specific k"""
        if k_value not in self.results:
            print(f"No results for k={k_value}")
            return

        Path(output_dir).mkdir(exist_ok=True, parents=True)
        merged_df = self.results[k_value]['merged_df']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, dim in enumerate(self.dimensions):
            human_col = f'{dim}_avg_human'
            llm_col = f'{dim}_avg'

            if human_col not in merged_df.columns or llm_col not in merged_df.columns:
                continue

            # Scatter plot
            axes[i].scatter(merged_df[human_col], merged_df[llm_col],
                          alpha=0.5, s=30)

            # Perfect agreement line
            axes[i].plot([1, 7], [1, 7], 'r--', linewidth=2, label='Perfect Agreement')

            # Compute correlation
            valid_data = merged_df[[human_col, llm_col]].dropna()
            if len(valid_data) > 0:
                r, p = pearsonr(valid_data[human_col], valid_data[llm_col])
                axes[i].text(0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}',
                           transform=axes[i].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            axes[i].set_xlabel('Human Rating', fontsize=12)
            axes[i].set_ylabel(f'LLM Rating (k={k_value})', fontsize=12)
            axes[i].set_title(dim, fontsize=14)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0.5, 7.5)
            axes[i].set_ylim(0.5, 7.5)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/rating_scatter_human_vs_llm_k{k_value}.png',
                   dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_dir}/rating_scatter_human_vs_llm_k{k_value}.png")
        plt.close()

    def plot_comment_scatter(self, k_value, output_dir="few_shot/figures"):
        """Plot scatter plots of human vs LLM comment similarity for a specific k"""
        if k_value not in self.results:
            print(f"No results for k={k_value}")
            return

        Path(output_dir).mkdir(exist_ok=True, parents=True)
        comment_sims = self.results[k_value].get('comment_similarities', {})

        if not comment_sims:
            print(f"No comment similarity data for k={k_value}")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, dim in enumerate(self.dimensions):
            if dim not in comment_sims:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=axes[i].transAxes)
                axes[i].set_title(dim, fontsize=14)
                continue

            sim_data = comment_sims[dim]
            similarities = sim_data.get('similarities', [])

            if len(similarities) > 0:
                # Histogram of comment similarities
                axes[i].hist(similarities, bins=20, alpha=0.7, edgecolor='black')
                axes[i].axvline(sim_data['mean_similarity'], color='red',
                              linestyle='--', linewidth=2,
                              label=f"Mean: {sim_data['mean_similarity']:.3f}")
                axes[i].set_xlabel('Cosine Similarity', fontsize=12)
                axes[i].set_ylabel('Frequency', fontsize=12)
                axes[i].set_title(f'{dim} (n={sim_data["n_comments"]})', fontsize=14)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3, axis='y')
                axes[i].set_xlim(-0.1, 1.1)

        plt.suptitle(f'Comment Semantic Similarity Distribution (k={k_value})',
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comment_scatter_human_vs_llm_k{k_value}.png',
                   dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/comment_scatter_human_vs_llm_k{k_value}.png")
        plt.close()

    def save_comparison_table(self, comparison_df, output_path="few_shot/human_llm_agreement.csv"):
        """Save comparison table to CSV"""
        comparison_df.to_csv(output_path, index=False)
        print(f"\nSaved comparison table: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Human vs LLM agreement")
    parser.add_argument(
        '--mturk-data',
        type=str,
        default='data/mturk_survey_data_deduped.csv',
        help='Path to MTurk data (ground truth)'
    )
    parser.add_argument(
        '--llm-results',
        nargs='+',
        required=True,
        help='Paths to LLM result CSVs (one per k value)'
    )
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        required=True,
        help='k values corresponding to LLM results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='few_shot/figures',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    if len(args.llm_results) != len(args.k_values):
        print("Error: Number of LLM result files must match number of k values")
        return

    # Initialize analyzer
    analyzer = HumanLLMAgreementAnalyzer(args.mturk_data)
    analyzer.load_mturk_data()

    # Analyze each k value
    for llm_csv, k in zip(args.llm_results, args.k_values):
        result = analyzer.analyze_k_value(llm_csv, k)
        if result:
            analyzer.print_summary(result)
            analyzer.plot_rating_scatter(k, args.output_dir)
            analyzer.plot_comment_scatter(k, args.output_dir)

    # Compare across k values
    if len(analyzer.results) > 1:
        comparison_df = analyzer.compare_k_values(args.k_values)
        print("\n" + "="*70)
        print("Comparison Across k Values")
        print("="*70)
        print(comparison_df.to_string())

        analyzer.plot_agreement_by_k(comparison_df, args.output_dir)
        analyzer.save_comparison_table(comparison_df)


if __name__ == "__main__":
    main()
