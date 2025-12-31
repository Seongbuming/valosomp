#!/usr/bin/env python3
"""
Calculate processing time comparison between Human Coding and LLM Classification
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def estimate_human_coding_time():
    """
    Estimate human coding time based on survey data
    Assumptions:
    - Each tweet requires rating 3 dimensions × 3 statements = 9 ratings
    - Average time per rating: ~5-10 seconds (conservative estimate)
    - Total per tweet: ~45-90 seconds
    """

    # Load human data
    pilot_df = pd.read_csv('data/survey_data_ps_deduped_fixed.csv')
    main_df = pd.read_csv('data/survey_data_ms_deduped_fixed.csv')

    n_pilot_tweets = len(pilot_df)
    n_main_tweets = len(main_df)

    # Conservative estimate: 60 seconds per tweet (1 minute)
    # This includes reading the tweet, understanding context, and making 9 ratings
    avg_seconds_per_tweet = 60

    pilot_time_seconds = n_pilot_tweets * avg_seconds_per_tweet
    main_time_seconds = n_main_tweets * avg_seconds_per_tweet

    return {
        'pilot': {
            'n_tweets': n_pilot_tweets,
            'time_seconds': pilot_time_seconds,
            'time_minutes': pilot_time_seconds / 60,
            'time_hours': pilot_time_seconds / 3600
        },
        'main': {
            'n_tweets': n_main_tweets,
            'time_seconds': main_time_seconds,
            'time_minutes': main_time_seconds / 60,
            'time_hours': main_time_seconds / 3600
        },
        'combined': {
            'n_tweets': n_pilot_tweets + n_main_tweets,
            'time_seconds': pilot_time_seconds + main_time_seconds,
            'time_minutes': (pilot_time_seconds + main_time_seconds) / 60,
            'time_hours': (pilot_time_seconds + main_time_seconds) / 3600
        }
    }

def calculate_llm_time():
    """
    Calculate actual LLM processing time from evaluation files
    """

    results = {}

    for k in [0, 1, 4, 8, 16]:
        eval_file = f'few_shot/data/evaluations/combined_k{k}.json'

        try:
            with open(eval_file, 'r') as f:
                evals = json.load(f)

            # Extract timestamps
            timestamps = []
            for eval in evals:
                if 'timestamp' in eval:
                    try:
                        ts = datetime.fromisoformat(eval['timestamp'])
                        timestamps.append(ts)
                    except:
                        pass

            if len(timestamps) >= 2:
                # Calculate time span
                timestamps.sort()
                time_span = (timestamps[-1] - timestamps[0]).total_seconds()
                n_tweets = len(evals)
                avg_per_tweet = time_span / n_tweets if n_tweets > 0 else 0

                results[k] = {
                    'n_tweets': n_tweets,
                    'total_time_seconds': time_span,
                    'avg_seconds_per_tweet': avg_per_tweet,
                    'start': timestamps[0],
                    'end': timestamps[-1]
                }
        except Exception as e:
            print(f"Error processing k={k}: {e}")
            results[k] = None

    return results

def create_comparison_figure(human_time, llm_times, output_path):
    """
    Create publication-quality visualization comparing processing times
    Using professional academic color palette and clean design
    """

    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.3)

    # Professional color palette (ColorBrewer Set2)
    colors = {
        'human': '#8da0cb',      # Muted blue
        'llm': '#fc8d62',        # Muted orange
        'speedup': '#66c2a5',    # Teal
        'grid': '#e0e0e0',       # Light gray
        'text': '#2d2d2d'        # Dark gray for text
    }

    # Prepare data
    k_values = [k for k, v in llm_times.items() if v is not None]
    llm_per_tweet = [llm_times[k]['avg_seconds_per_tweet'] for k in k_values]
    human_per_tweet = 60  # 60 seconds per tweet
    speedup = [human_per_tweet / llm for llm in llm_per_tweet]

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ===== Panel A: Processing Time Comparison =====
    x = np.arange(len(k_values))
    width = 0.35

    bars1 = ax1.bar(x - width/2, [human_per_tweet] * len(k_values), width,
                    label='Human Coding', color=colors['human'],
                    edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, llm_per_tweet, width,
                    label='LLM Classification', color=colors['llm'],
                    edgecolor='white', linewidth=1.5)

    ax1.set_xlabel('Few-Shot Examples (k)', fontsize=11, color=colors['text'])
    ax1.set_ylabel('Processing Time (seconds/tweet)', fontsize=11, color=colors['text'])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{k}' if k > 0 else 'Zero-shot' for k in k_values])
    ax1.legend(loc='upper left', frameon=True, fancybox=False,
               edgecolor=colors['grid'], fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, color=colors['grid'])
    ax1.set_axisbelow(True)
    ax1.tick_params(colors=colors['text'])

    # Add value labels on bars (only for LLM to avoid clutter)
    for i, (bar, val) in enumerate(zip(bars2, llm_per_tweet)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{val:.1f}s',
                ha='center', va='bottom', fontsize=9,
                color=colors['text'], fontweight='500')

    # Add annotation for human baseline
    ax1.axhline(y=human_per_tweet, color=colors['human'],
                linestyle=':', alpha=0.4, linewidth=1.5)
    ax1.text(len(k_values) - 0.5, human_per_tweet + 3,
             'Human baseline: 60s',
             fontsize=9, color=colors['human'],
             ha='right', va='bottom', fontstyle='italic')

    # Panel label
    ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')

    # ===== Panel B: Efficiency Gain (Speedup) =====
    bars = ax2.bar(range(len(k_values)), speedup,
                   color=colors['speedup'], alpha=0.85,
                   edgecolor='white', linewidth=1.5)

    ax2.set_xlabel('Few-Shot Examples (k)', fontsize=11, color=colors['text'])
    ax2.set_ylabel('Speedup Factor (×)', fontsize=11, color=colors['text'])
    ax2.set_xticks(range(len(k_values)))
    ax2.set_xticklabels([f'{k}' if k > 0 else 'Zero-shot' for k in k_values])
    ax2.axhline(y=1, color='#d62728', linestyle='--',
                label='Equal Speed', alpha=0.6, linewidth=1.5)
    ax2.legend(loc='upper left', frameon=True, fancybox=False,
               edgecolor=colors['grid'], fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, color=colors['grid'])
    ax2.set_axisbelow(True)
    ax2.tick_params(colors=colors['text'])

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, speedup)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{val:.1f}×',
                ha='center', va='bottom', fontsize=9,
                color=colors['text'], fontweight='600')

    # Highlight most efficient configuration
    max_speedup_idx = speedup.index(max(speedup))
    bars[max_speedup_idx].set_alpha(1.0)
    bars[max_speedup_idx].set_edgecolor(colors['speedup'])
    bars[max_speedup_idx].set_linewidth(2.5)

    # Panel label
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")

    # Calculate statistics
    avg_speedup = np.mean(speedup)
    std_speedup = np.std(speedup)
    avg_llm_time = np.mean(llm_per_tweet)
    n_tweets = human_time['combined']['n_tweets']
    human_total_hours = human_time['combined']['time_hours']
    llm_total_hours = [llm_times[k]['avg_seconds_per_tweet'] * n_tweets / 3600
                       for k in k_values]
    time_saved_hours = human_total_hours - np.mean(llm_total_hours)
    time_saved_pct = (time_saved_hours / human_total_hours) * 100

    return {
        'avg_speedup': avg_speedup,
        'std_speedup': std_speedup,
        'time_saved_pct': time_saved_pct
    }

def main():
    print("="*70)
    print("Human-LLM Processing Time Comparison Analysis")
    print("="*70)
    print()

    # Calculate human coding time
    print("Estimating human coding time...")
    human_time = estimate_human_coding_time()

    print(f"\nHuman Coding Estimates:")
    print(f"  Pilot ({human_time['pilot']['n_tweets']} tweets): "
          f"{human_time['pilot']['time_hours']:.1f} hours")
    print(f"  Main ({human_time['main']['n_tweets']} tweets): "
          f"{human_time['main']['time_hours']:.1f} hours")
    print(f"  Combined ({human_time['combined']['n_tweets']} tweets): "
          f"{human_time['combined']['time_hours']:.1f} hours")

    # Calculate LLM time
    print("\nCalculating LLM processing time from evaluation logs...")
    llm_times = calculate_llm_time()

    print("\nLLM Classification Times:")
    for k, data in llm_times.items():
        if data:
            print(f"  k={k}: {data['avg_seconds_per_tweet']:.2f} sec/tweet "
                  f"({data['n_tweets']} tweets, {data['total_time_seconds']/3600:.2f} hours total)")

    # Create comparison figure
    print("\nGenerating comparison figure...")
    output_path = 'survey/time_comparison/processing_time_comparison.png'
    stats = create_comparison_figure(human_time, llm_times, output_path)

    # Print TBD values for the paper
    print("\n" + "="*70)
    print("VALUES TO FILL IN DRAFT.MD:")
    print("="*70)
    print(f"LLM classification was {stats['avg_speedup']:.1f}× faster than human coding")
    print(f"  (Time saved: {stats['time_saved_pct']:.1f}%)")
    print(f"Standard deviation of speedup: {stats['std_speedup']:.2f}×")
    print()
    print("Replace in draft.md:")
    print(f"  [TBD]% faster → {stats['time_saved_pct']:.1f}% faster")
    print(f"  (SD = [TBD]%) → (SD = {stats['std_speedup']:.2f}×)")
    print()
    print(f"Figure saved to: {output_path}")
    print("="*70)

    # Save detailed results
    results = {
        'human_time': human_time,
        'llm_times': {k: {key: val for key, val in v.items() if key not in ['start', 'end']}
                      if v else None for k, v in llm_times.items()},
        'statistics': stats
    }

    with open('survey/time_comparison/time_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nDetailed results saved to: survey/time_comparison/time_comparison_results.json")

if __name__ == '__main__':
    main()
