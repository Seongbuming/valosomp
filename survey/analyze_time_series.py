#!/usr/bin/env python3
"""
Time-Series Analysis: Emotion & Intent Evolution
- Emotion/Intent patterns over time since crisis start
- Human-induced vs Natural disaster temporal comparison
- Involvement level temporal patterns
- Visualizations
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import mannwhitneyu, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("Time-Series Analysis: Emotion & Intent Evolution")
print("=" * 80)

# ============================================================================
# 1. Data Loading
# ============================================================================

print("\n[1] Loading data with timestamps...")

df = pd.read_csv('emotional_and_intent/combined_analysis_data_with_timestamps.csv')
print(f"  Total data: {len(df):,} tweets")

# Parse datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Filter: exclude Mixed category, keep only with timestamps
df_analysis = df[(df['crisis_category'].isin(['Human-induced', 'Natural'])) &
                  (df['time_since_start'].notna())].copy()

print(f"  Analysis data (Human/Natural with timestamps): {len(df_analysis):,} tweets")

# Emotion/Intent columns
emotion_cols = [c for c in df.columns if c.startswith('emotion_')]
intent_cols = [c for c in df.columns if c.startswith('intent_')]

# Create time bins (hours since crisis start)
time_bins = [0, 6, 12, 24, 48, 72, 168, df_analysis['time_since_start'].max()]  # 0-6h, 6-12h, 12-24h, 24-48h, 48-72h, 72-168h (1 week), >1 week
time_labels = ['0-6h', '6-12h', '12-24h', '24-48h', '48-72h', '72h-1w', '>1w']
df_analysis['time_bin'] = pd.cut(df_analysis['time_since_start'], bins=time_bins, labels=time_labels, include_lowest=True)

print(f"\nTime bin distribution:")
print(df_analysis['time_bin'].value_counts().sort_index())

# ============================================================================
# 2. Overall Temporal Patterns
# ============================================================================

print("\n" + "=" * 80)
print("2. Overall Temporal Patterns (All Crisis Types)")
print("=" * 80)

# Emotion over time
print("\n[A] Emotion Evolution by Time Bin:")
emotion_time_results = []

for time_bin in time_labels:
    bin_data = df_analysis[df_analysis['time_bin'] == time_bin]
    if len(bin_data) > 0:
        row = {'Time': time_bin, 'N': len(bin_data)}
        for emotion in emotion_cols:
            row[emotion.replace('emotion_', '')] = bin_data[emotion].mean()
        emotion_time_results.append(row)

emotion_time_df = pd.DataFrame(emotion_time_results)
print(tabulate(emotion_time_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# Intent over time
print("\n[B] Intent Evolution by Time Bin:")
intent_time_results = []

for time_bin in time_labels:
    bin_data = df_analysis[df_analysis['time_bin'] == time_bin]
    if len(bin_data) > 0:
        row = {'Time': time_bin, 'N': len(bin_data)}
        for intent in intent_cols:
            row[intent.replace('intent_', '')] = bin_data[intent].mean()
        intent_time_results.append(row)

intent_time_df = pd.DataFrame(intent_time_results)
print(tabulate(intent_time_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# ============================================================================
# 3. Temporal Patterns by Crisis Type
# ============================================================================

print("\n" + "=" * 80)
print("3. Temporal Patterns: Human-induced vs Natural Disasters")
print("=" * 80)

# Emotion evolution by crisis type
print("\n[A] Fear Evolution by Crisis Type:")
fear_comparison = []

for time_bin in time_labels:
    human_data = df_analysis[(df_analysis['time_bin'] == time_bin) & (df_analysis['crisis_category'] == 'Human-induced')]
    natural_data = df_analysis[(df_analysis['time_bin'] == time_bin) & (df_analysis['crisis_category'] == 'Natural')]

    if len(human_data) > 0 and len(natural_data) > 0:
        fear_comparison.append({
            'Time': time_bin,
            'Human-induced (N)': len(human_data),
            'Human Fear': human_data['emotion_fear'].mean(),
            'Natural (N)': len(natural_data),
            'Natural Fear': natural_data['emotion_fear'].mean(),
            'Diff': natural_data['emotion_fear'].mean() - human_data['emotion_fear'].mean()
        })

fear_df = pd.DataFrame(fear_comparison)
print(tabulate(fear_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# Sadness evolution by crisis type
print("\n[B] Sadness Evolution by Crisis Type:")
sadness_comparison = []

for time_bin in time_labels:
    human_data = df_analysis[(df_analysis['time_bin'] == time_bin) & (df_analysis['crisis_category'] == 'Human-induced')]
    natural_data = df_analysis[(df_analysis['time_bin'] == time_bin) & (df_analysis['crisis_category'] == 'Natural')]

    if len(human_data) > 0 and len(natural_data) > 0:
        sadness_comparison.append({
            'Time': time_bin,
            'Human Sadness': human_data['emotion_sadness'].mean(),
            'Natural Sadness': natural_data['emotion_sadness'].mean(),
            'Diff': human_data['emotion_sadness'].mean() - natural_data['emotion_sadness'].mean()
        })

sadness_df = pd.DataFrame(sadness_comparison)
print(tabulate(sadness_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# Intent evolution by crisis type
print("\n[C] Call-to-Action Evolution by Crisis Type:")
cta_comparison = []

for time_bin in time_labels:
    human_data = df_analysis[(df_analysis['time_bin'] == time_bin) & (df_analysis['crisis_category'] == 'Human-induced')]
    natural_data = df_analysis[(df_analysis['time_bin'] == time_bin) & (df_analysis['crisis_category'] == 'Natural')]

    if len(human_data) > 0 and len(natural_data) > 0:
        cta_comparison.append({
            'Time': time_bin,
            'Human CTA': human_data['intent_call-to-action'].mean(),
            'Natural CTA': natural_data['intent_call-to-action'].mean(),
            'Diff': human_data['intent_call-to-action'].mean() - natural_data['intent_call-to-action'].mean()
        })

cta_df = pd.DataFrame(cta_comparison)
print(tabulate(cta_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# ============================================================================
# 4. Correlation with Time
# ============================================================================

print("\n" + "=" * 80)
print("4. Correlation with Time Since Crisis Start")
print("=" * 80)

print("\n[A] Emotion correlations (Spearman's rho):")
emotion_corr_results = []

for emotion in emotion_cols:
    data = df_analysis[['time_since_start', emotion]].dropna()
    if len(data) > 0:
        rho, p_value = spearmanr(data['time_since_start'], data[emotion])
        emotion_corr_results.append({
            'Emotion': emotion.replace('emotion_', ''),
            'rho': rho,
            'p-value': p_value,
            'Sig': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
        })

emotion_corr_df = pd.DataFrame(emotion_corr_results)
print(tabulate(emotion_corr_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

print("\n[B] Intent correlations (Spearman's rho):")
intent_corr_results = []

for intent in intent_cols:
    data = df_analysis[['time_since_start', intent]].dropna()
    if len(data) > 0:
        rho, p_value = spearmanr(data['time_since_start'], data[intent])
        intent_corr_results.append({
            'Intent': intent.replace('intent_', ''),
            'rho': rho,
            'p-value': p_value,
            'Sig': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
        })

intent_corr_df = pd.DataFrame(intent_corr_results)
print(tabulate(intent_corr_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# ============================================================================
# 5. Save Results
# ============================================================================

output_dir = 'emotional_and_intent/timeseries_results'
os.makedirs(output_dir, exist_ok=True)

emotion_time_df.to_csv(f'{output_dir}/emotion_over_time.csv', index=False)
intent_time_df.to_csv(f'{output_dir}/intent_over_time.csv', index=False)
fear_df.to_csv(f'{output_dir}/fear_by_crisis_time.csv', index=False)
sadness_df.to_csv(f'{output_dir}/sadness_by_crisis_time.csv', index=False)
cta_df.to_csv(f'{output_dir}/cta_by_crisis_time.csv', index=False)
emotion_corr_df.to_csv(f'{output_dir}/emotion_time_correlations.csv', index=False)
intent_corr_df.to_csv(f'{output_dir}/intent_time_correlations.csv', index=False)

print(f"\n결과 저장 완료: {output_dir}/")

# ============================================================================
# 6. Visualizations
# ============================================================================

print("\n" + "=" * 80)
print("6. Visualizations")
print("=" * 80)

# Figure 1: Emotion evolution over time (line plot)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, emotion in enumerate(emotion_cols):
    ax = axes[idx]
    emotion_name = emotion.replace('emotion_', '')

    # Plot overall trend
    time_means = emotion_time_df[emotion_name].values
    ax.plot(range(len(time_labels)), time_means, marker='o', linewidth=2, markersize=8, label='Overall')

    # Plot by crisis type
    human_means = []
    natural_means = []
    for time_bin in time_labels:
        human_val = df_analysis[(df_analysis['time_bin']==time_bin) & (df_analysis['crisis_category']=='Human-induced')][emotion].mean()
        natural_val = df_analysis[(df_analysis['time_bin']==time_bin) & (df_analysis['crisis_category']=='Natural')][emotion].mean()
        human_means.append(human_val)
        natural_means.append(natural_val)

    ax.plot(range(len(time_labels)), human_means, marker='s', linewidth=2, markersize=6,
            alpha=0.7, label='Human-induced', color='#e74c3c')
    ax.plot(range(len(time_labels)), natural_means, marker='^', linewidth=2, markersize=6,
            alpha=0.7, label='Natural', color='#3498db')

    ax.set_xlabel('Time Since Crisis Start')
    ax.set_ylabel('Mean Score')
    ax.set_title(f'{emotion_name.capitalize()}', fontweight='bold')
    ax.set_xticks(range(len(time_labels)))
    ax.set_xticklabels(time_labels, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/emotion_evolution.png', dpi=300, bbox_inches='tight')
print(f"  저장: emotion_evolution.png")

# Figure 2: Intent evolution over time (line plot)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, intent in enumerate(intent_cols):
    ax = axes[idx]
    intent_name = intent.replace('intent_', '')

    # Plot overall trend
    time_means = intent_time_df[intent_name].values
    ax.plot(range(len(time_labels)), time_means, marker='o', linewidth=2, markersize=8, label='Overall')

    # Plot by crisis type
    human_means = []
    natural_means = []
    for time_bin in time_labels:
        human_val = df_analysis[(df_analysis['time_bin']==time_bin) & (df_analysis['crisis_category']=='Human-induced')][intent].mean()
        natural_val = df_analysis[(df_analysis['time_bin']==time_bin) & (df_analysis['crisis_category']=='Natural')][intent].mean()
        human_means.append(human_val)
        natural_means.append(natural_val)

    ax.plot(range(len(time_labels)), human_means, marker='s', linewidth=2, markersize=6,
            alpha=0.7, label='Human-induced', color='#e74c3c')
    ax.plot(range(len(time_labels)), natural_means, marker='^', linewidth=2, markersize=6,
            alpha=0.7, label='Natural', color='#3498db')

    ax.set_xlabel('Time Since Crisis Start')
    ax.set_ylabel('Mean Score')
    title_name = intent_name.replace('-', ' ').title()
    ax.set_title(f'{title_name}', fontweight='bold')
    ax.set_xticks(range(len(time_labels)))
    ax.set_xticklabels(time_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/intent_evolution.png', dpi=300, bbox_inches='tight')
print(f"  저장: intent_evolution.png")

# Figure 3: Key emotions comparison (Fear, Sadness)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Fear
ax = axes[0]
for crisis_type, color, marker in [('Human-induced', '#e74c3c', 's'), ('Natural', '#3498db', '^')]:
    means = []
    for time_bin in time_labels:
        val = df_analysis[(df_analysis['time_bin']==time_bin) & (df_analysis['crisis_category']==crisis_type)]['emotion_fear'].mean()
        means.append(val)
    ax.plot(range(len(time_labels)), means, marker=marker, linewidth=2.5, markersize=10,
            label=crisis_type, color=color, alpha=0.8)

ax.set_xlabel('Time Since Crisis Start', fontsize=12)
ax.set_ylabel('Mean Fear Score', fontsize=12)
ax.set_title('Fear Evolution by Crisis Type', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Sadness
ax = axes[1]
for crisis_type, color, marker in [('Human-induced', '#e74c3c', 's'), ('Natural', '#3498db', '^')]:
    means = []
    for time_bin in time_labels:
        val = df_analysis[(df_analysis['time_bin']==time_bin) & (df_analysis['crisis_category']==crisis_type)]['emotion_sadness'].mean()
        means.append(val)
    ax.plot(range(len(time_labels)), means, marker=marker, linewidth=2.5, markersize=10,
            label=crisis_type, color=color, alpha=0.8)

ax.set_xlabel('Time Since Crisis Start', fontsize=12)
ax.set_ylabel('Mean Sadness Score', fontsize=12)
ax.set_title('Sadness Evolution by Crisis Type', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/fear_sadness_evolution.png', dpi=300, bbox_inches='tight')
print(f"  저장: fear_sadness_evolution.png")

print("\n" + "=" * 80)
print("Time-Series Analysis Complete!")
print("=" * 80)
print(f"\n결과 저장 위치: {output_dir}/")
print("  - CSV: 7 files")
print("  - Figures: 3 PNG files")
