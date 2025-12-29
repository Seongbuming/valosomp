#!/usr/bin/env python3
"""
Crisis Type Analysis: Human-induced vs Natural Disasters
- Emotion & Intent patterns by crisis category
- Involvement level × Crisis type interaction
- Statistical significance testing
- Comprehensive visualizations
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("Crisis Type Analysis: Human-induced vs Natural Disasters")
print("=" * 80)

# ============================================================================
# 1. Data Loading & Merging
# ============================================================================

print("\n[1] Loading data...")

# Load combined analysis data
combined_df = pd.read_csv('emotional_and_intent/combined_analysis_data.csv')
print(f"  Combined data: {len(combined_df)} tweets")

# Load crisis type data
crisis_df = pd.read_csv('emotional_and_intent/2_crisis_type_analysis/survey_with_crisis_type.csv')
print(f"  Crisis type data: {len(crisis_df)} tweets")

# Merge on Tweet Text (more reliable than Tweet ID due to precision issues)
df = combined_df.merge(
    crisis_df[['Tweet Text', 'crisis_category', 'crisis_type']],
    on='Tweet Text',
    how='inner'
)
print(f"  Merged data: {len(df)} tweets (with crisis type info)")

# Crisis category distribution
print("\nCrisis Category Distribution:")
print(df['crisis_category'].value_counts())
print(f"\n  Human-induced: {(df['crisis_category']=='Human-induced').sum()} ({(df['crisis_category']=='Human-induced').sum()/len(df)*100:.1f}%)")
print(f"  Natural: {(df['crisis_category']=='Natural').sum()} ({(df['crisis_category']=='Natural').sum()/len(df)*100:.1f}%)")

# Emotion/Intent columns
emotion_cols = [c for c in df.columns if c.startswith('emotion_')]
intent_cols = [c for c in df.columns if c.startswith('intent_')]

# ============================================================================
# 2. Emotion Patterns by Crisis Type
# ============================================================================

print("\n" + "=" * 80)
print("2. Emotion Patterns by Crisis Category")
print("=" * 80)

emotion_results = []
for emotion in emotion_cols:
    human_mean = df[df['crisis_category']=='Human-induced'][emotion].mean()
    natural_mean = df[df['crisis_category']=='Natural'][emotion].mean()

    # Mann-Whitney U test
    human_data = df[df['crisis_category']=='Human-induced'][emotion]
    natural_data = df[df['crisis_category']=='Natural'][emotion]
    stat, p_value = mannwhitneyu(human_data, natural_data, alternative='two-sided')

    emotion_name = emotion.replace('emotion_', '')
    emotion_results.append({
        'Emotion': emotion_name,
        'Human-induced': human_mean,
        'Natural': natural_mean,
        'Diff': natural_mean - human_mean,
        'p-value': p_value,
        'Sig': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
    })

emotion_df = pd.DataFrame(emotion_results)
print(tabulate(emotion_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# ============================================================================
# 3. Intent Patterns by Crisis Type
# ============================================================================

print("\n" + "=" * 80)
print("3. Intent Patterns by Crisis Category")
print("=" * 80)

intent_results = []
for intent in intent_cols:
    human_mean = df[df['crisis_category']=='Human-induced'][intent].mean()
    natural_mean = df[df['crisis_category']=='Natural'][intent].mean()

    human_data = df[df['crisis_category']=='Human-induced'][intent]
    natural_data = df[df['crisis_category']=='Natural'][intent]
    stat, p_value = mannwhitneyu(human_data, natural_data, alternative='two-sided')

    intent_name = intent.replace('intent_', '')
    intent_results.append({
        'Intent': intent_name,
        'Human-induced': human_mean,
        'Natural': natural_mean,
        'Diff': natural_mean - human_mean,
        'p-value': p_value,
        'Sig': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
    })

intent_df = pd.DataFrame(intent_results)
print(tabulate(intent_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# ============================================================================
# 4. Involvement Level × Crisis Type Interaction
# ============================================================================

print("\n" + "=" * 80)
print("4. Involvement Level × Crisis Type Interaction")
print("=" * 80)

# For each dimension, compare High vs Low within each crisis category
for dimension in ['I', 'You', 'We']:
    print(f"\n[{dimension}-Involvement × Crisis Type]")

    interaction_results = []

    # Human-induced
    human_low = df[(df['crisis_category']=='Human-induced') & (df[f'{dimension}_level']=='Low')]
    human_high = df[(df['crisis_category']=='Human-induced') & (df[f'{dimension}_level']=='High')]

    # Natural
    natural_low = df[(df['crisis_category']=='Natural') & (df[f'{dimension}_level']=='Low')]
    natural_high = df[(df['crisis_category']=='Natural') & (df[f'{dimension}_level']=='High')]

    print(f"\n  Sample sizes:")
    print(f"    Human-induced: Low={len(human_low)}, High={len(human_high)}")
    print(f"    Natural: Low={len(natural_low)}, High={len(natural_high)}")

    # Dominant emotion for each group
    print(f"\n  Dominant Emotions:")
    for category, grp in [('Human-Low', human_low), ('Human-High', human_high),
                          ('Natural-Low', natural_low), ('Natural-High', natural_high)]:
        if len(grp) > 0:
            emotion_means = {e.replace('emotion_', ''): grp[e].mean() for e in emotion_cols}
            dominant = max(emotion_means, key=emotion_means.get)
            print(f"    {category:15s}: {dominant:10s} ({emotion_means[dominant]:.3f})")

    # Dominant intent for each group
    print(f"\n  Dominant Intents:")
    for category, grp in [('Human-Low', human_low), ('Human-High', human_high),
                          ('Natural-Low', natural_low), ('Natural-High', natural_high)]:
        if len(grp) > 0:
            intent_means = {i.replace('intent_', ''): grp[i].mean() for i in intent_cols}
            dominant = max(intent_means, key=intent_means.get)
            print(f"    {category:15s}: {dominant:18s} ({intent_means[dominant]:.3f})")

# ============================================================================
# 5. Crisis Type Distribution by Involvement Level
# ============================================================================

print("\n" + "=" * 80)
print("5. Crisis Type Distribution by Involvement Level")
print("=" * 80)

for dimension in ['I', 'You', 'We']:
    print(f"\n[{dimension}-Involvement]")

    # Cross-tabulation
    crosstab = pd.crosstab(
        df[f'{dimension}_level'],
        df['crisis_category'],
        normalize='index'
    ) * 100

    print(tabulate(crosstab, headers='keys', tablefmt='pipe', floatfmt='.1f'))

    # Chi-square test
    contingency = pd.crosstab(df[f'{dimension}_level'], df['crisis_category'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print(f"  χ²({dof}) = {chi2:.2f}, p = {p_value:.4f} {'***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))}")

# ============================================================================
# 6. Save Results
# ============================================================================

output_dir = 'emotional_and_intent/crisis_type_results'
os.makedirs(output_dir, exist_ok=True)

# Save CSVs
emotion_df.to_csv(f'{output_dir}/emotion_by_crisis_category.csv', index=False)
intent_df.to_csv(f'{output_dir}/intent_by_crisis_category.csv', index=False)

print(f"\n결과 저장 완료: {output_dir}/")

# ============================================================================
# 7. Visualizations
# ============================================================================

print("\n" + "=" * 80)
print("7. Visualizations")
print("=" * 80)

# Helper function for significance markers
def add_significance_markers(ax, x_positions, vals1, vals2, p_values, offset=0.02):
    """Add significance markers above bars"""
    for i, (v1, v2, p) in enumerate(zip(vals1, vals2, p_values)):
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''

        if sig:
            y_pos = max(v1, v2) + offset
            ax.text(x_positions[i], y_pos, sig, ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

# Figure 1: Emotion by Crisis Category
fig, ax = plt.subplots(figsize=(12, 6))

emotions = [e.replace('emotion_', '') for e in emotion_cols]
human_means = [df[df['crisis_category']=='Human-induced'][e].mean() for e in emotion_cols]
natural_means = [df[df['crisis_category']=='Natural'][e].mean() for e in emotion_cols]
p_values = [r['p-value'] for r in emotion_results]

x = np.arange(len(emotions))
width = 0.35

bars1 = ax.bar(x - width/2, human_means, width, label='Human-induced', alpha=0.8, color='#e74c3c')
bars2 = ax.bar(x + width/2, natural_means, width, label='Natural', alpha=0.8, color='#3498db')

add_significance_markers(ax, x, human_means, natural_means, p_values)

ax.set_xlabel('Emotion', fontsize=12)
ax.set_ylabel('Mean Score', fontsize=12)
ax.set_title('Emotion Patterns by Crisis Category', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(emotions, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(human_means), max(natural_means)) * 1.15)

plt.tight_layout()
plt.savefig(f'{output_dir}/emotion_by_crisis_category.png', dpi=300, bbox_inches='tight')
print(f"  저장: emotion_by_crisis_category.png")

# Figure 2: Intent by Crisis Category
fig, ax = plt.subplots(figsize=(12, 6))

intents = [i.replace('intent_', '') for i in intent_cols]
human_means = [df[df['crisis_category']=='Human-induced'][i].mean() for i in intent_cols]
natural_means = [df[df['crisis_category']=='Natural'][i].mean() for i in intent_cols]
p_values = [r['p-value'] for r in intent_results]

x = np.arange(len(intents))
width = 0.35

bars1 = ax.bar(x - width/2, human_means, width, label='Human-induced', alpha=0.8, color='#e74c3c')
bars2 = ax.bar(x + width/2, natural_means, width, label='Natural', alpha=0.8, color='#3498db')

add_significance_markers(ax, x, human_means, natural_means, p_values)

ax.set_xlabel('Intent', fontsize=12)
ax.set_ylabel('Mean Score', fontsize=12)
ax.set_title('Intent Patterns by Crisis Category', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(intents, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(human_means), max(natural_means)) * 1.15)

plt.tight_layout()
plt.savefig(f'{output_dir}/intent_by_crisis_category.png', dpi=300, bbox_inches='tight')
print(f"  저장: intent_by_crisis_category.png")

# Figure 3: Heatmap - Emotion by Crisis × Involvement
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for row_idx, crisis in enumerate(['Human-induced', 'Natural']):
    for col_idx, dimension in enumerate(['I', 'You', 'We']):
        ax = axes[row_idx, col_idx]

        crisis_df_subset = df[df['crisis_category'] == crisis]

        low_means = [crisis_df_subset[crisis_df_subset[f'{dimension}_level']=='Low'][e].mean()
                     for e in emotion_cols]
        high_means = [crisis_df_subset[crisis_df_subset[f'{dimension}_level']=='High'][e].mean()
                      for e in emotion_cols]

        emotions = [e.replace('emotion_', '') for e in emotion_cols]

        # Create heatmap data
        heatmap_data = pd.DataFrame(
            [low_means, high_means],
            index=['Low', 'High'],
            columns=emotions
        )

        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': 'Mean Score'}, vmin=0, vmax=0.5)
        ax.set_title(f'{crisis} - {dimension}-Involvement', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

plt.tight_layout()
plt.savefig(f'{output_dir}/emotion_heatmap_crisis_involvement.png', dpi=300, bbox_inches='tight')
print(f"  저장: emotion_heatmap_crisis_involvement.png")

# Figure 4: Heatmap - Intent by Crisis × Involvement
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for row_idx, crisis in enumerate(['Human-induced', 'Natural']):
    for col_idx, dimension in enumerate(['I', 'You', 'We']):
        ax = axes[row_idx, col_idx]

        crisis_df_subset = df[df['crisis_category'] == crisis]

        low_means = [crisis_df_subset[crisis_df_subset[f'{dimension}_level']=='Low'][i].mean()
                     for i in intent_cols]
        high_means = [crisis_df_subset[crisis_df_subset[f'{dimension}_level']=='High'][i].mean()
                      for i in intent_cols]

        intents = [i.replace('intent_', '') for i in intent_cols]

        # Create heatmap data
        heatmap_data = pd.DataFrame(
            [low_means, high_means],
            index=['Low', 'High'],
            columns=intents
        )

        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu',
                   ax=ax, cbar_kws={'label': 'Mean Score'}, vmin=0, vmax=1.0)
        ax.set_title(f'{crisis} - {dimension}-Involvement', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

plt.tight_layout()
plt.savefig(f'{output_dir}/intent_heatmap_crisis_involvement.png', dpi=300, bbox_inches='tight')
print(f"  저장: intent_heatmap_crisis_involvement.png")

# Figure 5: Involvement Distribution by Crisis Type
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, dimension in enumerate(['I', 'You', 'We']):
    ax = axes[idx]

    # Prepare data
    human_low = len(df[(df['crisis_category']=='Human-induced') & (df[f'{dimension}_level']=='Low')])
    human_high = len(df[(df['crisis_category']=='Human-induced') & (df[f'{dimension}_level']=='High')])
    natural_low = len(df[(df['crisis_category']=='Natural') & (df[f'{dimension}_level']=='Low')])
    natural_high = len(df[(df['crisis_category']=='Natural') & (df[f'{dimension}_level']=='High')])

    labels = ['Low', 'High']
    human_counts = [human_low, human_high]
    natural_counts = [natural_low, natural_high]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, human_counts, width, label='Human-induced', alpha=0.8, color='#e74c3c')
    ax.bar(x + width/2, natural_counts, width, label='Natural', alpha=0.8, color='#3498db')

    ax.set_xlabel('Involvement Level', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{dimension}-Involvement Distribution', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/involvement_distribution_by_crisis.png', dpi=300, bbox_inches='tight')
print(f"  저장: involvement_distribution_by_crisis.png")

print("\n" + "=" * 80)
print("Crisis Type Analysis Complete!")
print("=" * 80)
print(f"\n결과 저장 위치: {output_dir}/")
print("  - CSV: emotion_by_crisis_category.csv, intent_by_crisis_category.csv")
print("  - Figures: 5 PNG files")
