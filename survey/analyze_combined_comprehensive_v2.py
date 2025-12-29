#!/usr/bin/env python3
"""
통합 데이터 종합 분석 (개선 버전)
1. I/You/We-Involvement 패턴 (모두 포함)
2. 세 차원 비교 테이블
3. 통계적 유의성 검정 (Mann-Whitney U)
4. 시각화 (통계적 유의성 표시 포함)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("통합 데이터 종합 분석 (개선 버전)")
print("=" * 80)

# ============================================================================
# 데이터 로드
# ============================================================================

df = pd.read_csv('emotional_and_intent/combined_analysis_data.csv')
print(f"\n총 데이터: {len(df)} tweets")
print(f"  Pilot: {(df['source_dataset']=='Pilot').sum()}")
print(f"  Main: {(df['source_dataset']=='Main').sum()}")

# Emotion/Intent 컬럼
emotion_cols = [c for c in df.columns if c.startswith('emotion_')]
intent_cols = [c for c in df.columns if c.startswith('intent_')]

# ============================================================================
# 1. I-Involvement 패턴
# ============================================================================

print("\n" + "=" * 80)
print("1. I-Involvement 패턴 분석")
print("=" * 80)

print("\n[A] Emotion by I-Involvement")
i_emotion_results = []
for emotion in emotion_cols:
    low_mean = df[df['I_level']=='Low'][emotion].mean()
    high_mean = df[df['I_level']=='High'][emotion].mean()

    # Mann-Whitney U test
    low_data = df[df['I_level']=='Low'][emotion]
    high_data = df[df['I_level']=='High'][emotion]
    stat, p_value = mannwhitneyu(low_data, high_data, alternative='two-sided')

    emotion_name = emotion.replace('emotion_', '')
    i_emotion_results.append({
        'Emotion': emotion_name,
        'Low': low_mean,
        'High': high_mean,
        'Diff': high_mean - low_mean,
        'p-value': p_value,
        'Sig': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
    })

i_emotion_df = pd.DataFrame(i_emotion_results)
print(tabulate(i_emotion_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

print("\n[B] Intent by I-Involvement")
i_intent_results = []
for intent in intent_cols:
    low_mean = df[df['I_level']=='Low'][intent].mean()
    high_mean = df[df['I_level']=='High'][intent].mean()

    low_data = df[df['I_level']=='Low'][intent]
    high_data = df[df['I_level']=='High'][intent]
    stat, p_value = mannwhitneyu(low_data, high_data, alternative='two-sided')

    intent_name = intent.replace('intent_', '')
    i_intent_results.append({
        'Intent': intent_name,
        'Low': low_mean,
        'High': high_mean,
        'Diff': high_mean - low_mean,
        'p-value': p_value,
        'Sig': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
    })

i_intent_df = pd.DataFrame(i_intent_results)
print(tabulate(i_intent_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# ============================================================================
# 2. You-Involvement 패턴
# ============================================================================

print("\n" + "=" * 80)
print("2. You-Involvement 패턴 분석")
print("=" * 80)

print("\n[A] Emotion by You-Involvement")
you_emotion_results = []
for emotion in emotion_cols:
    low_mean = df[df['You_level']=='Low'][emotion].mean()
    high_mean = df[df['You_level']=='High'][emotion].mean()

    # Mann-Whitney U test
    low_data = df[df['You_level']=='Low'][emotion]
    high_data = df[df['You_level']=='High'][emotion]
    stat, p_value = mannwhitneyu(low_data, high_data, alternative='two-sided')

    emotion_name = emotion.replace('emotion_', '')
    you_emotion_results.append({
        'Emotion': emotion_name,
        'Low': low_mean,
        'High': high_mean,
        'Diff': high_mean - low_mean,
        'p-value': p_value,
        'Sig': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
    })

you_emotion_df = pd.DataFrame(you_emotion_results)
print(tabulate(you_emotion_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

print("\n[B] Intent by You-Involvement")
you_intent_results = []
for intent in intent_cols:
    low_mean = df[df['You_level']=='Low'][intent].mean()
    high_mean = df[df['You_level']=='High'][intent].mean()

    low_data = df[df['You_level']=='Low'][intent]
    high_data = df[df['You_level']=='High'][intent]
    stat, p_value = mannwhitneyu(low_data, high_data, alternative='two-sided')

    intent_name = intent.replace('intent_', '')
    you_intent_results.append({
        'Intent': intent_name,
        'Low': low_mean,
        'High': high_mean,
        'Diff': high_mean - low_mean,
        'p-value': p_value,
        'Sig': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
    })

you_intent_df = pd.DataFrame(you_intent_results)
print(tabulate(you_intent_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# ============================================================================
# 3. We-Involvement 패턴
# ============================================================================

print("\n" + "=" * 80)
print("3. We-Involvement 패턴 분석")
print("=" * 80)

print("\n[A] Emotion by We-Involvement")
we_emotion_results = []
for emotion in emotion_cols:
    low_mean = df[df['We_level']=='Low'][emotion].mean()
    high_mean = df[df['We_level']=='High'][emotion].mean()

    low_data = df[df['We_level']=='Low'][emotion]
    high_data = df[df['We_level']=='High'][emotion]
    stat, p_value = mannwhitneyu(low_data, high_data, alternative='two-sided')

    emotion_name = emotion.replace('emotion_', '')
    we_emotion_results.append({
        'Emotion': emotion_name,
        'Low': low_mean,
        'High': high_mean,
        'Diff': high_mean - low_mean,
        'p-value': p_value,
        'Sig': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
    })

we_emotion_df = pd.DataFrame(we_emotion_results)
print(tabulate(we_emotion_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

print("\n[B] Intent by We-Involvement")
we_intent_results = []
for intent in intent_cols:
    low_mean = df[df['We_level']=='Low'][intent].mean()
    high_mean = df[df['We_level']=='High'][intent].mean()

    low_data = df[df['We_level']=='Low'][intent]
    high_data = df[df['We_level']=='High'][intent]
    stat, p_value = mannwhitneyu(low_data, high_data, alternative='two-sided')

    intent_name = intent.replace('intent_', '')
    we_intent_results.append({
        'Intent': intent_name,
        'Low': low_mean,
        'High': high_mean,
        'Diff': high_mean - low_mean,
        'p-value': p_value,
        'Sig': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
    })

we_intent_df = pd.DataFrame(we_intent_results)
print(tabulate(we_intent_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# ============================================================================
# 4. 세 차원 비교 테이블
# ============================================================================

print("\n" + "=" * 80)
print("4. 세 차원 비교: Dominant Emotion & Intent (High level)")
print("=" * 80)

comparison_results = []

for dimension in ['I', 'You', 'We']:
    high_df = df[df[f'{dimension}_level'] == 'High']

    # Dominant emotion
    emotion_means = {e.replace('emotion_', ''): high_df[e].mean() for e in emotion_cols}
    dominant_emotion = max(emotion_means, key=emotion_means.get)
    emotion_score = emotion_means[dominant_emotion]

    # Dominant intent
    intent_means = {i.replace('intent_', ''): high_df[i].mean() for i in intent_cols}
    dominant_intent = max(intent_means, key=intent_means.get)
    intent_score = intent_means[dominant_intent]

    comparison_results.append({
        'Dimension': f'{dimension}-High',
        'N': len(high_df),
        'Dominant_Emotion': dominant_emotion,
        'Emotion_Score': emotion_score,
        'Dominant_Intent': dominant_intent,
        'Intent_Score': intent_score
    })

comparison_df = pd.DataFrame(comparison_results)
print(tabulate(comparison_df, headers='keys', tablefmt='pipe', floatfmt='.3f'))

# ============================================================================
# 5. 결과 저장
# ============================================================================

output_dir = 'emotional_and_intent/combined_results'
os.makedirs(output_dir, exist_ok=True)

# CSV 저장
i_emotion_df.to_csv(f'{output_dir}/i_emotion_patterns.csv', index=False)
i_intent_df.to_csv(f'{output_dir}/i_intent_patterns.csv', index=False)
you_emotion_df.to_csv(f'{output_dir}/you_emotion_patterns.csv', index=False)
you_intent_df.to_csv(f'{output_dir}/you_intent_patterns.csv', index=False)
we_emotion_df.to_csv(f'{output_dir}/we_emotion_patterns.csv', index=False)
we_intent_df.to_csv(f'{output_dir}/we_intent_patterns.csv', index=False)
comparison_df.to_csv(f'{output_dir}/dimension_comparison.csv', index=False)

print(f"\n결과 저장 완료: {output_dir}/")

# ============================================================================
# 6. 시각화 (통계적 유의성 표시 포함)
# ============================================================================

print("\n" + "=" * 80)
print("6. 시각화 생성 (통계적 유의성 표시 포함)")
print("=" * 80)

# Helper function to add significance markers
def add_significance_markers(ax, x_positions, low_means, high_means, p_values, width):
    """Add significance markers above bars"""
    for i, (low, high, p) in enumerate(zip(low_means, high_means, p_values)):
        # Determine significance level
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''

        if sig:
            # Position at the top of the higher bar
            y_pos = max(low, high) + 0.02
            ax.text(x_positions[i], y_pos, sig, ha='center', va='bottom', fontsize=10, fontweight='bold')

# Figure 1: Emotion comparison across dimensions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, dimension in enumerate(['I', 'You', 'We']):
    ax = axes[idx]

    low_df = df[df[f'{dimension}_level'] == 'Low']
    high_df = df[df[f'{dimension}_level'] == 'High']

    emotions = [e.replace('emotion_', '') for e in emotion_cols]
    low_means = [low_df[e].mean() for e in emotion_cols]
    high_means = [high_df[e].mean() for e in emotion_cols]

    # Calculate p-values for each emotion
    p_values = []
    for emotion in emotion_cols:
        low_data = df[df[f'{dimension}_level']=='Low'][emotion]
        high_data = df[df[f'{dimension}_level']=='High'][emotion]
        _, p_value = mannwhitneyu(low_data, high_data, alternative='two-sided')
        p_values.append(p_value)

    x = np.arange(len(emotions))
    width = 0.35

    ax.bar(x - width/2, low_means, width, label='Low', alpha=0.8)
    ax.bar(x + width/2, high_means, width, label='High', alpha=0.8)

    # Add significance markers
    add_significance_markers(ax, x, low_means, high_means, p_values, width)

    ax.set_xlabel('Emotion')
    ax.set_ylabel('Mean Score')
    ax.set_title(f'{dimension}-Involvement')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(low_means), max(high_means)) * 1.15)  # Extra space for markers

plt.tight_layout()
plt.savefig(f'{output_dir}/emotion_comparison.png', dpi=300, bbox_inches='tight')
print(f"  저장: emotion_comparison.png")

# Figure 2: Intent comparison across dimensions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, dimension in enumerate(['I', 'You', 'We']):
    ax = axes[idx]

    low_df = df[df[f'{dimension}_level'] == 'Low']
    high_df = df[df[f'{dimension}_level'] == 'High']

    intents = [i.replace('intent_', '') for i in intent_cols]
    low_means = [low_df[i].mean() for i in intent_cols]
    high_means = [high_df[i].mean() for i in intent_cols]

    # Calculate p-values for each intent
    p_values = []
    for intent in intent_cols:
        low_data = df[df[f'{dimension}_level']=='Low'][intent]
        high_data = df[df[f'{dimension}_level']=='High'][intent]
        _, p_value = mannwhitneyu(low_data, high_data, alternative='two-sided')
        p_values.append(p_value)

    x = np.arange(len(intents))
    width = 0.35

    ax.bar(x - width/2, low_means, width, label='Low', alpha=0.8)
    ax.bar(x + width/2, high_means, width, label='High', alpha=0.8)

    # Add significance markers
    add_significance_markers(ax, x, low_means, high_means, p_values, width)

    ax.set_xlabel('Intent')
    ax.set_ylabel('Mean Score')
    ax.set_title(f'{dimension}-Involvement')
    ax.set_xticks(x)
    ax.set_xticklabels(intents, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(low_means), max(high_means)) * 1.15)  # Extra space for markers

plt.tight_layout()
plt.savefig(f'{output_dir}/intent_comparison.png', dpi=300, bbox_inches='tight')
print(f"  저장: intent_comparison.png")

# Figure 3: Heatmap - Emotion patterns (with annotations)
emotion_heatmap_data = []
for dimension in ['I', 'You', 'We']:
    high_df = df[df[f'{dimension}_level'] == 'High']
    emotion_means = [high_df[e].mean() for e in emotion_cols]
    emotion_heatmap_data.append(emotion_means)

emotion_heatmap_df = pd.DataFrame(
    emotion_heatmap_data,
    index=['I-High', 'You-High', 'We-High'],
    columns=[e.replace('emotion_', '') for e in emotion_cols]
)

plt.figure(figsize=(12, 4))
sns.heatmap(emotion_heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd',
            cbar_kws={'label': 'Mean Score'}, linewidths=0.5)
plt.title('Emotion Patterns by Involvement Level (High)', fontsize=14, fontweight='bold')
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Involvement', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/emotion_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  저장: emotion_heatmap.png")

# Figure 4: Heatmap - Intent patterns (with annotations)
intent_heatmap_data = []
for dimension in ['I', 'You', 'We']:
    high_df = df[df[f'{dimension}_level'] == 'High']
    intent_means = [high_df[i].mean() for i in intent_cols]
    intent_heatmap_data.append(intent_means)

intent_heatmap_df = pd.DataFrame(
    intent_heatmap_data,
    index=['I-High', 'You-High', 'We-High'],
    columns=[i.replace('intent_', '') for i in intent_cols]
)

plt.figure(figsize=(12, 4))
sns.heatmap(intent_heatmap_df, annot=True, fmt='.3f', cmap='YlGnBu',
            cbar_kws={'label': 'Mean Score'}, linewidths=0.5)
plt.title('Intent Patterns by Involvement Level (High)', fontsize=14, fontweight='bold')
plt.xlabel('Intent', fontsize=12)
plt.ylabel('Involvement', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/intent_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  저장: intent_heatmap.png")

# Figure 5: Summary comparison - Difference scores
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Emotion differences
ax = axes[0]
emotion_diffs = {
    'I-Involvement': [r['Diff'] for r in i_emotion_results],
    'You-Involvement': [r['Diff'] for r in you_emotion_results],
    'We-Involvement': [r['Diff'] for r in we_emotion_results]
}
emotion_labels = [e.replace('emotion_', '') for e in emotion_cols]

x = np.arange(len(emotion_labels))
width = 0.25

ax.bar(x - width, emotion_diffs['I-Involvement'], width, label='I-Involvement', alpha=0.8)
ax.bar(x, emotion_diffs['You-Involvement'], width, label='You-Involvement', alpha=0.8)
ax.bar(x + width, emotion_diffs['We-Involvement'], width, label='We-Involvement', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Emotion')
ax.set_ylabel('Difference (High - Low)')
ax.set_title('Emotion Differences by Involvement Type', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(emotion_labels, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Intent differences
ax = axes[1]
intent_diffs = {
    'I-Involvement': [r['Diff'] for r in i_intent_results],
    'You-Involvement': [r['Diff'] for r in you_intent_results],
    'We-Involvement': [r['Diff'] for r in we_intent_results]
}
intent_labels = [i.replace('intent_', '') for i in intent_cols]

x = np.arange(len(intent_labels))

ax.bar(x - width, intent_diffs['I-Involvement'], width, label='I-Involvement', alpha=0.8)
ax.bar(x, intent_diffs['You-Involvement'], width, label='You-Involvement', alpha=0.8)
ax.bar(x + width, intent_diffs['We-Involvement'], width, label='We-Involvement', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Intent')
ax.set_ylabel('Difference (High - Low)')
ax.set_title('Intent Differences by Involvement Type', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(intent_labels, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/difference_comparison.png', dpi=300, bbox_inches='tight')
print(f"  저장: difference_comparison.png")

print("\n" + "=" * 80)
print("모든 분석 완료!")
print("=" * 80)
print(f"\n결과 저장 위치: {output_dir}/")
print("  - CSV 파일: i/you/we_*_patterns.csv, dimension_comparison.csv")
print("  - 그래프: emotion_comparison.png, intent_comparison.png")
print("           emotion_heatmap.png, intent_heatmap.png, difference_comparison.png")
print("  - 통계적 유의성: *, **, *** 마커로 표시")
