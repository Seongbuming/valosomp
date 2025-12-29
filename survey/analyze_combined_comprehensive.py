#!/usr/bin/env python3
"""
통합 데이터 종합 분석
1. You/We-Involvement 패턴
2. 세 차원 비교 테이블
3. Crisis type별 분석
4. 통계적 유의성 검정
5. 시각화
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
print("통합 데이터 종합 분석")
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
# 1. You-Involvement 패턴
# ============================================================================

print("\n" + "=" * 80)
print("1. You-Involvement 패턴 분석")
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
# 2. We-Involvement 패턴
# ============================================================================

print("\n" + "=" * 80)
print("2. We-Involvement 패턴 분석")
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
# 3. 세 차원 비교 테이블
# ============================================================================

print("\n" + "=" * 80)
print("3. 세 차원 비교: Dominant Emotion & Intent (High level)")
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
# 4. 결과 저장
# ============================================================================

output_dir = 'emotional_and_intent/combined_results'
os.makedirs(output_dir, exist_ok=True)

# CSV 저장
you_emotion_df.to_csv(f'{output_dir}/you_emotion_patterns.csv', index=False)
you_intent_df.to_csv(f'{output_dir}/you_intent_patterns.csv', index=False)
we_emotion_df.to_csv(f'{output_dir}/we_emotion_patterns.csv', index=False)
we_intent_df.to_csv(f'{output_dir}/we_intent_patterns.csv', index=False)
comparison_df.to_csv(f'{output_dir}/dimension_comparison.csv', index=False)

print(f"\n결과 저장 완료: {output_dir}/")

# ============================================================================
# 5. 시각화
# ============================================================================

print("\n" + "=" * 80)
print("5. 시각화 생성")
print("=" * 80)

# Figure 1: Emotion comparison across dimensions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, dimension in enumerate(['I', 'You', 'We']):
    ax = axes[idx]

    low_df = df[df[f'{dimension}_level'] == 'Low']
    high_df = df[df[f'{dimension}_level'] == 'High']

    emotions = [e.replace('emotion_', '') for e in emotion_cols]
    low_means = [low_df[e].mean() for e in emotion_cols]
    high_means = [high_df[e].mean() for e in emotion_cols]

    x = np.arange(len(emotions))
    width = 0.35

    ax.bar(x - width/2, low_means, width, label='Low', alpha=0.8)
    ax.bar(x + width/2, high_means, width, label='High', alpha=0.8)

    ax.set_xlabel('Emotion')
    ax.set_ylabel('Mean Score')
    ax.set_title(f'{dimension}-Involvement')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

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

    x = np.arange(len(intents))
    width = 0.35

    ax.bar(x - width/2, low_means, width, label='Low', alpha=0.8)
    ax.bar(x + width/2, high_means, width, label='High', alpha=0.8)

    ax.set_xlabel('Intent')
    ax.set_ylabel('Mean Score')
    ax.set_title(f'{dimension}-Involvement')
    ax.set_xticks(x)
    ax.set_xticklabels(intents, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/intent_comparison.png', dpi=300, bbox_inches='tight')
print(f"  저장: intent_comparison.png")

# Figure 3: Heatmap - Emotion patterns
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

plt.figure(figsize=(10, 4))
sns.heatmap(emotion_heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Mean Score'})
plt.title('Emotion Patterns by Involvement Level (High)')
plt.xlabel('Emotion')
plt.ylabel('Involvement')
plt.tight_layout()
plt.savefig(f'{output_dir}/emotion_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  저장: emotion_heatmap.png")

# Figure 4: Heatmap - Intent patterns
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

plt.figure(figsize=(10, 4))
sns.heatmap(intent_heatmap_df, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Mean Score'})
plt.title('Intent Patterns by Involvement Level (High)')
plt.xlabel('Intent')
plt.ylabel('Involvement')
plt.tight_layout()
plt.savefig(f'{output_dir}/intent_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  저장: intent_heatmap.png")

print("\n" + "=" * 80)
print("모든 분석 완료!")
print("=" * 80)
print(f"\n결과 저장 위치: {output_dir}/")
print("  - CSV 파일: *_patterns.csv, dimension_comparison.csv")
print("  - 그래프: emotion_comparison.png, intent_comparison.png, *_heatmap.png")
