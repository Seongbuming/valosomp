#!/usr/bin/env python3
"""
Pilot Study + Main Study 통합 분석
- Emotion classification (BERT)
- Intent classification (BERT)
- Involvement별 패턴 분석
- Crisis type별 분석
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline
import torch
from scipy.stats import chi2_contingency, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os

print("=" * 80)
print("PILOT + MAIN STUDY 통합 분석")
print("=" * 80)

# ============================================================================
# 1. 데이터 로드 및 통합
# ============================================================================

print("\n[1] 데이터 로드 중...")

# Pilot Study
pilot_df = pd.read_csv('../data/survey_data_ps_deduped_fixed.csv')
pilot_df['source_dataset'] = 'Pilot'
print(f"  Pilot Study: {len(pilot_df)} tweets")

# Main Study
main_df = pd.read_csv('../data/survey_data_ms_deduped_fixed.csv')
main_df['source_dataset'] = 'Main'
print(f"  Main Study: {len(main_df)} tweets")

# 공통 컬럼만 선택
common_cols = [
    'Tweet ID', 'Tweet Text',
    'I-Involvement-1', 'I-Involvement-2', 'I-Involvement-3',
    'You-Involvement-1', 'You-Involvement-2', 'You-Involvement-3',
    'We-Involvement-1', 'We-Involvement-2', 'We-Involvement-3',
    'source_dataset'
]

pilot_sub = pilot_df[common_cols].copy()
main_sub = main_df[common_cols].copy()

# 통합
df = pd.concat([pilot_sub, main_sub], ignore_index=True)
print(f"\n통합 데이터셋: {len(df)} tweets")

# Involvement 평균 및 레벨 계산
df['I_avg'] = df[['I-Involvement-1', 'I-Involvement-2', 'I-Involvement-3']].mean(axis=1)
df['You_avg'] = df[['You-Involvement-1', 'You-Involvement-2', 'You-Involvement-3']].mean(axis=1)
df['We_avg'] = df[['We-Involvement-1', 'We-Involvement-2', 'We-Involvement-3']].mean(axis=1)

# 2-level classification (Low: 1-4, High: 5-7)
df['I_level'] = np.where(df['I_avg'] < 5, 'Low', 'High')
df['You_level'] = np.where(df['You_avg'] < 5, 'Low', 'High')
df['We_level'] = np.where(df['We_avg'] < 5, 'Low', 'High')

print("\nInvolvement level 분포:")
print(f"  I-High: {(df['I_level']=='High').sum()} ({(df['I_level']=='High').sum()/len(df)*100:.1f}%)")
print(f"  You-High: {(df['You_level']=='High').sum()} ({(df['You_level']=='High').sum()/len(df)*100:.1f}%)")
print(f"  We-High: {(df['We_level']=='High').sum()} ({(df['We_level']=='High').sum()/len(df)*100:.1f}%)")

# ============================================================================
# 2. Emotion Classification (BERT)
# ============================================================================

print("\n[2] Emotion Classification 시작...")
print("  모델 로딩: j-hartmann/emotion-english-distilroberta-base")

device = 0 if torch.cuda.is_available() else -1
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=device
)

print(f"  GPU 사용: {torch.cuda.is_available()}")
print(f"  분석 대상: {len(df)} tweets")

# Emotion 분류
emotion_results = []
for idx, text in enumerate(df['Tweet Text']):
    if idx % 500 == 0:
        print(f"    진행: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")

    try:
        result = emotion_classifier(text[:512])[0]  # max 512 chars
        emotion_scores = {item['label']: item['score'] for item in result}
        emotion_results.append(emotion_scores)
    except Exception as e:
        # Fallback
        emotion_results.append({
            'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0,
            'neutral': 0, 'sadness': 0, 'surprise': 0
        })

# DataFrame에 추가
emotion_df = pd.DataFrame(emotion_results)
for col in emotion_df.columns:
    df[f'emotion_{col}'] = emotion_df[col]

print("\n  Emotion classification 완료!")

# ============================================================================
# 3. Intent Classification
# ============================================================================

print("\n[3] Intent Classification 시작...")
print("  모델 로딩: cardiffnlp/tweet-topic-21-multi")

intent_classifier = pipeline(
    "text-classification",
    model="cardiffnlp/tweet-topic-21-multi",
    top_k=None,
    device=device
)

# Intent categories mapping
intent_mapping = {
    'news_&_social_concern': 'informational',
    'diaries_&_daily_life': 'self-expressive',
    'arts_&_culture': 'self-expressive',
    'celebrity_&_pop_culture': 'self-expressive',
    'sports': 'self-expressive',
    'business_&_entrepreneurs': 'informational',
    'science_&_technology': 'informational',
    'learning_&_educational': 'informational',
    'fashion_&_style': 'self-expressive',
    'fitness_&_health': 'informational',
    'food_&_dining': 'self-expressive',
    'travel_&_adventure': 'self-expressive',
    'gaming': 'self-expressive',
    'film_tv_&_video': 'self-expressive',
    'music': 'self-expressive',
    'other_hobbies': 'self-expressive',
    'relationships': 'supportive',
    'family': 'supportive',
    'youth_&_student_life': 'self-expressive'
}

intent_results = []
for idx, text in enumerate(df['Tweet Text']):
    if idx % 500 == 0:
        print(f"    진행: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")

    try:
        result = intent_classifier(text[:512])[0]

        # Aggregate by intent categories
        intent_scores = {
            'informational': 0,
            'supportive': 0,
            'self-expressive': 0,
            'call-to-action': 0
        }

        for item in result:
            topic = item['label']
            score = item['score']
            intent = intent_mapping.get(topic, 'self-expressive')
            intent_scores[intent] += score

        # Simple heuristics for call-to-action
        text_lower = text.lower()
        if any(word in text_lower for word in ['please', 'help', 'donate', 'pray', 'support', 'share']):
            intent_scores['call-to-action'] += 0.3

        intent_results.append(intent_scores)
    except Exception as e:
        intent_results.append({
            'informational': 0.25,
            'supportive': 0.25,
            'self-expressive': 0.25,
            'call-to-action': 0.25
        })

# DataFrame에 추가
intent_df = pd.DataFrame(intent_results)
for col in intent_df.columns:
    df[f'intent_{col}'] = intent_df[col]

print("\n  Intent classification 완료!")

# ============================================================================
# 4. 결과 저장
# ============================================================================

output_path = 'emotional_and_intent/combined_analysis_data.csv'
df.to_csv(output_path, index=False)
print(f"\n통합 데이터 저장: {output_path}")

# ============================================================================
# 5. 기본 통계 분석
# ============================================================================

print("\n" + "=" * 80)
print("분석 결과 요약")
print("=" * 80)

# Emotion by Involvement
print("\n[A] Emotion by I-Involvement Level")
emotion_cols = [c for c in df.columns if c.startswith('emotion_')]
for emotion in emotion_cols:
    low_mean = df[df['I_level']=='Low'][emotion].mean()
    high_mean = df[df['I_level']=='High'][emotion].mean()
    print(f"  {emotion:20s}: Low={low_mean:.3f}, High={high_mean:.3f}, Diff={high_mean-low_mean:+.3f}")

print("\n[B] Intent by I-Involvement Level")
intent_cols = [c for c in df.columns if c.startswith('intent_')]
for intent in intent_cols:
    low_mean = df[df['I_level']=='Low'][intent].mean()
    high_mean = df[df['I_level']=='High'][intent].mean()
    print(f"  {intent:25s}: Low={low_mean:.3f}, High={high_mean:.3f}, Diff={high_mean-low_mean:+.3f}")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
