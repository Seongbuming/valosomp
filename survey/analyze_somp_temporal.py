#!/usr/bin/env python3
"""
시간별 SOMP 비율 분석 및 시각화

이 스크립트는 다음을 수행합니다:
1. 전체 트윗에 대해 시간별 SOMP 비율을 stacked bar chart로 시각화
2. 각 사건(source_dataset)별로 시간별 SOMP 비율을 stacked bar chart로 시각화

사용법:
    python analyze_somp_temporal.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import numpy as np

# 파일 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "mturk_survey_with_timestamps.csv"
OUTPUT_DIR = BASE_DIR / "survey" / "figures"

# 출력 디렉토리 생성
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """데이터 로드 및 전처리"""
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} records")

    # Tweet Timestamp를 datetime으로 변환
    df['datetime'] = pd.to_datetime(df['Tweet Timestamp'], format='%a %b %d %H:%M:%S %z %Y')

    # 각 involvement의 평균 계산 (3개 측정치의 평균)
    df['I-Involvement'] = df[['I-Involvement-1', 'I-Involvement-2', 'I-Involvement-3']].mean(axis=1)
    df['You-Involvement'] = df[['You-Involvement-1', 'You-Involvement-2', 'You-Involvement-3']].mean(axis=1)
    df['We-Involvement'] = df[['We-Involvement-1', 'We-Involvement-2', 'We-Involvement-3']].mean(axis=1)

    # SOMP 합계 (정규화를 위해)
    df['Total-Involvement'] = df['I-Involvement'] + df['You-Involvement'] + df['We-Involvement']

    # 비율 계산 (0-1 사이)
    df['I-Ratio'] = df['I-Involvement'] / df['Total-Involvement']
    df['You-Ratio'] = df['You-Involvement'] / df['Total-Involvement']
    df['We-Ratio'] = df['We-Involvement'] / df['Total-Involvement']

    return df


def create_time_bins(df, freq='6H'):
    """시간 구간 생성"""
    df['time_bin'] = df['datetime'].dt.floor(freq)
    return df


def calculate_somp_ratios(group_df):
    """그룹별 SOMP 비율 계산"""
    return pd.Series({
        'I-Ratio': group_df['I-Ratio'].mean(),
        'You-Ratio': group_df['You-Ratio'].mean(),
        'We-Ratio': group_df['We-Ratio'].mean(),
        'count': len(group_df)
    })


def plot_overall_temporal_somp(df, output_file):
    """전체 데이터의 시간별 SOMP 비율 시각화"""
    print("\nCreating overall temporal SOMP visualization...")

    # 시간 구간별 그룹화
    df_binned = create_time_bins(df.copy(), freq='12H')
    temporal_somp = df_binned.groupby('time_bin').apply(calculate_somp_ratios).reset_index()

    # 시간순 정렬
    temporal_somp = temporal_somp.sort_values('time_bin')

    print(f"  Time bins: {len(temporal_somp)}")
    print(f"  Time range: {temporal_somp['time_bin'].min()} to {temporal_somp['time_bin'].max()}")

    # 시각화
    fig, ax = plt.subplots(figsize=(14, 6))

    # Stacked bar chart
    x = range(len(temporal_somp))
    width = 0.8

    p1 = ax.bar(x, temporal_somp['I-Ratio'], width, label='I-Involvement', color='#E74C3C')
    p2 = ax.bar(x, temporal_somp['You-Ratio'], width, bottom=temporal_somp['I-Ratio'],
                label='You-Involvement', color='#3498DB')
    p3 = ax.bar(x, temporal_somp['We-Ratio'], width,
                bottom=temporal_somp['I-Ratio'] + temporal_somp['You-Ratio'],
                label='We-Involvement', color='#2ECC71')

    # X축 레이블 설정
    ax.set_xticks(x[::max(1, len(x)//10)])  # 최대 10개 레이블만 표시
    ax.set_xticklabels([temporal_somp.iloc[i]['time_bin'].strftime('%Y-%m-%d %H:%M')
                        for i in x[::max(1, len(x)//10)]], rotation=45, ha='right')

    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title('Temporal Distribution of SOMP - All Events',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1)

    # 그리드 추가
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

    return temporal_somp


def plot_event_temporal_somp(df, output_dir):
    """사건별 시간별 SOMP 비율 시각화"""
    print("\nCreating per-event temporal SOMP visualizations...")

    events = df['source_dataset'].unique()
    print(f"  Events: {len(events)}")

    for event in sorted(events):
        print(f"\n  Processing: {event}")
        event_df = df[df['source_dataset'] == event].copy()

        # 시간 구간별 그룹화
        event_df = create_time_bins(event_df, freq='6H')
        temporal_somp = event_df.groupby('time_bin').apply(calculate_somp_ratios).reset_index()

        # 시간순 정렬
        temporal_somp = temporal_somp.sort_values('time_bin')

        print(f"    Time bins: {len(temporal_somp)}")
        print(f"    Total tweets: {len(event_df)}")

        # 시각화
        fig, ax = plt.subplots(figsize=(12, 6))

        # Stacked bar chart
        x = range(len(temporal_somp))
        width = 0.8

        p1 = ax.bar(x, temporal_somp['I-Ratio'], width, label='I-Involvement', color='#E74C3C')
        p2 = ax.bar(x, temporal_somp['You-Ratio'], width, bottom=temporal_somp['I-Ratio'],
                    label='You-Involvement', color='#3498DB')
        p3 = ax.bar(x, temporal_somp['We-Ratio'], width,
                    bottom=temporal_somp['I-Ratio'] + temporal_somp['You-Ratio'],
                    label='We-Involvement', color='#2ECC71')

        # X축 레이블 설정
        ax.set_xticks(x)
        ax.set_xticklabels([temporal_somp.iloc[i]['time_bin'].strftime('%m-%d %H:%M')
                            for i in x], rotation=45, ha='right', fontsize=9)

        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_title(f'Temporal SOMP Distribution - {event}',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 1)

        # 그리드 추가
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()

        # 파일명 생성 (안전한 파일명)
        safe_event_name = event.replace('/', '_').replace(' ', '_')
        output_file = output_dir / f"somp_temporal_{safe_event_name}.png"

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    Saved: {output_file}")


def print_summary_statistics(df):
    """요약 통계 출력"""
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    print("\nOverall SOMP Ratios:")
    print(f"  I-Involvement:   {df['I-Ratio'].mean():.3f} ± {df['I-Ratio'].std():.3f}")
    print(f"  You-Involvement: {df['You-Ratio'].mean():.3f} ± {df['You-Ratio'].std():.3f}")
    print(f"  We-Involvement:  {df['We-Ratio'].mean():.3f} ± {df['We-Ratio'].std():.3f}")

    print("\nSOMP Ratios by Event:")
    for event in sorted(df['source_dataset'].unique()):
        event_df = df[df['source_dataset'] == event]
        print(f"\n  {event} (n={len(event_df)}):")
        print(f"    I-Involvement:   {event_df['I-Ratio'].mean():.3f}")
        print(f"    You-Involvement: {event_df['You-Ratio'].mean():.3f}")
        print(f"    We-Involvement:  {event_df['We-Ratio'].mean():.3f}")


def main():
    """메인 함수"""
    print("="*80)
    print("Temporal SOMP Analysis")
    print("="*80)

    # 데이터 로드
    df = load_data()

    # 요약 통계
    print_summary_statistics(df)

    # 전체 시간별 SOMP 시각화
    overall_output = OUTPUT_DIR / "somp_temporal_overall.png"
    plot_overall_temporal_somp(df, overall_output)

    # 사건별 시간별 SOMP 시각화
    plot_event_temporal_somp(df, OUTPUT_DIR)

    print("\n" + "="*80)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
