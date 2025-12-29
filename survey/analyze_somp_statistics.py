#!/usr/bin/env python3
"""
SOMP 통계 분석 - 통계적 유의성 및 신뢰구간 추가

이 스크립트는 다음을 수행합니다:
1. 사건별 SOMP 비율의 통계적 유의성 검정 (ANOVA, Kruskal-Wallis)
2. 신뢰구간을 포함한 시각화
3. 시간에 따른 SOMP 변화의 통계적 검정
4. 상세한 통계 보고서 생성

사용법:
    python analyze_somp_statistics.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime, timedelta

# 파일 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "mturk_survey_with_timestamps.csv"
OUTPUT_DIR = BASE_DIR / "survey" / "figures"
REPORT_FILE = BASE_DIR / "survey" / "somp_statistical_report.txt"

# 출력 디렉토리 생성
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """데이터 로드 및 전처리"""
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} records")

    # Tweet Timestamp를 datetime으로 변환
    df['datetime'] = pd.to_datetime(df['Tweet Timestamp'], format='%a %b %d %H:%M:%S %z %Y')

    # 각 involvement의 평균 계산
    df['I-Involvement'] = df[['I-Involvement-1', 'I-Involvement-2', 'I-Involvement-3']].mean(axis=1)
    df['You-Involvement'] = df[['You-Involvement-1', 'You-Involvement-2', 'You-Involvement-3']].mean(axis=1)
    df['We-Involvement'] = df[['We-Involvement-1', 'We-Involvement-2', 'We-Involvement-3']].mean(axis=1)

    # SOMP 합계
    df['Total-Involvement'] = df['I-Involvement'] + df['You-Involvement'] + df['We-Involvement']

    # 비율 계산
    df['I-Ratio'] = df['I-Involvement'] / df['Total-Involvement']
    df['You-Ratio'] = df['You-Involvement'] / df['Total-Involvement']
    df['We-Ratio'] = df['We-Involvement'] / df['Total-Involvement']

    return df


def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """Bootstrap을 사용한 신뢰구간 계산"""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return lower, upper


def test_event_differences(df, report_file):
    """사건별 SOMP 차이의 통계적 유의성 검정"""
    print("\nTesting statistical significance of SOMP differences across events...")

    report = []
    report.append("=" * 80)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total samples: {len(df)}")
    report.append("\n" + "=" * 80)
    report.append("1. EVENT-LEVEL SOMP DIFFERENCES")
    report.append("=" * 80)

    # 사건별로 데이터 그룹화
    events = sorted(df['source_dataset'].unique())

    for involvement_type in ['I-Ratio', 'You-Ratio', 'We-Ratio']:
        report.append(f"\n{involvement_type.replace('-Ratio', '-Involvement')}:")
        report.append("-" * 40)

        # 사건별 데이터 준비
        groups = [df[df['source_dataset'] == event][involvement_type].values
                  for event in events]

        # ANOVA (정규분포 가정)
        f_stat, p_value_anova = stats.f_oneway(*groups)

        # Kruskal-Wallis (비모수 검정)
        h_stat, p_value_kw = stats.kruskal(*groups)

        report.append(f"  ANOVA: F={f_stat:.4f}, p={p_value_anova:.4f} {'***' if p_value_anova < 0.001 else '**' if p_value_anova < 0.01 else '*' if p_value_anova < 0.05 else 'ns'}")
        report.append(f"  Kruskal-Wallis: H={h_stat:.4f}, p={p_value_kw:.4f} {'***' if p_value_kw < 0.001 else '**' if p_value_kw < 0.01 else '*' if p_value_kw < 0.05 else 'ns'}")

        # 사건별 평균과 신뢰구간
        report.append(f"\n  Event-wise statistics (Mean ± 95% CI):")
        for event in events:
            event_data = df[df['source_dataset'] == event][involvement_type].values
            mean_val = np.mean(event_data)
            ci_lower, ci_upper = bootstrap_ci(event_data)
            report.append(f"    {event:35s}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] (n={len(event_data)})")

    # Pairwise comparisons (post-hoc) - 주요 비교만
    report.append("\n" + "=" * 80)
    report.append("2. PAIRWISE COMPARISONS (Selected pairs)")
    report.append("=" * 80)

    # We-Involvement에서 가장 높은 사건과 낮은 사건 비교
    event_we_means = [(event, df[df['source_dataset'] == event]['We-Ratio'].mean())
                      for event in events]
    event_we_means.sort(key=lambda x: x[1])

    lowest_event = event_we_means[0][0]
    highest_event = event_we_means[-1][0]

    for involvement_type in ['I-Ratio', 'You-Ratio', 'We-Ratio']:
        report.append(f"\n{involvement_type.replace('-Ratio', '-Involvement')}:")
        report.append(f"  Comparing {lowest_event} vs {highest_event}")

        group1 = df[df['source_dataset'] == lowest_event][involvement_type].values
        group2 = df[df['source_dataset'] == highest_event][involvement_type].values

        # T-test
        t_stat, p_value = stats.ttest_ind(group1, group2)

        # Mann-Whitney U (비모수)
        u_stat, p_value_mw = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        report.append(f"    T-test: t={t_stat:.4f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        report.append(f"    Mann-Whitney: U={u_stat:.4f}, p={p_value_mw:.4f} {'***' if p_value_mw < 0.001 else '**' if p_value_mw < 0.01 else '*' if p_value_mw < 0.05 else 'ns'}")

    # 파일로 저장
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"  Statistical report saved: {report_file}")

    return '\n'.join(report)


def test_temporal_trends(df, report_file):
    """시간에 따른 SOMP 변화의 통계적 검정"""
    print("\nTesting temporal trends...")

    report = []
    report.append("\n" + "=" * 80)
    report.append("3. TEMPORAL TREND ANALYSIS")
    report.append("=" * 80)

    # 각 사건별로 시간 경과에 따른 상관관계 분석
    events = sorted(df['source_dataset'].unique())

    for event in events:
        report.append(f"\n{event}:")
        report.append("-" * 40)

        event_df = df[df['source_dataset'] == event].copy()

        # 사건 시작 시점으로부터의 경과 시간 (시간 단위)
        min_time = event_df['datetime'].min()
        event_df['hours_since_start'] = (event_df['datetime'] - min_time).dt.total_seconds() / 3600

        for involvement_type in ['I-Ratio', 'You-Ratio', 'We-Ratio']:
            # Pearson correlation
            corr, p_value = stats.pearsonr(event_df['hours_since_start'],
                                          event_df[involvement_type])

            # Spearman correlation (비모수)
            corr_spearman, p_value_spearman = stats.spearmanr(event_df['hours_since_start'],
                                                              event_df[involvement_type])

            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'

            report.append(f"  {involvement_type.replace('-Ratio', '')}:")
            report.append(f"    Pearson r={corr:7.4f}, p={p_value:.4f} {sig}")
            report.append(f"    Spearman ρ={corr_spearman:7.4f}, p={p_value_spearman:.4f}")

    # 파일에 추가
    with open(report_file, 'a') as f:
        f.write('\n'.join(report))

    return '\n'.join(report)


def plot_somp_with_ci(df, output_file):
    """신뢰구간을 포함한 사건별 SOMP 비교 시각화"""
    print("\nCreating SOMP comparison with confidence intervals...")

    events = sorted(df['source_dataset'].unique())
    involvement_types = ['I-Ratio', 'You-Ratio', 'We-Ratio']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    labels = ['I-Involvement', 'You-Involvement', 'We-Involvement']

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(events))
    width = 0.25

    for i, (inv_type, color, label) in enumerate(zip(involvement_types, colors, labels)):
        means = []
        errors = []

        for event in events:
            event_data = df[df['source_dataset'] == event][inv_type].values
            mean_val = np.mean(event_data)
            ci_lower, ci_upper = bootstrap_ci(event_data)

            means.append(mean_val)
            errors.append([(mean_val - ci_lower), (ci_upper - mean_val)])

        errors = np.array(errors).T

        ax.bar(x + i * width, means, width, label=label, color=color, alpha=0.8,
               yerr=errors, capsize=5, error_kw={'linewidth': 2, 'elinewidth': 1})

    ax.set_xlabel('Event', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion (with 95% CI)', fontsize=12, fontweight='bold')
    ax.set_title('SOMP Distribution Across Events with 95% Confidence Intervals',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([e.replace('_', ' ') for e in events], rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 0.7)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_temporal_with_ci(df, output_file):
    """신뢰구간을 포함한 전체 시간별 SOMP 시각화"""
    print("\nCreating temporal SOMP with confidence intervals...")

    # 시간 구간 생성
    df_copy = df.copy()
    df_copy['time_bin'] = df_copy['datetime'].dt.floor('12h')

    # 시간 구간별 통계
    time_bins = sorted(df_copy['time_bin'].unique())

    involvement_types = ['I-Ratio', 'You-Ratio', 'We-Ratio']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    labels = ['I-Involvement', 'You-Involvement', 'We-Involvement']

    fig, ax = plt.subplots(figsize=(14, 6))

    for inv_type, color, label in zip(involvement_types, colors, labels):
        means = []
        ci_lowers = []
        ci_uppers = []

        for time_bin in time_bins:
            bin_data = df_copy[df_copy['time_bin'] == time_bin][inv_type].values
            if len(bin_data) > 0:
                mean_val = np.mean(bin_data)
                ci_lower, ci_upper = bootstrap_ci(bin_data)
                means.append(mean_val)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
            else:
                means.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)

        x = range(len(time_bins))
        ax.plot(x, means, color=color, linewidth=2, label=label, marker='o', markersize=4)
        ax.fill_between(x, ci_lowers, ci_uppers, color=color, alpha=0.2)

    # X축 레이블 설정
    ax.set_xticks(range(0, len(time_bins), max(1, len(time_bins)//10)))
    ax.set_xticklabels([time_bins[i].strftime('%Y-%m-%d %H:%M')
                        for i in range(0, len(time_bins), max(1, len(time_bins)//10))],
                       rotation=45, ha='right')

    ax.set_ylabel('Proportion', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_title('Temporal Evolution of SOMP with 95% Confidence Intervals',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 0.8)
    ax.grid(axis='both', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def add_effect_sizes(df, report_file):
    """효과 크기(effect size) 계산"""
    print("\nCalculating effect sizes...")

    report = []
    report.append("\n" + "=" * 80)
    report.append("4. EFFECT SIZES")
    report.append("=" * 80)
    report.append("\nCohen's d for pairwise comparisons:")

    events = sorted(df['source_dataset'].unique())

    # We-Ratio에서 극단적인 사건 비교
    event_we_means = [(event, df[df['source_dataset'] == event]['We-Ratio'].mean())
                      for event in events]
    event_we_means.sort(key=lambda x: x[1])

    lowest_event = event_we_means[0][0]
    highest_event = event_we_means[-1][0]

    for involvement_type in ['I-Ratio', 'You-Ratio', 'We-Ratio']:
        group1 = df[df['source_dataset'] == lowest_event][involvement_type].values
        group2 = df[df['source_dataset'] == highest_event][involvement_type].values

        # Cohen's d
        pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
        cohens_d = (np.mean(group2) - np.mean(group1)) / pooled_std

        # 효과 크기 해석
        if abs(cohens_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "small"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        report.append(f"\n{involvement_type.replace('-Ratio', '-Involvement')}:")
        report.append(f"  {lowest_event} vs {highest_event}")
        report.append(f"  Cohen's d = {cohens_d:.4f} ({interpretation})")

    # 파일에 추가
    with open(report_file, 'a') as f:
        f.write('\n'.join(report))

    return '\n'.join(report)


def main():
    """메인 함수"""
    print("=" * 80)
    print("SOMP Statistical Analysis")
    print("=" * 80)

    # 데이터 로드
    df = load_data()

    # 통계적 검정
    test_event_differences(df, REPORT_FILE)
    test_temporal_trends(df, REPORT_FILE)
    add_effect_sizes(df, REPORT_FILE)

    # 신뢰구간을 포함한 시각화
    plot_somp_with_ci(df, OUTPUT_DIR / "somp_comparison_with_ci.png")
    plot_temporal_with_ci(df, OUTPUT_DIR / "somp_temporal_overall_with_ci.png")

    # 보고서 내용 출력
    print("\n" + "=" * 80)
    print("Statistical report saved to:")
    print(f"  {REPORT_FILE}")
    print("\nKey findings:")
    with open(REPORT_FILE, 'r') as f:
        content = f.read()
        # ANOVA 결과만 출력
        for line in content.split('\n'):
            if 'ANOVA' in line or 'Kruskal' in line:
                print(f"  {line}")

    print("\n" + "=" * 80)
    print("All visualizations and reports completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
