#!/usr/bin/env python3
"""
중복된 Tweet Text가 있을 경우 마지막 행을 선택하는 스크립트

Usage:
    python deduplicate_tweets.py
"""

import pandas as pd
from pathlib import Path


def deduplicate_survey_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    같은 Tweet Text에 대해 중복 데이터가 있으면 제일 나중 행을 선택

    Args:
        input_path: 입력 CSV 파일 경로
        output_path: 출력 CSV 파일 경로 (None이면 저장하지 않음)

    Returns:
        중복이 제거된 DataFrame
    """
    # CSV 파일 읽기
    df = pd.read_csv(input_path)

    print(f"원본 데이터: {len(df)} 행")

    # Tweet Text 열이 있는지 확인
    if 'Tweet Text' not in df.columns:
        raise ValueError("'Tweet Text' 열을 찾을 수 없습니다.")

    # Tweet Text별 중복 개수 확인
    duplicates = df['Tweet Text'].duplicated(keep=False)
    num_duplicates = duplicates.sum()

    if num_duplicates > 0:
        print(f"중복된 Tweet Text를 가진 행: {num_duplicates} 개")

        # 중복된 Tweet Text 목록 출력
        duplicate_texts = df[duplicates]['Tweet Text'].unique()
        print(f"중복된 Tweet Text 개수: {len(duplicate_texts)} 개")

        for tweet_text in duplicate_texts[:5]:  # 처음 5개만 출력
            count = (df['Tweet Text'] == tweet_text).sum()
            print(f"  - Tweet Text \"{tweet_text}\": {count} 회 등장")

        if len(duplicate_texts) > 5:
            print(f"  ... 외 {len(duplicate_texts) - 5} 개")
    else:
        print("중복된 Tweet Text가 없습니다.")

    # 마지막 행만 유지 (keep='last')
    df_deduped = df.drop_duplicates(subset=['Tweet Text'], keep='last')

    print(f"중복 제거 후: {len(df_deduped)} 행")
    print(f"제거된 행: {len(df) - len(df_deduped)} 개")

    # 출력 파일이 지정되면 저장
    if output_path:
        df_deduped.to_csv(output_path, index=False)
        print(f"\n저장됨: {output_path}")

    return df_deduped


def main():
    # 파일 경로 설정
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "survey_data_ms.csv"
    output_file = project_root / "data" / "survey_data_ms_deduped.csv"

    # 중복 제거 실행
    df_deduped = deduplicate_survey_data(
        input_path=str(input_file),
        output_path=str(output_file)
    )

    print("\n완료!")
    return df_deduped


if __name__ == "__main__":
    main()
