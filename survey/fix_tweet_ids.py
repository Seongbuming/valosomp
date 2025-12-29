#!/usr/bin/env python3
"""
Fix Tweet IDs in Survey Data
=============================
CSV 파일의 잘못된 Tweet ID를 실제 Tweet ID로 업데이트합니다.
Tweet Text를 기준으로 참조 파일(targeted_tweets_*.csv)에서 올바른 ID를 찾습니다.

사용법:
    python fix_tweet_ids.py -i data/survey_data_ps1_deduped.csv
    python fix_tweet_ids.py -i data/survey_data_ps1_deduped.csv -o data/survey_data_ps1_fixed.csv
    python fix_tweet_ids.py -i data/survey_data_ps1_deduped.csv -r data/targeted_tweets_1.csv data/targeted_tweets_2.csv
"""

import pandas as pd
import argparse
import os
import sys
from pathlib import Path


def load_reference_tweets(reference_files, encoding='ISO-8859-1'):
    """
    참조 파일들을 로드하여 Tweet Text -> Tweet ID 매핑을 생성합니다.

    Args:
        reference_files: 참조 파일 경로 리스트
        encoding: 파일 인코딩

    Returns:
        dict: {Tweet Text: Tweet ID} 매핑
    """
    tweet_mapping = {}

    for ref_file in reference_files:
        if not os.path.exists(ref_file):
            print(f"[경고] 참조 파일을 찾을 수 없습니다: {ref_file}")
            continue

        print(f"  - {ref_file} 로드 중...")
        try:
            df = pd.read_csv(ref_file, encoding=encoding)

            # Tweet Text와 Tweet ID 컬럼 확인
            if 'Tweet Text' not in df.columns or 'Tweet ID' not in df.columns:
                print(f"    [경고] 필수 컬럼(Tweet Text, Tweet ID)이 없습니다. 건너뜀.")
                continue

            # 매핑 추가
            for _, row in df.iterrows():
                tweet_text = row['Tweet Text']
                tweet_id = row['Tweet ID']
                tweet_mapping[tweet_text] = tweet_id

            print(f"    → {len(df)} 트윗 추가됨")

        except Exception as e:
            print(f"    [오류] 파일 읽기 실패: {e}")
            continue

    return tweet_mapping


def fix_tweet_ids(input_file, output_file=None, reference_files=None, encoding='ISO-8859-1',
                  create_backup=True, update_inplace=False):
    """
    CSV 파일의 Tweet ID를 수정합니다.

    Args:
        input_file: 입력 CSV 파일 경로
        output_file: 출력 CSV 파일 경로 (None이면 _fixed 접미사 추가)
        reference_files: 참조 파일 경로 리스트 (None이면 기본값 사용)
        encoding: 파일 인코딩
        create_backup: True면 원본 파일 백업 생성
        update_inplace: True면 Tweet ID 컬럼 직접 업데이트, False면 새 컬럼 추가
    """
    print("=" * 80)
    print("Tweet ID 수정 작업 시작")
    print("=" * 80)

    # 입력 파일 확인
    if not os.path.exists(input_file):
        print(f"[오류] 입력 파일을 찾을 수 없습니다: {input_file}")
        return False

    # 출력 파일 경로 설정
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")

    # 참조 파일 설정
    if reference_files is None:
        data_dir = Path(input_file).parent
        reference_files = [
            str(data_dir / 'targeted_tweets_1.csv'),
            str(data_dir / 'targeted_tweets_2.csv')
        ]

    print(f"\n[1] 설정")
    print(f"  입력 파일: {input_file}")
    print(f"  출력 파일: {output_file}")
    print(f"  참조 파일: {len(reference_files)}개")
    print(f"  직접 업데이트: {'예' if update_inplace else '아니오 (새 컬럼 추가)'}")

    # 데이터 로드
    print(f"\n[2] 입력 파일 로드")
    try:
        survey_df = pd.read_csv(input_file, encoding=encoding)
        print(f"  → {len(survey_df)} 레코드 로드됨")

        # 필수 컬럼 확인
        if 'Tweet Text' not in survey_df.columns or 'Tweet ID' not in survey_df.columns:
            print(f"[오류] 필수 컬럼(Tweet Text, Tweet ID)이 없습니다.")
            return False

    except Exception as e:
        print(f"[오류] 파일 읽기 실패: {e}")
        return False

    # 참조 데이터 로드
    print(f"\n[3] 참조 파일 로드")
    tweet_mapping = load_reference_tweets(reference_files, encoding)

    if not tweet_mapping:
        print(f"[오류] 참조 데이터를 로드할 수 없습니다.")
        return False

    print(f"  → 총 {len(tweet_mapping)} 개의 유니크한 트윗 매핑 생성됨")

    # Tweet ID 수정
    print(f"\n[4] Tweet ID 찾기 및 업데이트")

    def get_correct_tweet_id(tweet_text):
        """Tweet Text로 올바른 Tweet ID 찾기"""
        return tweet_mapping.get(tweet_text, None)

    # 올바른 Tweet ID 찾기
    correct_ids = survey_df['Tweet Text'].apply(get_correct_tweet_id)

    # 매칭 통계
    matched = correct_ids.notna().sum()
    unmatched = correct_ids.isna().sum()

    print(f"\n  매칭 결과:")
    print(f"    - 매칭 성공: {matched} / {len(survey_df)} ({matched/len(survey_df)*100:.1f}%)")
    print(f"    - 매칭 실패: {unmatched} / {len(survey_df)} ({unmatched/len(survey_df)*100:.1f}%)")

    if matched > 0:
        # 정수형으로 변환
        correct_ids_int = correct_ids.astype('Int64')

        # 원본 ID와 비교
        original_ids = survey_df['Tweet ID'].astype('Int64')
        changed = (original_ids != correct_ids_int) & correct_ids_int.notna()
        num_changed = changed.sum()
        num_same = ((original_ids == correct_ids_int) & correct_ids_int.notna()).sum()

        print(f"    - ID 변경됨: {num_changed} ({num_changed/matched*100:.1f}%)")
        print(f"    - ID 동일함: {num_same} ({num_same/matched*100:.1f}%)")

        # 변경 예시 출력
        if num_changed > 0:
            print(f"\n  변경 예시 (처음 3개):")
            changed_samples = survey_df[changed].head(3)
            for idx, row in changed_samples.iterrows():
                original_id = row['Tweet ID']
                new_id = correct_ids_int.loc[idx]
                print(f"    • {original_id} → {new_id}")
                print(f"      텍스트: {row['Tweet Text'][:60]}...")

        # 업데이트 적용
        if update_inplace:
            # 기존 컬럼 직접 업데이트 (매칭된 경우만)
            survey_df.loc[correct_ids.notna(), 'Tweet ID'] = correct_ids_int[correct_ids.notna()]
            print(f"\n  → Tweet ID 컬럼 직접 업데이트됨")
        else:
            # 새 컬럼 추가
            survey_df['Corrected_Tweet_ID'] = correct_ids_int
            print(f"\n  → 'Corrected_Tweet_ID' 컬럼 추가됨")

    # 매칭 실패 케이스 출력
    if unmatched > 0:
        print(f"\n  [경고] 매칭 실패한 케이스 (처음 3개):")
        unmatched_samples = survey_df[correct_ids.isna()].head(3)
        for idx, row in unmatched_samples.iterrows():
            print(f"    • Tweet ID: {row['Tweet ID']}")
            print(f"      텍스트: {row['Tweet Text'][:60]}...")

    # 백업 생성
    if create_backup and os.path.exists(output_file):
        backup_file = output_file + '.backup'
        print(f"\n[5] 백업 생성")
        print(f"  → {backup_file}")
        import shutil
        shutil.copy2(output_file, backup_file)

    # 결과 저장
    print(f"\n[6] 결과 저장")
    try:
        survey_df.to_csv(output_file, index=False, encoding=encoding)
        print(f"  → {output_file}")
        print(f"  → {len(survey_df)} 레코드 저장됨")
    except Exception as e:
        print(f"[오류] 파일 저장 실패: {e}")
        return False

    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='CSV 파일의 잘못된 Tweet ID를 실제 Tweet ID로 업데이트합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (자동으로 targeted_tweets_*.csv 참조)
  python fix_tweet_ids.py -i data/survey_data_ps1_deduped.csv

  # 출력 파일 지정
  python fix_tweet_ids.py -i data/survey_data_ps1_deduped.csv -o data/survey_fixed.csv

  # 참조 파일 직접 지정
  python fix_tweet_ids.py -i data/survey_data_ps1.csv -r data/targeted_tweets_1.csv

  # Tweet ID 컬럼 직접 업데이트 (새 컬럼 추가 대신)
  python fix_tweet_ids.py -i data/survey_data_ps1.csv --update-inplace
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='입력 CSV 파일 경로')
    parser.add_argument('-o', '--output', default=None,
                        help='출력 CSV 파일 경로 (기본값: {input}_fixed.csv)')
    parser.add_argument('-r', '--reference', nargs='+', default=None,
                        help='참조 파일 경로들 (기본값: data/targeted_tweets_*.csv)')
    parser.add_argument('--encoding', default='ISO-8859-1',
                        help='파일 인코딩 (기본값: ISO-8859-1)')
    parser.add_argument('--no-backup', action='store_true',
                        help='백업 파일 생성하지 않음')
    parser.add_argument('--update-inplace', action='store_true',
                        help='새 컬럼 추가 대신 Tweet ID 컬럼 직접 업데이트')

    args = parser.parse_args()

    # 실행
    success = fix_tweet_ids(
        input_file=args.input,
        output_file=args.output,
        reference_files=args.reference,
        encoding=args.encoding,
        create_backup=not args.no_backup,
        update_inplace=args.update_inplace
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
