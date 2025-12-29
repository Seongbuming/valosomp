"""
Update Tweet ID in Survey Data
================================
현재 survey data의 Tweet ID가 실제와 맞지 않음.
Tweet Text를 이용해서 targeted_tweets_1.csv에서 진짜 Tweet ID를 찾아서
original_tweet_id 컬럼으로 추가.
"""

import pandas as pd
import os

# 파일 경로
DATA_DIR = '../data'
SURVEY_FILE = os.path.join(DATA_DIR, 'mturk_survey_data_deduped.csv')
TWEETS_FILE = os.path.join(DATA_DIR, 'targeted_tweets_1.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'mturk_survey_data_deduped.csv')

print("="*80)
print("Update Tweet ID in Survey Data")
print("="*80)

# 데이터 로드
print("\n[1] 데이터 로드")
survey_df = pd.read_csv(SURVEY_FILE, encoding='ISO-8859-1')
tweets_df = pd.read_csv(TWEETS_FILE, encoding='ISO-8859-1')

print(f"Survey data: {len(survey_df)} rows")
print(f"Targeted tweets: {len(tweets_df)} rows")

# Tweet Text를 key로 사용할 dictionary 생성
print("\n[2] Tweet Text → Tweet ID 매핑 생성")
tweet_text_to_id = dict(zip(tweets_df['Tweet Text'], tweets_df['Tweet ID']))
print(f"매핑 생성 완료: {len(tweet_text_to_id)} unique tweets")

# Survey data에서 Tweet Text로 original Tweet ID 찾기
print("\n[3] Survey data에 original_tweet_id 추가")

def get_original_tweet_id(tweet_text):
    """Tweet Text로 원본 Tweet ID 찾기"""
    return tweet_text_to_id.get(tweet_text, None)

survey_df['original_tweet_id'] = survey_df['Tweet Text'].apply(get_original_tweet_id)

# 결과 확인
matched = survey_df['original_tweet_id'].notna().sum()
unmatched = survey_df['original_tweet_id'].isna().sum()

print(f"\n매칭 결과:")
print(f"  - 매칭 성공: {matched} / {len(survey_df)} ({matched/len(survey_df)*100:.1f}%)")
print(f"  - 매칭 실패: {unmatched} / {len(survey_df)} ({unmatched/len(survey_df)*100:.1f}%)")

# ID 불일치 확인
if matched > 0:
    survey_df['original_tweet_id'] = survey_df['original_tweet_id'].astype('Int64')
    mismatch = (survey_df['Tweet ID'] != survey_df['original_tweet_id']).sum()
    print(f"  - ID 불일치: {mismatch} / {matched} ({mismatch/matched*100:.1f}%)")

    # if mismatch > 0:
    #     print(f"\n[예시] ID 불일치 사례 (처음 5개):")
    #     mismatched_samples = survey_df[survey_df['Tweet ID'] != survey_df['original_tweet_id']].head(5)
    #     for idx, row in mismatched_samples.iterrows():
    #         print(f"  Survey ID: {row['Tweet ID']}")
    #         print(f"  Original ID: {row['original_tweet_id']}")
    #         print(f"  Text: {row['Tweet Text'][:80]}...")
    #         print()

# 매칭 실패한 케이스 확인
# if unmatched > 0:
#     print(f"\n[경고] 매칭 실패한 케이스 (처음 5개):")
#     unmatched_samples = survey_df[survey_df['original_tweet_id'].isna()].head(5)
#     for idx, row in unmatched_samples.iterrows():
#         print(f"  Tweet ID: {row['Tweet ID']}")
#         print(f"  Text: {row['Tweet Text'][:80]}...")
#         print()

# 결과 저장
print(f"\n[4] 결과 저장")
survey_df.to_csv(OUTPUT_FILE, index=False, encoding='ISO-8859-1')
print(f"저장 완료: {OUTPUT_FILE}")

# 컬럼 순서 확인
print(f"\n[5] 컬럼 순서 (original_tweet_id 위치 확인)")
cols = survey_df.columns.tolist()
if 'original_tweet_id' in cols:
    pos = cols.index('original_tweet_id')
    print(f"original_tweet_id 위치: {pos+1}/{len(cols)}")
    print(f"주변 컬럼: ... {cols[max(0,pos-2):min(len(cols),pos+3)]} ...")

print("\n" + "="*80)
print("완료!")
print("="*80)
