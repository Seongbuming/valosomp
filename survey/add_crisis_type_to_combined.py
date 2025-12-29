#!/usr/bin/env python3
"""
Add crisis type information to combined_analysis_data.csv
Using source_dataset from targeted_tweets files and
categorization from CrisisLexT26 event descriptions
"""

import pandas as pd
import json
import os
from pathlib import Path

print("=" * 80)
print("Adding Crisis Type to Combined Dataset")
print("=" * 80)

# ============================================================================
# 1. Extract crisis categorization from CrisisLexT26
# ============================================================================

print("\n[1] Extracting crisis categorization from CrisisLexT26...")

crisis_categories = {}
crisis_lex_dir = Path('../data/CrisisLexT26')

for crisis_dir in crisis_lex_dir.iterdir():
    if crisis_dir.is_dir():
        event_name = crisis_dir.name
        json_file = crisis_dir / f"{event_name}-event_description.json"

        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                category = data.get('categorization', {}).get('category', 'Unknown')
                crisis_categories[event_name] = category
                print(f"  {event_name:40s}: {category}")
        else:
            print(f"  {event_name:40s}: JSON file not found")

print(f"\nTotal crisis events: {len(crisis_categories)}")

# ============================================================================
# 2. Load targeted tweets and map Tweet ID to crisis category
# ============================================================================

print("\n[2] Loading targeted tweets files...")

# Load all targeted tweets files
ps1 = pd.read_csv('../data/targeted_tweets_ps1.csv')
ps2 = pd.read_csv('../data/targeted_tweets_ps2.csv')
ms = pd.read_csv('../data/targeted_tweets_ms.csv')

print(f"  Pilot Study 1: {len(ps1)} tweets")
print(f"  Pilot Study 2: {len(ps2)} tweets")
print(f"  Main Study: {len(ms)} tweets")

# Combine all targeted tweets
all_targeted = pd.concat([ps1, ps2, ms], ignore_index=True)
print(f"  Total targeted tweets: {len(all_targeted)}")

# Add crisis category
all_targeted['crisis_category'] = all_targeted['source_dataset'].map(crisis_categories)

# Check unmapped
unmapped = all_targeted[all_targeted['crisis_category'].isna()]
if len(unmapped) > 0:
    print(f"\n  WARNING: {len(unmapped)} tweets have unmapped crisis categories:")
    print(unmapped['source_dataset'].value_counts())
else:
    print("  All crisis categories mapped successfully!")

# Create Tweet ID -> crisis mapping
tweet_crisis_map = all_targeted[['Tweet ID', 'source_dataset', 'crisis_category']].copy()
tweet_crisis_map = tweet_crisis_map.drop_duplicates(subset='Tweet ID')

print(f"\n  Unique tweets with crisis info: {len(tweet_crisis_map)}")
print(f"\n  Crisis category distribution:")
print(tweet_crisis_map['crisis_category'].value_counts())

# ============================================================================
# 3. Merge with combined_analysis_data.csv
# ============================================================================

print("\n[3] Merging with combined analysis data...")

# Load combined data
combined = pd.read_csv('emotional_and_intent/combined_analysis_data.csv')
print(f"  Combined data: {len(combined)} tweets")

# Merge on Tweet ID (convert to string to avoid precision issues)
combined['Tweet ID'] = combined['Tweet ID'].astype(str)
tweet_crisis_map['Tweet ID'] = tweet_crisis_map['Tweet ID'].astype(str)

merged = combined.merge(
    tweet_crisis_map[['Tweet ID', 'source_dataset', 'crisis_category']],
    on='Tweet ID',
    how='left',
    suffixes=('', '_crisis')
)

print(f"  Merged data: {len(merged)} tweets")

# Check how many tweets have crisis info
tweets_with_crisis = merged[merged['crisis_category'].notna()]
print(f"  Tweets with crisis category: {len(tweets_with_crisis)} ({len(tweets_with_crisis)/len(merged)*100:.1f}%)")

print(f"\n  Crisis category distribution in merged data:")
print(merged['crisis_category'].value_counts(dropna=False))

# ============================================================================
# 4. Save updated data
# ============================================================================

output_path = 'emotional_and_intent/combined_analysis_data_with_crisis.csv'
merged.to_csv(output_path, index=False)

print(f"\n[4] Saved updated data to: {output_path}")

# ============================================================================
# 5. Summary Statistics
# ============================================================================

print("\n" + "=" * 80)
print("Summary Statistics")
print("=" * 80)

# By source dataset (Pilot vs Main)
print("\n[A] Crisis coverage by source dataset:")
for source in ['Pilot', 'Main']:
    source_df = merged[merged['source_dataset_y'] == source]
    with_crisis = source_df[source_df['crisis_category'].notna()]
    print(f"  {source:10s}: {len(with_crisis):4d} / {len(source_df):4d} ({len(with_crisis)/len(source_df)*100:.1f}%) have crisis info")

# By crisis event
print("\n[B] Top crisis events:")
crisis_events = merged[merged['source_dataset_x'].notna()]['source_dataset_x'].value_counts().head(10)
for event, count in crisis_events.items():
    category = crisis_categories.get(event, 'Unknown')
    print(f"  {event:40s}: {count:4d} tweets ({category})")

# By Involvement level
print("\n[C] Crisis category by Involvement level:")
for dimension in ['I', 'You', 'We']:
    print(f"\n  {dimension}-Involvement:")
    crosstab = pd.crosstab(
        merged[f'{dimension}_level'],
        merged['crisis_category'],
        dropna=False
    )
    print(crosstab)

print("\n" + "=" * 80)
print("Crisis Type Addition Complete!")
print("=" * 80)
print(f"\nOutput: {output_path}")
