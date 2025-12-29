#!/usr/bin/env python3
"""
Add timestamps to combined_analysis_data_with_crisis.csv
Extract from CrisisLexT26/{event}/{event}-tweetids_entire_period.csv
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("Adding Timestamps to Combined Dataset")
print("=" * 80)

# ============================================================================
# 1. Extract timestamps from all CrisisLexT26 events
# ============================================================================

print("\n[1] Extracting timestamps from CrisisLexT26...")

crisis_lex_dir = Path('../data/CrisisLexT26')
all_timestamps = []

for crisis_dir in crisis_lex_dir.iterdir():
    if crisis_dir.is_dir():
        event_name = crisis_dir.name
        csv_file = crisis_dir / f"{event_name}-tweetids_entire_period.csv"

        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                # Column names may have spaces
                df.columns = df.columns.str.strip()

                # Rename for consistency
                if 'Tweet-ID' in df.columns:
                    df = df.rename(columns={'Tweet-ID': 'Tweet ID'})

                # Keep only relevant columns
                df = df[['Timestamp', 'Tweet ID']].copy()
                df['source_dataset'] = event_name

                all_timestamps.append(df)
                print(f"  {event_name:40s}: {len(df):,} tweets")
            except Exception as e:
                print(f"  {event_name:40s}: ERROR - {e}")
        else:
            print(f"  {event_name:40s}: No timestamp file")

print(f"\nTotal events with timestamps: {len(all_timestamps)}")

# Combine all timestamps
if all_timestamps:
    timestamps_df = pd.concat(all_timestamps, ignore_index=True)
    print(f"Total tweet-timestamp mappings: {len(timestamps_df):,}")

    # Parse timestamps
    print("\n[2] Parsing timestamps...")
    timestamps_df['datetime'] = pd.to_datetime(timestamps_df['Timestamp'], format='%a %b %d %H:%M:%S %z %Y')

    # Convert Tweet ID to string for matching
    timestamps_df['Tweet ID'] = timestamps_df['Tweet ID'].astype(str)

    print(f"  Timestamp range: {timestamps_df['datetime'].min()} to {timestamps_df['datetime'].max()}")

    # ========================================================================
    # 3. Merge with combined data
    # ========================================================================

    print("\n[3] Merging with combined analysis data...")

    # Load combined data
    combined = pd.read_csv('emotional_and_intent/combined_analysis_data_with_crisis.csv')
    print(f"  Combined data: {len(combined):,} tweets")

    # Convert Tweet ID to string for matching
    combined['Tweet ID'] = combined['Tweet ID'].astype(str)

    # Merge on Tweet ID
    merged = combined.merge(
        timestamps_df[['Tweet ID', 'datetime']],
        on='Tweet ID',
        how='left'
    )

    print(f"  Merged data: {len(merged):,} tweets")

    # Check how many have timestamps
    with_timestamp = merged[merged['datetime'].notna()]
    print(f"  Tweets with timestamp: {len(with_timestamp):,} ({len(with_timestamp)/len(merged)*100:.1f}%)")

    # Extract date components for analysis
    merged['year'] = merged['datetime'].dt.year
    merged['month'] = merged['datetime'].dt.month
    merged['day'] = merged['datetime'].dt.day
    merged['hour'] = merged['datetime'].dt.hour
    merged['dayofweek'] = merged['datetime'].dt.dayofweek  # Monday=0, Sunday=6

    # Calculate time since crisis start (for each event)
    print("\n[4] Calculating time since crisis start...")

    merged['time_since_start'] = None

    for event in merged['source_dataset_crisis'].dropna().unique():
        event_tweets = merged[merged['source_dataset_crisis'] == event]
        if len(event_tweets) > 0:
            min_time = event_tweets['datetime'].min()
            merged.loc[merged['source_dataset_crisis'] == event, 'time_since_start'] = (
                (merged.loc[merged['source_dataset_crisis'] == event, 'datetime'] - min_time).dt.total_seconds() / 3600  # hours
            )

    # ========================================================================
    # 5. Save updated data
    # ========================================================================

    output_path = 'emotional_and_intent/combined_analysis_data_with_timestamps.csv'
    merged.to_csv(output_path, index=False)

    print(f"\n[5] Saved updated data to: {output_path}")

    # ========================================================================
    # 6. Summary Statistics
    # ========================================================================

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    print("\n[A] Temporal Coverage:")
    print(f"  Overall range: {merged['datetime'].min()} to {merged['datetime'].max()}")
    print(f"  Span: {(merged['datetime'].max() - merged['datetime'].min()).days} days")

    print("\n[B] Tweets by Year:")
    year_counts = merged['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        if pd.notna(year):
            print(f"  {int(year)}: {count:,} tweets")

    print("\n[C] Tweets by Month (2013):")
    tweets_2013 = merged[merged['year'] == 2013]
    month_counts = tweets_2013['month'].value_counts().sort_index()
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, count in month_counts.items():
        if pd.notna(month):
            print(f"  {month_names[int(month)]}: {count:,} tweets")

    print("\n[D] Tweets by Day of Week:")
    dow_counts = merged['dayofweek'].value_counts().sort_index()
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for dow, count in dow_counts.items():
        if pd.notna(dow):
            print(f"  {dow_names[int(dow)]}: {count:,} tweets")

    print("\n[E] Time Since Crisis Start (hours):")
    tsc = merged['time_since_start'].dropna()
    print(f"  Mean: {tsc.mean():.1f} hours")
    print(f"  Median: {tsc.median():.1f} hours")
    print(f"  Range: {tsc.min():.1f} - {tsc.max():.1f} hours ({tsc.max()/24:.1f} days)")

    print("\n" + "=" * 80)
    print("Timestamp Addition Complete!")
    print("=" * 80)
    print(f"\nOutput: {output_path}")
    print(f"Tweets with timestamps: {len(with_timestamp):,} / {len(merged):,} ({len(with_timestamp)/len(merged)*100:.1f}%)")

else:
    print("\nNo timestamp files found!")
