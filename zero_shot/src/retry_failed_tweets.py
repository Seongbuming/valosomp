#!/usr/bin/env python3
"""
Retry evaluation for failed tweets from a previous run
"""

import json
import sys
from evaluate_tweets_gemma import TweetEvaluator
from datetime import datetime
from pathlib import Path
import pandas as pd

def retry_failed_evaluations(json_file_path):
    """Load a previous evaluation and retry only the failed ones"""

    # Load previous evaluations
    with open(Path("zero_shot") / "data" / "evaluations" / json_file_path, 'r') as f:
        evaluations = json.load(f)

    # Find failed tweets
    failed_tweets = [e for e in evaluations if e.get("has_error", False)]
    successful_tweets = [e for e in evaluations if not e.get("has_error", False)]

    print(f"Loaded {len(evaluations)} evaluations")
    print(f"✅ Successful: {len(successful_tweets)}")
    print(f"❌ Failed: {len(failed_tweets)}")

    if not failed_tweets:
        print("No failed tweets to retry!")
        return evaluations

    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = TweetEvaluator()

    # Retry failed tweets
    print(f"\nRetrying {len(failed_tweets)} failed tweets...")
    retry_count = 0

    for eval_item in failed_tweets:
        tweet_id = eval_item["tweet_id"]
        tweet_text = eval_item["tweet_text"]

        print(f"\nRetrying Tweet ID {tweet_id}: {tweet_text[:100]}...")

        try:
            evaluation = evaluator.evaluate_single_tweet(tweet_text)

            # Check if successful
            has_error = any("error" in dim_eval for dim_eval in evaluation.values())

            if not has_error:
                # Update the evaluation
                eval_item["evaluation"] = evaluation
                eval_item["has_error"] = False
                eval_item["retry_timestamp"] = datetime.now().isoformat()
                eval_item.pop("error", None)  # Remove error field if exists
                retry_count += 1
                print(f"  ✅ Successfully evaluated")
            else:
                print(f"  ⚠️ Still has errors")

        except Exception as e:
            print(f"  ❌ Retry failed: {str(e)}")

    # Re-sort: successful first, failed last
    updated_successful = [e for e in evaluations if not e.get("has_error", False)]
    still_failed = [e for e in evaluations if e.get("has_error", False)]

    final_evaluations = updated_successful + still_failed

    # Save updated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"zero_shot/data/evaluations/gemma_k0_retry_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(final_evaluations, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Retry complete!")
    print(f"✅ Total successful: {len(updated_successful)} (+{retry_count})")
    print(f"❌ Still failed: {len(still_failed)}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Also save as CSV
    # from zero_shot.src.evaluate_tweets_gemma import TweetEvaluator
    temp_evaluator = TweetEvaluator.__new__(TweetEvaluator)
    csv_output = output_path.replace('.json', '.csv')
    temp_evaluator.save_as_csv = TweetEvaluator.save_as_csv.__get__(temp_evaluator)
    temp_evaluator.save_as_csv(final_evaluations, csv_output)

    return final_evaluations

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retry_failed_tweets.py <json_file_path>")
        print("Example: python retry_failed_tweets.py gemma_k0_20250101_120000.json")
        sys.exit(1)

    json_file = sys.argv[1]
    retry_failed_evaluations(json_file)