#!/usr/bin/env python3
"""
Simple test script to verify Gemma-3 setup and evaluate a few sample tweets
"""

import pandas as pd
import json
from dotenv import load_dotenv
from evaluate_tweets_gemma import TweetEvaluator
from config import GEMMA_MODELS

# Load environment variables
load_dotenv()

def test_sample_tweets():
    """Test with a few sample tweets from the dataset"""

    # Load a few sample tweets
    df = pd.read_csv("data/targeted_tweets.csv")
    sample_tweets = df.head(3)  # Test with first 3 tweets

    print("=== Testing Gemma-3 Tweet Evaluation ===\n")

    # Initialize evaluator (using small model for testing)
    print("Initializing Gemma evaluator...")
    print(f"Available models: {list(GEMMA_MODELS.keys())}")
    evaluator = TweetEvaluator(model_name=GEMMA_MODELS["small"])  # Using small model for testing

    # Evaluate each sample tweet
    for idx, row in sample_tweets.iterrows():
        tweet_id = row['Tweet ID']
        tweet_text = row['Tweet Text']

        print(f"\n{'='*60}")
        print(f"Tweet {idx+1}:")
        print(f"ID: {tweet_id}")
        print(f"Text: {tweet_text[:200]}...")
        print(f"{'='*60}")

        try:
            evaluation = evaluator.evaluate_single_tweet(tweet_text)

            # Display results
            for dimension, results in evaluation.items():
                print(f"\n{dimension}:")
                if "scores" in results and isinstance(results["scores"], list):
                    scores = results['scores']
                    print(f"  Scores: {scores}")
                    print(f"  Average: {results.get('average', sum(scores)/len(scores)):.2f}")
                    if "comment" in results:
                        print(f"  Comment: {results['comment']}")
                    if "error" in results:
                        print(f"  ⚠️ Warning: {results['error']}")
                else:
                    print(f"  ⚠️ Error: Could not parse evaluation")
                    if "error" in results:
                        print(f"  Details: {results['error']}")
                    if "raw_response" in results:
                        print(f"  Raw: {results['raw_response'][:100]}...")

            # Save test result
            with open(f"test_result_tweet_{idx+1}.json", 'w') as f:
                json.dump({
                    "tweet_id": tweet_id,
                    "tweet_text": tweet_text,
                    "evaluation": evaluation
                }, f, indent=2)

        except Exception as e:
            print(f"Error evaluating tweet: {str(e)}")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_sample_tweets()