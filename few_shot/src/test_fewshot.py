#!/usr/bin/env python3
"""
Quick test script for few-shot evaluation system.
Tests on a small subset to verify functionality.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_tweets_gemma_fewshot import FewShotTweetEvaluator
import pandas as pd


def test_example_selection():
    """Test the similarity-based example selection"""
    print("="*60)
    print("Testing Example Selection")
    print("="*60)

    evaluator = FewShotTweetEvaluator(k_shots=4)

    # Test tweet
    test_tweet = "Praying for all the victims of the Boston bombing. Stay strong!"

    print(f"\nTest Tweet: {test_tweet}")
    print("\nSelecting 4 most similar examples from MTurk data...")

    example_indices = evaluator.select_fewshot_examples(test_tweet, k=4)

    print(f"\nSelected example indices: {example_indices}")
    print("\nSelected examples:")
    print("-"*60)

    for i, idx in enumerate(example_indices, 1):
        row = evaluator.mturk_df.iloc[idx]
        print(f"\nExample {i}:")
        print(f"  Tweet: {row['Tweet Text'][:100]}...")
        print(f"  I-Involvement: {row['I-Involvement-1']}, {row['I-Involvement-2']}, {row['I-Involvement-3']}")
        print(f"  Comment: {row['I-Involvement Comment'][:100]}...")

    print("\n" + "="*60)


def test_prompt_creation():
    """Test few-shot prompt creation"""
    print("="*60)
    print("Testing Few-Shot Prompt Creation")
    print("="*60)

    evaluator = FewShotTweetEvaluator(k_shots=2)

    test_tweet = "My thoughts and prayers go out to Boston. This is terrible."

    # Select examples
    example_indices = evaluator.select_fewshot_examples(test_tweet, k=2)

    # Create prompt for I-Involvement
    dimension_info = evaluator.evaluation_dimensions["I-Involvement"]
    prompt = evaluator.create_fewshot_prompt(
        test_tweet,
        dimension_info["title"],
        dimension_info["questions"],
        example_indices
    )

    print("\nGenerated Prompt:")
    print("-"*60)
    print(prompt[:1000])  # Print first 1000 chars
    print("...")
    print("-"*60)


def test_single_evaluation():
    """Test evaluation of a single tweet"""
    print("="*60)
    print("Testing Single Tweet Evaluation (k=2)")
    print("="*60)

    evaluator = FewShotTweetEvaluator(k_shots=2)

    test_tweet = "Sending prayers to everyone affected by the Colorado wildfires."

    print(f"\nEvaluating: {test_tweet}")
    print("\nThis will use the actual Gemma model and may take a minute...")

    try:
        results = evaluator.evaluate_single_tweet(test_tweet)

        print("\nEvaluation Results:")
        print("-"*60)

        for dimension, result in results.items():
            print(f"\n{dimension}:")
            print(f"  Scores: {result['scores']}")
            print(f"  Average: {result['average']:.2f}")
            print(f"  Comment: {result['comment']}")
            if 'example_indices' in result:
                print(f"  Used examples: {result['example_indices']}")

        print("\n" + "="*60)
        print("✅ Single tweet evaluation successful!")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


def test_small_dataset():
    """Test evaluation on a very small dataset"""
    print("="*60)
    print("Testing Small Dataset Evaluation (k=1)")
    print("="*60)

    # Create a small test dataset
    test_data = pd.DataFrame([
        {
            "Tweet ID": "test_001",
            "Tweet Text": "Praying for Boston. Stay strong!",
            "source_dataset": "test"
        },
        {
            "Tweet ID": "test_002",
            "Tweet Text": "My heart goes out to all the families affected by this tragedy.",
            "source_dataset": "test"
        }
    ])

    # Save to temporary file
    test_csv = "few_shot/test_tweets.csv"
    test_data.to_csv(test_csv, index=False)

    print(f"\nCreated test dataset with {len(test_data)} tweets")
    print(f"Saved to: {test_csv}")

    evaluator = FewShotTweetEvaluator(k_shots=1)

    print("\nRunning evaluation...")
    try:
        results = evaluator.evaluate_dataset(
            test_csv,
            output_path="few_shot/test_results.json",
            max_retries=1
        )

        print("\n" + "="*60)
        print("✅ Small dataset evaluation successful!")
        print(f"Processed {len(results)} tweets")
        print("Output saved to: few_shot/test_results.json")
        print("                 few_shot/test_results.csv")

    except Exception as e:
        print(f"\n❌ Error during dataset evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Test few-shot evaluation system")
    parser.add_argument(
        "--test",
        choices=["selection", "prompt", "single", "dataset", "all"],
        default="all",
        help="Which test to run"
    )

    args = parser.parse_args()

    tests = {
        "selection": test_example_selection,
        "prompt": test_prompt_creation,
        "single": test_single_evaluation,
        "dataset": test_small_dataset
    }

    if args.test == "all":
        # Run tests in order (skip heavy model tests by default)
        print("\nRunning lightweight tests (selection and prompt)...")
        print("For model-based tests, run with --test single or --test dataset\n")

        test_example_selection()
        print("\n")
        test_prompt_creation()

        print("\n" + "="*60)
        print("Lightweight tests completed!")
        print("To test actual model evaluation:")
        print("  python few_shot/test_fewshot.py --test single")
        print("  python few_shot/test_fewshot.py --test dataset")
        print("="*60)
    else:
        tests[args.test]()


if __name__ == "__main__":
    main()
