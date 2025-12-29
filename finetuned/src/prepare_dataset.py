#!/usr/bin/env python3
"""
Prepare dataset for fine-tuning Gemma model on tweet evaluation task.
Creates instruction-response pairs in the format needed for training.
"""

import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Evaluation dimensions (same as in the original code)
EVALUATION_DIMENSIONS = {
    "I-Involvement": {
        "title": "I-Involvement Recognition (Individual)",
        "questions": [
            "The tweet expresses concern about the issue's impact on the self",
            "The tweet indicates that the issue has personal consequences for the speaker",
            "The tweet emphasizes individual experience or personal situation regarding the issue"
        ],
        "rating_cols": ["I-Involvement-1", "I-Involvement-2", "I-Involvement-3"]
    },
    "You-Involvement": {
        "title": "You-Involvement Recognition (Relational)",
        "questions": [
            "The tweet expresses concern about the issue's impact on family, friends, or close others",
            "The tweet indicates consequences for someone the speaker cares about",
            "The tweet emphasizes relational ties (support, loyalty, care) in relation to the issue"
        ],
        "rating_cols": ["You-Involvement-1", "You-Involvement-2", "You-Involvement-3"]
    },
    "We-Involvement": {
        "title": "We-Involvement Recognition (Collective)",
        "questions": [
            "The tweet expresses concern about the issue's impact on the community, group, or nation",
            "The tweet indicates collective consequences (e.g., society, city, country, 'us')",
            "The tweet emphasizes shared responsibility, solidarity, or collective well-being regarding the issue"
        ],
        "rating_cols": ["We-Involvement-1", "We-Involvement-2", "We-Involvement-3"]
    }
}

def create_evaluation_prompt(tweet_text, dimension, questions):
    """Create the instruction part (same format as zero-shot)"""
    prompt = f"""You are evaluating a tweet about a disaster or crisis situation.
Please evaluate how much you agree with the following statements about this tweet.

Tweet: "{tweet_text}"

Evaluation Dimension: {dimension}

For each statement below, provide a score from 1 to 7:
- 1 = Strongly Disagree
- 2 = Disagree
- 3 = Somewhat Disagree
- 4 = Neutral
- 5 = Somewhat Agree
- 6 = Agree
- 7 = Strongly Agree

Statements to evaluate:
"""

    for i, question in enumerate(questions, 1):
        prompt += f"{i}. {question}\n"

    prompt += """
Please respond in JSON format with scores and a brief explanation:
{
    "scores": [score1, score2, score3],
    "comment": "brief explanation of why you gave these scores for this dimension"
}

Provide exactly 3 scores (integers from 1-7) for the 3 statements above, and a brief comment (1-2 sentences) explaining your reasoning for this specific dimension.
Analyze the tweet carefully and provide your evaluation:"""

    return prompt

def create_training_example(row, dimension_key, dimension_info):
    """
    Create a single training example (instruction + response)

    Returns:
        dict with 'instruction' and 'response' keys
    """
    tweet_text = row['Tweet Text']

    # Get human scores
    scores = [
        int(row[dimension_info['rating_cols'][0]]),
        int(row[dimension_info['rating_cols'][1]]),
        int(row[dimension_info['rating_cols'][2]])
    ]

    # Get comment if available
    comment_col = f"{dimension_key} Comment"
    comment = row.get(comment_col, "")
    if pd.isna(comment) or comment == "":
        comment = f"The tweet shows {'high' if np.median(scores) >= 5 else 'low'} {dimension_key.lower()}."

    # Create instruction (prompt)
    instruction = create_evaluation_prompt(
        tweet_text,
        dimension_info['title'],
        dimension_info['questions']
    )

    # Create response (ground truth)
    response = json.dumps({
        "scores": scores,
        "comment": comment
    }, indent=2)

    return {
        'instruction': instruction,
        'response': response,
        'tweet_text': tweet_text,
        'dimension': dimension_key,
        'scores': scores
    }

def prepare_dataset(
    input_path="data/survey_data_ps_deduped.csv",
    output_dir="finetuned/data",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
):
    """
    Prepare training dataset from MTurk data

    Args:
        input_path: Path to MTurk CSV
        output_dir: Where to save prepared datasets
        train_ratio: Proportion for training (default: 0.70)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for test (default: 0.15)
        random_seed: Random seed for reproducibility
    """

    print("="*70)
    print("FINE-TUNING DATASET PREPARATION")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Split: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    print()

    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} tweets from MTurk data")

    # Split by tweet (not by dimension)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=random_seed
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_seed
    )

    print(f"Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test tweets")
    print()

    # Create training examples (3 per tweet, one for each dimension)
    def create_examples_from_df(df_subset, split_name):
        examples = []

        for _, row in df_subset.iterrows():
            for dim_key, dim_info in EVALUATION_DIMENSIONS.items():
                try:
                    example = create_training_example(row, dim_key, dim_info)
                    examples.append(example)
                except Exception as e:
                    print(f"Warning: Failed to create example for tweet {row.get('Tweet ID', 'unknown')}, {dim_key}: {e}")
                    continue

        print(f"{split_name:5}: {len(examples):4} examples from {len(df_subset)} tweets")
        return examples

    train_examples = create_examples_from_df(train_df, "Train")
    val_examples = create_examples_from_df(val_df, "Val")
    test_examples = create_examples_from_df(test_df, "Test")

    print()
    print(f"Total training examples: {len(train_examples)}")
    print(f"  (3 dimensions × {len(train_df)} tweets)")
    print()

    # Save datasets
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        'train': train_examples,
        'val': val_examples,
        'test': test_examples
    }

    for split_name, examples in datasets.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")

        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')

        print(f"Saved {len(examples)} examples to {output_path}")

    # Save split info
    split_info = {
        'random_seed': random_seed,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'num_train_tweets': len(train_df),
        'num_val_tweets': len(val_df),
        'num_test_tweets': len(test_df),
        'num_train_examples': len(train_examples),
        'num_val_examples': len(val_examples),
        'num_test_examples': len(test_examples),
        'train_tweet_ids': train_df['Tweet ID'].tolist(),
        'val_tweet_ids': val_df['Tweet ID'].tolist(),
        'test_tweet_ids': test_df['Tweet ID'].tolist()
    }

    split_info_path = os.path.join(output_dir, 'split_info.json')
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved split info to {split_info_path}")

    # Show a sample example
    print()
    print("="*70)
    print("SAMPLE TRAINING EXAMPLE")
    print("="*70)
    sample = train_examples[0]
    print(f"Tweet: {sample['tweet_text'][:100]}...")
    print(f"Dimension: {sample['dimension']}")
    print()
    print("INSTRUCTION:")
    print(sample['instruction'][:500] + "...")
    print()
    print("RESPONSE:")
    print(sample['response'])
    print("="*70)

    return datasets, split_info

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare fine-tuning dataset")
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Training set ratio (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    prepare_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
