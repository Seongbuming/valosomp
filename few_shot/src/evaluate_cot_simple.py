#!/usr/bin/env python3
"""
Chain-of-Thought Few-shot evaluation with high-quality hand-crafted examples.
Simpler implementation focused on testing CoT effectiveness.
"""

import json
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
from datetime import datetime

load_dotenv()

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
            "The tweet indicates that the issue has consequences for people personally connected to the speaker",
            "The tweet emphasizes relationships or concern for specific individuals"
        ],
        "rating_cols": ["You-Involvement-1", "You-Involvement-2", "You-Involvement-3"]
    },
    "We-Involvement": {
        "title": "We-Involvement Recognition (Collective)",
        "questions": [
            "The tweet expresses concern about the issue's impact on the community or collective",
            "The tweet indicates that the issue has consequences for a group or society",
            "The tweet emphasizes collective experience or shared responsibility"
        ],
        "rating_cols": ["We-Involvement-1", "We-Involvement-2", "We-Involvement-3"]
    }
}


def load_model(model_name="google/gemma-2-9b-it", gpu_id=0):
    """Load Gemma model"""
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN or HUGGINGFACE_TOKEN not found in environment")

    print(f"Loading model: {model_name}")
    print(f"Using GPU: {gpu_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_id}",
        token=hf_token
    )

    print("Model loaded!")
    return model, tokenizer


def format_cot_example(example):
    """Format a single CoT example for the prompt"""
    steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(example['reasoning_steps'])])

    formatted = f"""Example:
Tweet: "{example['tweet']}"

Step-by-step reasoning:
{steps_text}

Based on this analysis:
{{
    "scores": {example['scores']},
    "comment": "{example['comment']}"
}}"""

    return formatted


def create_cot_prompt(tweet_text, dimension_key, dimension_info, cot_examples):
    """Create a CoT-enhanced few-shot prompt"""

    prompt = f"""You are evaluating a tweet about a disaster or crisis situation.
Please evaluate how much you agree with the following statements about this tweet.

Evaluation Dimension: {dimension_info['title']}

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

    for i, question in enumerate(dimension_info['questions'], 1):
        prompt += f"{i}. {question}\n"

    prompt += f"\nHere are {len(cot_examples)} examples showing step-by-step reasoning:\n\n"

    for example in cot_examples:
        prompt += format_cot_example(example) + "\n\n"

    prompt += f"""Now evaluate this tweet using the same step-by-step approach:
Tweet: "{tweet_text}"

Step-by-step reasoning:
1. [Analyze first aspect]
2. [Analyze second aspect]
3. [Analyze third aspect]
4. [Draw conclusion]

Based on this analysis:
{{
    "scores": [score1, score2, score3],
    "comment": "brief explanation"
}}

Provide exactly 3 scores (integers from 1-7) and reasoning:"""

    return prompt


def evaluate_single(model, tokenizer, tweet_text, dimension_key, dimension_info, cot_examples):
    """Evaluate one tweet on one dimension using CoT"""

    prompt = create_cot_prompt(tweet_text, dimension_key, dimension_info, cot_examples)

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Longer for CoT reasoning
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    # Parse JSON - extract FIRST complete JSON object
    try:
        json_start = response.find('{')
        if json_start != -1:
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                scores = result.get('scores', [])
                comment = result.get('comment', '')

                if len(scores) == 3 and all(isinstance(s, int) and 1 <= s <= 7 for s in scores):
                    return scores, comment, response
    except Exception as e:
        pass

    return [4, 4, 4], "Failed to parse", response


def evaluate_dataset(model, tokenizer, test_df, cot_examples_dict, output_path):
    """Evaluate test dataset using CoT few-shot"""

    results = []

    print(f"\nEvaluating {len(test_df)} tweets...")

    for idx in tqdm(range(len(test_df)), desc="Tweets"):
        row = test_df.iloc[idx]
        tweet_text = row['Tweet Text']
        tweet_id = row['Tweet ID']

        result = {
            'tweet_id': int(tweet_id),
            'tweet_text': str(tweet_text),
            'human_scores': {},
            'model_scores': {},
            'model_comments': {},
            'model_reasoning': {}
        }

        for dim_key, dim_info in EVALUATION_DIMENSIONS.items():
            # Get human scores
            human_scores = [
                int(row[dim_info['rating_cols'][0]]),
                int(row[dim_info['rating_cols'][1]]),
                int(row[dim_info['rating_cols'][2]])
            ]

            # Get CoT examples for this dimension
            cot_examples = cot_examples_dict[dim_key]

            # Evaluate with CoT
            model_scores, model_comment, full_response = evaluate_single(
                model, tokenizer, tweet_text, dim_key, dim_info, cot_examples
            )

            result['human_scores'][dim_key] = human_scores
            result['model_scores'][dim_key] = model_scores
            result['model_comments'][dim_key] = model_comment
            result['model_reasoning'][dim_key] = full_response[:500]  # Truncate for storage

        results.append(result)

        # Save checkpoint every 20 tweets
        if (idx + 1) % 20 == 0:
            checkpoint_path = output_path.replace('.json', '_checkpoint.json')
            with open(checkpoint_path, 'w') as f:
                json.dump({'results': results}, f, indent=2)
            print(f"\nCheckpoint saved at tweet {idx+1}")

    # Compute metrics
    metrics = compute_metrics(results)

    # Save final results
    output = {
        'detailed_predictions': results,
        'metrics': metrics,
        'num_examples': len(cot_examples_dict['I-Involvement']),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print_metrics(metrics)

    return metrics


def compute_metrics(results):
    """Compute 2-level accuracy and other metrics"""

    metrics = {}

    for dim_key in EVALUATION_DIMENSIONS.keys():
        human_all = []
        model_all = []

        for r in results:
            human_all.extend(r['human_scores'][dim_key])
            model_all.extend(r['model_scores'][dim_key])

        # MAE
        mae = np.mean(np.abs(np.array(human_all) - np.array(model_all)))

        # Pearson correlation
        corr = np.corrcoef(human_all, model_all)[0, 1]

        # Within-1 accuracy
        within_1 = np.mean(np.abs(np.array(human_all) - np.array(model_all)) <= 1) * 100

        # 2-level accuracy (Low: 1-4, High: 5-7)
        human_levels = ['low' if x <= 4 else 'high' for x in human_all]
        model_levels = ['low' if x <= 4 else 'high' for x in model_all]
        level_acc = np.mean(np.array(human_levels) == np.array(model_levels)) * 100

        metrics[dim_key] = {
            'mae': float(mae),
            'pearson_r': float(corr),
            'within_1_acc': float(within_1),
            'level_acc_2level': float(level_acc)
        }

    # Compute average
    metrics['average'] = {
        'mae': np.mean([m['mae'] for m in metrics.values()]),
        'pearson_r': np.mean([m['pearson_r'] for m in metrics.values()]),
        'within_1_acc': np.mean([m['within_1_acc'] for m in metrics.values()]),
        'level_acc_2level': np.mean([m['level_acc_2level'] for m in metrics.values()])
    }

    return metrics


def print_metrics(metrics):
    """Print metrics in table format"""

    print("\n" + "="*70)
    print("CHAIN-OF-THOUGHT FEW-SHOT RESULTS (2-LEVEL)")
    print("="*70)

    for dim_key in EVALUATION_DIMENSIONS.keys():
        m = metrics[dim_key]
        print(f"\n{dim_key}:")
        print(f"  MAE:           {m['mae']:.3f}")
        print(f"  Pearson r:     {m['pearson_r']:.3f}")
        print(f"  Within-1 Acc:  {m['within_1_acc']:.1f}%")
        print(f"  Level Acc:     {m['level_acc_2level']:.1f}% (2-level)")

    m = metrics['average']
    print(f"\nAVERAGE:")
    print(f"  MAE:           {m['mae']:.3f}")
    print(f"  Pearson r:     {m['pearson_r']:.3f}")
    print(f"  Within-1 Acc:  {m['within_1_acc']:.1f}%")
    print(f"  Level Acc:     {m['level_acc_2level']:.1f}% ⭐ (2-level)")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=str, default="data/survey_data_ps_deduped.csv")
    parser.add_argument("--cot-examples", type=str, default="few_shot/data/cot_examples.json")
    parser.add_argument("--output", type=str, default="few_shot/data/cot_results.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=4, help="Number of CoT examples to use (max 4 per dimension)")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample N tweets for quick testing")

    args = parser.parse_args()

    # Load CoT examples
    print(f"Loading CoT examples from: {args.cot_examples}")
    with open(args.cot_examples, 'r') as f:
        cot_examples_all = json.load(f)

    # Select subset of examples
    cot_examples_dict = {}
    for dim_key in EVALUATION_DIMENSIONS.keys():
        cot_examples_dict[dim_key] = cot_examples_all[dim_key][:args.num_examples]
        print(f"{dim_key}: Using {len(cot_examples_dict[dim_key])} CoT examples")

    # Load model
    model, tokenizer = load_model(gpu_id=args.gpu)

    # Load test data
    print(f"\nLoading test data from: {args.test_data}")
    df = pd.read_csv(args.test_data)

    if args.sample_size:
        print(f"Sampling {args.sample_size} tweets for testing...")
        df = df.sample(n=args.sample_size, random_state=42)

    print(f"Test set: {len(df)} tweets")

    # Evaluate
    evaluate_dataset(model, tokenizer, df, cot_examples_dict, args.output)
