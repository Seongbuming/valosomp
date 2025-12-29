#!/usr/bin/env python3
"""
Memory-efficient evaluation of fine-tuned model on test set.
Processes in small batches and clears GPU cache regularly.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import gc

# Evaluation dimensions (same as training)
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
    """Create the evaluation prompt (same format as training)"""
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

def load_model_once(model_path, base_model_name="google/gemma-2-9b-it"):
    """Load model once and return model + tokenizer"""

    print(f"Loading model from: {model_path}")

    hf_token = os.environ.get('HF_TOKEN')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True
    )

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded!")
    return model, tokenizer

def evaluate_single(model, tokenizer, tweet_text, dimension_key, dimension_info):
    """Evaluate one dimension with memory cleanup - returns (scores, comment)"""

    prompt = create_evaluation_prompt(
        tweet_text,
        dimension_info['title'],
        dimension_info['questions']
    )

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
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    # Parse JSON - extract FIRST complete JSON object only
    try:
        json_start = response.find('{')
        if json_start != -1:
            # Find the matching closing brace for the first JSON object
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
                    return scores, comment
    except Exception as e:
        pass

    return [4, 4, 4], "Failed to parse model response"  # Fallback

def clear_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def evaluate_test_set_efficient(model_path, test_data_path="data/survey_data_ps_deduped.csv"):
    """Memory-efficient evaluation with periodic cleanup"""

    print("="*70)
    print("MEMORY-EFFICIENT FINE-TUNED MODEL EVALUATION")
    print("="*70)

    # Load split info
    split_info_path = "finetuned/data/split_info.json"
    with open(split_info_path, 'r') as f:
        split_info = json.load(f)

    test_tweet_ids = set(split_info['test_tweet_ids'])
    print(f"Test set: {len(test_tweet_ids)} tweets")

    # Load data
    df = pd.read_csv(test_data_path)
    test_df = df[df['Tweet ID'].isin(test_tweet_ids)].reset_index(drop=True)
    print(f"Loaded {len(test_df)} test tweets\n")

    # Load model ONCE
    model, tokenizer = load_model_once(model_path)

    # Results storage
    results = []
    checkpoint_file = "finetuned/data/test_results_checkpoint.json"

    # Resume from checkpoint if exists
    start_idx = 0
    if os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint...")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            results = checkpoint.get('results', [])
            start_idx = len(results)
            print(f"Resuming from tweet {start_idx}/{len(test_df)}")

    # Evaluate
    print("Evaluating...")
    for idx in tqdm(range(start_idx, len(test_df)), desc="Tweets"):
        row = test_df.iloc[idx]
        tweet_text = row['Tweet Text']
        tweet_id = row['Tweet ID']

        result = {
            'tweet_id': int(tweet_id),  # Convert to Python int
            'tweet_text': str(tweet_text),  # Ensure string
            'human_scores': {},
            'model_scores': {},
            'model_comments': {}
        }

        for dim_key, dim_info in EVALUATION_DIMENSIONS.items():
            # Get human scores
            human_scores = [
                int(row[dim_info['rating_cols'][0]]),
                int(row[dim_info['rating_cols'][1]]),
                int(row[dim_info['rating_cols'][2]])
            ]

            # Get model scores and comment
            model_scores, model_comment = evaluate_single(model, tokenizer, tweet_text, dim_key, dim_info)

            result['human_scores'][dim_key] = human_scores
            result['model_scores'][dim_key] = model_scores
            result['model_comments'][dim_key] = model_comment

            # Clear cache after each dimension
            clear_memory()

        results.append(result)

        # Save checkpoint every 10 tweets
        if (idx + 1) % 10 == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump({'results': results, 'last_idx': idx}, f)
            print(f"\nCheckpoint saved at tweet {idx + 1}/{len(test_df)}")
            clear_memory()

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"finetuned/data/test_results_{timestamp}.json"

    output = {
        'model_path': model_path,
        'test_set_size': len(test_df),
        'metrics': metrics,
        'detailed_predictions': results,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed")

    return results, metrics

def compute_metrics(results):
    """Compute evaluation metrics"""
    metrics = {}

    for dim in ['I-Involvement', 'You-Involvement', 'We-Involvement']:
        human_all = []
        model_all = []

        for result in results:
            human_all.extend(result['human_scores'][dim])
            model_all.extend(result['model_scores'][dim])

        # MAE
        mae = np.mean(np.abs(np.array(human_all) - np.array(model_all)))

        # Pearson
        pearson = np.corrcoef(human_all, model_all)[0, 1]

        # Within-1
        within_1 = np.mean(np.abs(np.array(human_all) - np.array(model_all)) <= 1)

        # 2-Level accuracy
        human_levels = ['low' if s <= 4 else 'high' for s in human_all]
        model_levels = ['low' if s <= 4 else 'high' for s in model_all]
        level_acc = np.mean([h == m for h, m in zip(human_levels, model_levels)])

        metrics[dim] = {
            'mae': float(mae),
            'pearson': float(pearson),
            'within_1': float(within_1),
            'level_accuracy': float(level_acc)
        }

    # Average
    metrics['average'] = {
        'mae': float(np.mean([m['mae'] for m in metrics.values()])),
        'pearson': float(np.mean([m['pearson'] for m in metrics.values()])),
        'within_1': float(np.mean([m['within_1'] for m in metrics.values()])),
        'level_accuracy': float(np.mean([m['level_accuracy'] for m in metrics.values()]))
    }

    return metrics

def print_metrics(metrics):
    """Print metrics"""
    print()
    print("="*70)
    print("TEST SET RESULTS (2-LEVEL)")
    print("="*70)

    for dim in ['I-Involvement', 'You-Involvement', 'We-Involvement']:
        m = metrics[dim]
        print(f"\n{dim}:")
        print(f"  MAE:           {m['mae']:.3f}")
        print(f"  Pearson r:     {m['pearson']:.3f}")
        print(f"  Within-1 Acc:  {m['within_1']:.1%}")
        print(f"  Level Acc:     {m['level_accuracy']:.1%} (2-level)")

    avg = metrics['average']
    print(f"\nAVERAGE:")
    print(f"  MAE:           {avg['mae']:.3f}")
    print(f"  Pearson r:     {avg['pearson']:.3f}")
    print(f"  Within-1 Acc:  {avg['within_1']:.1%}")
    print(f"  Level Acc:     {avg['level_accuracy']:.1%} ⭐ (2-level)")
    print("="*70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str,
        default="finetuned/models/gemma-2-9b-tweet-eval/final",
        help="Path to fine-tuned model"
    )

    args = parser.parse_args()

    evaluate_test_set_efficient(args.model_path)
