#!/usr/bin/env python3
"""
Gemma-3 based tweet evaluation system that mirrors the human survey process.
Evaluates tweets on three involvement dimensions: I (Individual), You (Relational), We (Collective)
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from config import GEMMA_MODELS, DEFAULT_MODEL, VRAM_REQUIREMENTS

# Load environment variables
load_dotenv()

class TweetEvaluator:
    def __init__(self, model_name=None, gpu_id=None):
        """Initialize the Gemma model for tweet evaluation"""
        # Use provided model name or default
        if model_name is None:
            model_name = DEFAULT_MODEL

        print(f"Loading Gemma model: {model_name}")

        # Check VRAM requirements
        if model_name in VRAM_REQUIREMENTS:
            print(f"Estimated VRAM requirement: {VRAM_REQUIREMENTS[model_name]}")

        # Get HuggingFace token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in .env file. Please add your HuggingFace token.")

        print("Using HuggingFace token for authentication...")

        # Check CUDA availability and force GPU usage
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available!")
            print("Please run: ./install_pytorch_cuda.sh")
            print("Or update NVIDIA drivers if needed.")
            raise RuntimeError("GPU is required but CUDA is not available")

        print(f"CUDA available - using GPU")
        print(f"GPU count: {torch.cuda.device_count()}")

        # Determine which GPU to use
        if gpu_id is not None:
            gpu_idx = int(gpu_id)
            print(f"Using specified GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")
        else:
            # Default to GPU:1 for backwards compatibility
            gpu_idx = 1
            print(f"Using default GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")

        # Load tokenizer and model - using Gemma-2 which is text-only
        self.device = torch.device(f"cuda:{gpu_idx}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        # Load model with GPU support
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": gpu_idx},
            torch_dtype=torch.bfloat16,
            token=hf_token
        ).eval()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define evaluation criteria
        self.evaluation_dimensions = {
            "I-Involvement": {
                "title": "I-Involvement Recognition (Individual)",
                "questions": [
                    "The tweet expresses concern about the issue's impact on the self",
                    "The tweet indicates that the issue has personal consequences for the speaker",
                    "The tweet emphasizes individual experience or personal situation regarding the issue"
                ]
            },
            "You-Involvement": {
                "title": "You-Involvement Recognition (Relational)",
                "questions": [
                    "The tweet expresses concern about the issue's impact on family, friends, or close others",
                    "The tweet indicates consequences for someone the speaker cares about",
                    "The tweet emphasizes relational ties (support, loyalty, care) in relation to the issue"
                ]
            },
            "We-Involvement": {
                "title": "We-Involvement Recognition (Collective)",
                "questions": [
                    "The tweet expresses concern about the issue's impact on the community, group, or nation",
                    "The tweet indicates collective consequences (e.g., society, city, country, 'us')",
                    "The tweet emphasizes shared responsibility, solidarity, or collective well-being regarding the issue"
                ]
            }
        }

    def create_evaluation_prompt(self, tweet_text, dimension, questions):
        """Create a structured prompt for Gemma-3 to evaluate a tweet"""
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

    def evaluate_single_tweet(self, tweet_text):
        """Evaluate a single tweet across all dimensions"""
        results = {}

        for dimension_key, dimension_info in self.evaluation_dimensions.items():
            prompt = self.create_evaluation_prompt(
                tweet_text,
                dimension_info["title"],
                dimension_info["questions"]
            )

            # Prepare messages for the model using Gemma chat format
            chat = [
                {"role": "user", "content": prompt}
            ]

            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            # Generate response
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.1,  # Low temperature for consistent evaluation
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode response - only the generated part
            response = self.tokenizer.decode(
                generation[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

            # Debug: Print raw response for problematic dimensions
            print(f"\n==== Raw Response for {dimension_key} ====")
            print(response)
            print("=" * 50)

            # Preprocessing: Remove markdown code blocks
            response = response.replace('```json', '').replace('```', '').strip()

            # Parse JSON response
            try:
                # Try to find JSON in response with multiple strategies
                import re
                evaluation = None

                # Strategy 0: Manual extraction for problematic cases
                # Extract scores and comment separately, bypassing JSON parsing issues
                scores_match = re.search(r'"scores":\s*\[([^\]]+)\]', response)
                comment_match = re.search(r'"comment":\s*"(.+)"\s*\}', response, re.DOTALL)

                if scores_match and comment_match:
                    try:
                        scores_str = scores_match.group(1)
                        scores = [int(s.strip()) for s in scores_str.split(',')]
                        comment = comment_match.group(1)

                        evaluation = {
                            "scores": scores,
                            "comment": comment
                        }
                        print("✓ Strategy 0 succeeded (manual extraction)")
                    except Exception as e0:
                        print(f"✗ Strategy 0 failed: {type(e0).__name__}")
                        evaluation = None

                # Strategy 1: Try to parse the entire response as JSON
                if evaluation is None:
                    try:
                        evaluation = json.loads(response)
                        print("✓ Strategy 1 succeeded (whole response)")
                    except Exception as e1:
                        print(f"✗ Strategy 1 failed: {type(e1).__name__}")

                # Strategy 2: Find JSON block with balanced braces
                if evaluation is None:
                    # Find all { } pairs and extract the longest valid JSON
                    json_candidates = []
                    start_idx = 0
                    while True:
                        start = response.find('{', start_idx)
                        if start == -1:
                            break

                        # Find matching closing brace
                        brace_count = 0
                        for i in range(start, len(response)):
                            if response[i] == '{':
                                brace_count += 1
                            elif response[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_candidates.append(response[start:i+1])
                                    break
                        start_idx = start + 1

                    print(f"Found {len(json_candidates)} JSON candidates")
                    # Try each candidate
                    for idx, candidate in enumerate(json_candidates):
                        try:
                            evaluation = json.loads(candidate)
                            print(f"✓ Strategy 2 succeeded (candidate {idx})")
                            break
                        except Exception as e2:
                            print(f"✗ Candidate {idx} failed: {type(e2).__name__}")
                            continue

                # Strategy 3: Simple regex fallback
                if evaluation is None:
                    json_match = re.search(r'\{[^}]+\}', response)
                    if json_match:
                        try:
                            evaluation = json.loads(json_match.group())
                            print("✓ Strategy 3 succeeded (regex)")
                        except Exception as e3:
                            print(f"✗ Strategy 3 failed: {type(e3).__name__}")

                if evaluation:
                    print("---- Response Parsed Successfully ----")
                    print(evaluation)
                    print("--------------------------------------")
                else:
                    raise ValueError("Could not extract valid JSON from response")

                # Ensure scores exist and are valid
                if "scores" in evaluation and isinstance(evaluation["scores"], list):
                    # Validate scores are in range 1-7
                    scores = []
                    for score in evaluation["scores"]:
                        try:
                            s = int(score)
                            if 1 <= s <= 7:
                                scores.append(s)
                            else:
                                scores.append(4)  # Default to neutral
                        except:
                            scores.append(4)  # Default to neutral

                    # Store structured result
                    results[dimension_key] = {
                        "scores": scores[:3],  # Ensure exactly 3 scores
                        "average": sum(scores[:3]) / min(len(scores), 3) if scores else 4.0,
                        "comment": evaluation.get("comment", f"Evaluated {dimension_key} based on tweet content")
                    }
                else:
                    # Fallback - default scores
                    print(f"Warning: Invalid scores for {dimension_key}, using defaults")
                    results[dimension_key] = {
                        "scores": [4, 4, 4],
                        "average": 4.0,
                        "comment": "Unable to evaluate - using neutral scores",
                        "error": "Could not parse scores"
                    }

            except (json.JSONDecodeError, Exception) as e:
                # Fallback if JSON parsing fails completely
                print(f"Warning: Could not parse response for {dimension_key}: {str(e)}")
                results[dimension_key] = {
                    "scores": [4, 4, 4],  # Default neutral scores
                    "average": 4.0,
                    "comment": f"Error in evaluation: {str(e)[:100]}",
                    "error": str(e),
                    "raw_response": response[:200] if response else ""
                }

        return results

    def evaluate_dataset(self, csv_path, output_path=None, max_retries=2):
        """Evaluate all tweets in the dataset with retry mechanism"""
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} tweets from {csv_path}")

        # Prepare output
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"zero_shot/data/evaluations/gemma_k0_{timestamp}.json"

        evaluations = []
        failed_tweets = []  # Track failed tweets for retry

        # Process each tweet
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating tweets"):
            tweet_id = row['Tweet ID']
            tweet_text = row['Tweet Text']

            print(f"\nEvaluating Tweet {idx+1}/{len(df)}: {tweet_text[:100]}...")

            try:
                evaluation = self.evaluate_single_tweet(tweet_text)

                # Check if any dimension has errors
                has_error = any(
                    "error" in dim_eval
                    for dim_eval in evaluation.values()
                )

                # Add metadata
                result = {
                    "tweet_id": tweet_id,
                    "tweet_text": tweet_text,
                    "source_dataset": row.get('source_dataset', 'unknown'),
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                    "has_error": has_error
                }

                if has_error:
                    failed_tweets.append((row, result))
                    print(f"  ⚠️ Tweet {tweet_id} had evaluation errors, will retry later")

                evaluations.append(result)

                # Save intermediate results every 10 tweets
                if (idx + 1) % 10 == 0:
                    with open(output_path, 'w') as f:
                        json.dump(evaluations, f, indent=2)
                    print(f"Saved intermediate results ({idx+1} tweets)")

            except Exception as e:
                print(f"Error evaluating tweet {tweet_id}: {str(e)}")
                failed_result = {
                    "tweet_id": tweet_id,
                    "tweet_text": tweet_text,
                    "source_dataset": row.get('source_dataset', 'unknown'),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "has_error": True
                }
                evaluations.append(failed_result)
                failed_tweets.append((row, failed_result))

        # Retry failed tweets
        if failed_tweets and max_retries > 0:
            print(f"\n{'='*60}")
            print(f"Retrying {len(failed_tweets)} failed tweets...")
            print(f"{'='*60}")

            for retry_num in range(1, max_retries + 1):
                if not failed_tweets:
                    break

                print(f"\n--- Retry attempt {retry_num}/{max_retries} ---")
                newly_failed = []

                for row_data, original_result in tqdm(failed_tweets, desc=f"Retry {retry_num}"):
                    tweet_id = row_data['Tweet ID']
                    tweet_text = row_data['Tweet Text']

                    print(f"\nRetrying Tweet ID {tweet_id}: {tweet_text[:100]}...")

                    try:
                        evaluation = self.evaluate_single_tweet(tweet_text)

                        # Check if evaluation is successful
                        has_error = any(
                            "error" in dim_eval
                            for dim_eval in evaluation.values()
                        )

                        if not has_error:
                            # Update the successful evaluation in the list
                            for i, eval_item in enumerate(evaluations):
                                if eval_item["tweet_id"] == tweet_id:
                                    evaluations[i] = {
                                        "tweet_id": tweet_id,
                                        "tweet_text": tweet_text,
                                        "source_dataset": row_data.get('source_dataset', 'unknown'),
                                        "evaluation": evaluation,
                                        "timestamp": datetime.now().isoformat(),
                                        "has_error": False,
                                        "retry_count": retry_num
                                    }
                                    print(f"  ✅ Successfully evaluated on retry {retry_num}")
                                    break
                        else:
                            newly_failed.append((row_data, original_result))
                            print(f"  ⚠️ Still has errors after retry {retry_num}")

                    except Exception as e:
                        print(f"  ❌ Retry failed: {str(e)}")
                        newly_failed.append((row_data, original_result))

                failed_tweets = newly_failed

            if failed_tweets:
                print(f"\n⚠️ {len(failed_tweets)} tweets still failed after {max_retries} retries")

        # Move failed tweets to the end
        successful_evals = [e for e in evaluations if not e.get("has_error", False)]
        failed_evals = [e for e in evaluations if e.get("has_error", False)]

        evaluations = successful_evals + failed_evals

        # Save final results
        with open(output_path, 'w') as f:
            json.dump(evaluations, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Evaluation complete!")
        print(f"✅ Successful: {len(successful_evals)}")
        print(f"❌ Failed: {len(failed_evals)}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}")

        # Also save as CSV for easier analysis
        csv_output = output_path.replace('.json', '.csv')
        self.save_as_csv(evaluations, csv_output)

        return evaluations

    def save_as_csv(self, evaluations, csv_path):
        """Convert evaluations to CSV format for easier analysis"""
        rows = []

        for eval_item in evaluations:
            if "error" in eval_item and eval_item["error"]:
                # Skip completely failed evaluations
                continue

            row = {
                "tweet_id": eval_item["tweet_id"],
                "tweet_text": eval_item["tweet_text"],
                "source_dataset": eval_item.get("source_dataset", ""),
                "timestamp": eval_item.get("timestamp", "")
            }

            # Extract scores for each dimension
            for dimension in ["I-Involvement", "You-Involvement", "We-Involvement"]:
                if dimension in eval_item["evaluation"]:
                    dim_eval = eval_item["evaluation"][dimension]
                    if "scores" in dim_eval and isinstance(dim_eval["scores"], list):
                        scores = dim_eval["scores"]
                        # Add individual question scores
                        for i in range(3):
                            if i < len(scores):
                                row[f"{dimension}_Q{i+1}"] = scores[i]
                            else:
                                row[f"{dimension}_Q{i+1}"] = 4  # Default neutral

                        # Add average score
                        if "average" in dim_eval:
                            row[f"{dimension}_avg"] = dim_eval["average"]
                        else:
                            row[f"{dimension}_avg"] = sum(scores[:3]) / min(len(scores), 3) if scores else 4.0

                        # Add comment
                        if "comment" in dim_eval:
                            row[f"{dimension}_comment"] = dim_eval["comment"]
                    else:
                        # Default values if scores missing
                        for i in range(1, 4):
                            row[f"{dimension}_Q{i}"] = 4
                        row[f"{dimension}_avg"] = 4.0

            rows.append(row)

        if rows:
            df_results = pd.DataFrame(rows)

            # Reorder columns for better readability
            column_order = ["tweet_id", "tweet_text", "source_dataset"]

            # Add I-Involvement columns
            for i in range(1, 4):
                column_order.append(f"I-Involvement_Q{i}")
            column_order.append("I-Involvement_avg")
            if f"I-Involvement_comment" in df_results.columns:
                column_order.append("I-Involvement_comment")

            # Add You-Involvement columns
            for i in range(1, 4):
                column_order.append(f"You-Involvement_Q{i}")
            column_order.append("You-Involvement_avg")
            if f"You-Involvement_comment" in df_results.columns:
                column_order.append("You-Involvement_comment")

            # Add We-Involvement columns
            for i in range(1, 4):
                column_order.append(f"We-Involvement_Q{i}")
            column_order.append("We-Involvement_avg")
            if f"We-Involvement_comment" in df_results.columns:
                column_order.append("We-Involvement_comment")

            column_order.append("timestamp")

            # Filter column_order to only include existing columns
            column_order = [col for col in column_order if col in df_results.columns]

            # Reorder dataframe
            df_results = df_results[column_order]

            df_results.to_csv(csv_path, index=False)
            print(f"CSV results saved to: {csv_path}")
        else:
            print("No valid evaluations to save to CSV")

    def evaluate_on_validation(self, val_df):
        """
        Evaluate model on validation set and compare with human annotations

        Args:
            val_df: Validation dataframe with human annotations

        Returns:
            List of comparison results
        """
        from tqdm import tqdm

        validation_results = []

        print(f"\nEvaluating on {len(val_df)} validation examples...")

        for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validation"):
            tweet_text = row['Tweet Text']
            tweet_id = row.get('Tweet ID', None)

            # Get model predictions
            model_eval = self.evaluate_single_tweet(tweet_text)

            # Extract human scores
            human_scores = {}
            model_scores = {}

            # Define evaluation dimensions (same as in __init__)
            evaluation_dimensions = {
                "I-Involvement": ["I-Involvement-1", "I-Involvement-2", "I-Involvement-3"],
                "You-Involvement": ["You-Involvement-1", "You-Involvement-2", "You-Involvement-3"],
                "We-Involvement": ["We-Involvement-1", "We-Involvement-2", "We-Involvement-3"]
            }

            for dimension_key, rating_cols in evaluation_dimensions.items():
                # Human scores
                human = [
                    int(row[rating_cols[0]]),
                    int(row[rating_cols[1]]),
                    int(row[rating_cols[2]])
                ]
                human_scores[dimension_key] = human

                # Model scores (if available)
                if dimension_key in model_eval and 'scores' in model_eval[dimension_key]:
                    model = model_eval[dimension_key]['scores']
                    model_scores[dimension_key] = model
                else:
                    model_scores[dimension_key] = [4, 4, 4]  # Default

            validation_results.append({
                'tweet_id': tweet_id,
                'tweet_text': tweet_text,
                'human_scores': human_scores,
                'model_scores': model_scores
            })

        return validation_results

    def compute_metrics(self, validation_results):
        """
        Compute evaluation metrics comparing model to human scores

        Args:
            validation_results: List of dicts with human and model scores

        Returns:
            Dict of metrics per dimension
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from scipy.stats import pearsonr, spearmanr
        import numpy as np

        def score_to_level(score):
            """Convert score (1-7) to level (low/middle/high)"""
            if score <= 3:
                return 'low'
            elif score == 4:
                return 'middle'
            else:  # 5-7
                return 'high'

        metrics = {}

        for dimension_key in ["I-Involvement", "You-Involvement", "We-Involvement"]:
            human_all = []
            model_all = []

            for result in validation_results:
                human = result['human_scores'][dimension_key]
                model = result['model_scores'][dimension_key]

                human_all.extend(human)
                model_all.extend(model)

            # Calculate metrics
            mae = mean_absolute_error(human_all, model_all)
            rmse = np.sqrt(mean_squared_error(human_all, model_all))
            pearson_r, pearson_p = pearsonr(human_all, model_all)
            spearman_r, spearman_p = spearmanr(human_all, model_all)

            # Accuracy (exact match)
            exact_match = sum(h == m for h, m in zip(human_all, model_all)) / len(human_all)

            # Within-1 accuracy
            within_1 = sum(abs(h - m) <= 1 for h, m in zip(human_all, model_all)) / len(human_all)

            # Level-based accuracy (LOW: 1-3, MIDDLE: 4, HIGH: 5-7)
            human_levels = [score_to_level(s) for s in human_all]
            model_levels = [score_to_level(s) for s in model_all]
            level_accuracy = sum(h == m for h, m in zip(human_levels, model_levels)) / len(human_levels)

            metrics[dimension_key] = {
                'mae': mae,
                'rmse': rmse,
                'pearson_r': pearson_r,
                'spearman_r': spearman_r,
                'exact_match': exact_match,
                'within_1_acc': within_1,
                'level_accuracy': level_accuracy
            }

        return metrics

    def print_metrics(self, metrics):
        """Pretty print metrics"""
        import numpy as np

        print("\n" + "="*70)
        print("EVALUATION METRICS")
        print("="*70)

        for dimension_key, dim_metrics in metrics.items():
            print(f"\n{dimension_key}:")
            print(f"  MAE:           {dim_metrics['mae']:.3f}")
            print(f"  RMSE:          {dim_metrics['rmse']:.3f}")
            print(f"  Pearson r:     {dim_metrics['pearson_r']:.3f}")
            print(f"  Spearman r:    {dim_metrics['spearman_r']:.3f}")
            print(f"  Exact Match:   {dim_metrics['exact_match']:.1%}")
            print(f"  Within-1 Acc:  {dim_metrics['within_1_acc']:.1%}")
            print(f"  Level Acc:     {dim_metrics['level_accuracy']:.1%} ⭐ (Low/Mid/High)")

        # Average across dimensions
        avg_mae = np.mean([m['mae'] for m in metrics.values()])
        avg_pearson = np.mean([m['pearson_r'] for m in metrics.values()])
        avg_within_1 = np.mean([m['within_1_acc'] for m in metrics.values()])
        avg_level = np.mean([m['level_accuracy'] for m in metrics.values()])

        print(f"\nAVERAGE ACROSS DIMENSIONS:")
        print(f"  MAE:           {avg_mae:.3f}")
        print(f"  Pearson r:     {avg_pearson:.3f}")
        print(f"  Within-1 Acc:  {avg_within_1:.1%}")
        print(f"  Level Acc:     {avg_level:.1%} ⭐ (Primary Metric)")
        print("="*70 + "\n")

        return avg_mae, avg_pearson, avg_within_1, avg_level

def main():
    """Main execution function"""
    # Initialize evaluator (will use DEFAULT_MODEL from config)
    evaluator = TweetEvaluator()

    # Path to the tweet dataset
    # csv_path = "data/targeted_tweets.csv"
    csv_path = "data/survey_data_ps_deduped.csv"

    # Run evaluation with retries
    evaluator.evaluate_dataset(csv_path, max_retries=2)

if __name__ == "__main__":
    main()