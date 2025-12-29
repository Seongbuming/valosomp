#!/usr/bin/env python3
"""
Few-shot Gemma-3 based tweet evaluation system.
Uses human-annotated MTurk examples as few-shot demonstrations,
selected based on semantic similarity to the target tweet.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
from datetime import datetime
import os
import sys
from dotenv import load_dotenv
import numpy as np

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import from local config (few_shot/src/config.py)
from config import GEMMA_MODELS, DEFAULT_MODEL, VRAM_REQUIREMENTS

# Load environment variables
load_dotenv()


class FewShotTweetEvaluator:
    def __init__(self, model_name=None, k_shots=4, embedding_model="all-MiniLM-L6-v2",
                 temperature=0.1, top_p=1.0, use_dimension_specific=False, gpu_id="auto"):
        """
        Initialize the Few-Shot Gemma evaluator

        Args:
            model_name: Gemma model to use for evaluation
            k_shots: Number of few-shot examples to include (1, 4, 8, or 16)
            embedding_model: Sentence transformer model for similarity computation
            temperature: Sampling temperature for generation
            top_p: Nucleus sampling parameter
            use_dimension_specific: Use dimension-specific example selection
            gpu_id: GPU to use ("auto" for automatic selection, or integer like 0, 1, etc.)
        """
        self.k_shots = k_shots
        self.temperature = temperature
        self.top_p = top_p
        self.use_dimension_specific = use_dimension_specific
        self.calibration_params = {}  # Stores calibration parameters per dimension
        self.gpu_id = gpu_id

        # Use provided model name or default
        if model_name is None:
            model_name = DEFAULT_MODEL

        print(f"Loading Gemma model: {model_name}")
        print(f"Using {k_shots}-shot learning")

        # Check VRAM requirements
        if model_name in VRAM_REQUIREMENTS:
            print(f"Estimated VRAM requirement: {VRAM_REQUIREMENTS[model_name]}")

        # Get HuggingFace token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

        # Check CUDA availability
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available!")
            raise RuntimeError("GPU is required but CUDA is not available")

        print(f"CUDA available - using GPU")
        print(f"GPU count: {torch.cuda.device_count()}")

        # Determine which GPU to use
        if self.gpu_id == "auto":
            # Automatic GPU selection - use the one with most free memory
            if torch.cuda.device_count() > 1:
                # Get free memory for each GPU
                free_memory = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    free, total = torch.cuda.mem_get_info(i)
                    free_memory.append(free)
                    print(f"GPU {i} ({torch.cuda.get_device_name(i)}): {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")

                # Select GPU with most free memory
                selected_gpu = int(np.argmax(free_memory))
                print(f"Auto-selected GPU {selected_gpu} with most free memory")
            else:
                selected_gpu = 0
                print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            # Use specified GPU
            selected_gpu = int(self.gpu_id)
            if selected_gpu >= torch.cuda.device_count():
                raise ValueError(f"GPU {selected_gpu} not available. Only {torch.cuda.device_count()} GPUs found.")
            print(f"Using specified GPU {selected_gpu}: {torch.cuda.get_device_name(selected_gpu)}")

        # Load Gemma model and tokenizer
        self.device = torch.device(f"cuda:{selected_gpu}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": selected_gpu},
            torch_dtype=torch.bfloat16,
            token=hf_token
        ).eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load sentence transformer for similarity-based example selection
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Load and preprocess MTurk data for few-shot examples
        self.load_fewshot_pool()

        # Define evaluation dimensions
        self.evaluation_dimensions = {
            "I-Involvement": {
                "title": "I-Involvement Recognition (Individual)",
                "questions": [
                    "The tweet expresses concern about the issue's impact on the self",
                    "The tweet indicates that the issue has personal consequences for the speaker",
                    "The tweet emphasizes individual experience or personal situation regarding the issue"
                ],
                "comment_col": "I-Involvement Comment",
                "rating_cols": ["I-Involvement-1", "I-Involvement-2", "I-Involvement-3"]
            },
            "You-Involvement": {
                "title": "You-Involvement Recognition (Relational)",
                "questions": [
                    "The tweet expresses concern about the issue's impact on family, friends, or close others",
                    "The tweet indicates consequences for someone the speaker cares about",
                    "The tweet emphasizes relational ties (support, loyalty, care) in relation to the issue"
                ],
                "comment_col": "You-Involvement Comment",
                "rating_cols": ["You-Involvement-1", "You-Involvement-2", "You-Involvement-3"]
            },
            "We-Involvement": {
                "title": "We-Involvement Recognition (Collective)",
                "questions": [
                    "The tweet expresses concern about the issue's impact on the community, group, or nation",
                    "The tweet indicates collective consequences (e.g., society, city, country, 'us')",
                    "The tweet emphasizes shared responsibility, solidarity, or collective well-being regarding the issue"
                ],
                "comment_col": "We-Involvement Comment",
                "rating_cols": ["We-Involvement-1", "We-Involvement-2", "We-Involvement-3"]
            }
        }

    def load_fewshot_pool(self):
        """Load MTurk survey data to use as few-shot example pool"""
        mturk_path = "data/survey_data_ps_deduped.csv"
        print(f"Loading few-shot example pool from {mturk_path}")

        self.mturk_df = pd.read_csv(mturk_path)
        print(f"Loaded {len(self.mturk_df)} human-annotated examples")

        # Precompute embeddings for all MTurk tweets for efficient similarity search
        print("Computing embeddings for few-shot example pool...")
        self.mturk_tweets = self.mturk_df['Tweet Text'].tolist()
        self.mturk_embeddings = self.embedding_model.encode(
            self.mturk_tweets,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        print("Embeddings computed and cached")

    def select_fewshot_examples(self, target_tweet, target_tweet_id=None, k=None, exclude_same_text=True):
        """
        Select k most similar examples from MTurk data for few-shot learning

        Args:
            target_tweet: The tweet to evaluate
            target_tweet_id: Optional Tweet ID to exclude from examples
            k: Number of examples to select (defaults to self.k_shots)
            exclude_same_text: If True, exclude examples with identical tweet text

        Returns:
            List of k most similar example indices
        """
        if k is None:
            k = self.k_shots

        # Encode target tweet
        target_embedding = self.embedding_model.encode(
            target_tweet,
            convert_to_tensor=True
        )

        # Compute cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            target_embedding.unsqueeze(0),
            self.mturk_embeddings
        )

        # Exclude examples that match the target tweet (prevent data leakage)
        excluded_count = 0
        if exclude_same_text:
            for i, mturk_tweet in enumerate(self.mturk_tweets):
                # Check for exact text match (case-insensitive and whitespace-normalized)
                if mturk_tweet.strip().lower() == target_tweet.strip().lower():
                    similarities[i] = float('-inf')
                    excluded_count += 1
                    continue

                # Also check Tweet ID if provided
                if target_tweet_id is not None and 'Tweet ID' in self.mturk_df.columns:
                    mturk_tweet_id = self.mturk_df.iloc[i]['Tweet ID']
                    if str(mturk_tweet_id) == str(target_tweet_id):
                        similarities[i] = float('-inf')
                        excluded_count += 1

        if excluded_count > 0:
            print(f"  ℹ️ Excluded {excluded_count} identical tweet(s) from few-shot examples to prevent data leakage")

        # Get top-k indices (return empty list for zero-shot)
        if k == 0:
            return []

        top_k_indices = torch.topk(similarities, min(k, len(similarities))).indices

        return top_k_indices.cpu().tolist()

    def select_dimension_specific_examples(self, target_tweet, target_tweet_id, dimension_key, k=None):
        """
        Select examples specifically optimized for a given dimension

        Args:
            target_tweet: The tweet to evaluate
            target_tweet_id: Optional Tweet ID to exclude
            dimension_key: Which dimension (I/You/We-Involvement)
            k: Number of examples to select

        Returns:
            List of k most suitable example indices for this dimension
        """
        if k is None:
            k = self.k_shots

        # Return empty list for zero-shot
        if k == 0:
            return []

        # Get broader candidate pool (3x the target number)
        candidate_pool_size = min(k * 3, len(self.mturk_df))
        candidates = self.select_fewshot_examples(
            target_tweet,
            target_tweet_id=target_tweet_id,
            k=candidate_pool_size,
            exclude_same_text=True
        )

        # Get dimension info
        dimension_info = self.evaluation_dimensions[dimension_key]
        rating_cols = dimension_info['rating_cols']

        # Score each candidate based on dimension-specific quality
        scored_candidates = []
        for idx in candidates:
            row = self.mturk_df.iloc[idx]

            # Get scores for this dimension
            scores = [
                float(row[rating_cols[0]]),
                float(row[rating_cols[1]]),
                float(row[rating_cols[2]])
            ]

            # Quality metrics:
            # 1. Variance: High variance = clear signal
            variance = np.var(scores)

            # 2. Extremity: Distance from neutral (4)
            avg_score = np.mean(scores)
            extremity = abs(avg_score - 4.0)

            # 3. Comment quality: Has non-empty comment
            comment = row[dimension_info['comment_col']]
            has_comment = 1.0 if (isinstance(comment, str) and len(comment.strip()) > 10) else 0.0

            # Combined quality score
            quality = variance * 0.4 + extremity * 0.4 + has_comment * 0.2

            scored_candidates.append((idx, quality))

        # Sort by quality and select top-k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in scored_candidates[:k]]

        return selected_indices

    def fit_calibration(self, validation_results):
        """
        Fit calibration parameters based on validation results

        Args:
            validation_results: List of dicts with 'human_scores' and 'model_scores'
                               for each dimension
        """
        from sklearn.linear_model import LinearRegression

        print("\nFitting calibration parameters...")

        for dimension_key in ["I-Involvement", "You-Involvement", "We-Involvement"]:
            human_scores = []
            model_scores = []

            # Collect all scores for this dimension
            for result in validation_results:
                if dimension_key in result['human_scores'] and dimension_key in result['model_scores']:
                    human = result['human_scores'][dimension_key]
                    model = result['model_scores'][dimension_key]

                    # Each dimension has 3 questions
                    for i in range(min(len(human), len(model))):
                        human_scores.append(human[i])
                        model_scores.append(model[i])

            if len(human_scores) < 10:
                print(f"  {dimension_key}: Insufficient data ({len(human_scores)} points), skipping calibration")
                continue

            # Fit linear regression: human = slope * model + intercept
            X = np.array(model_scores).reshape(-1, 1)
            y = np.array(human_scores)

            reg = LinearRegression()
            reg.fit(X, y)

            # Calculate R² score
            from sklearn.metrics import r2_score
            predictions = reg.predict(X)
            r2 = r2_score(y, predictions)

            self.calibration_params[dimension_key] = {
                'slope': float(reg.coef_[0]),
                'intercept': float(reg.intercept_),
                'r2': r2
            }

            print(f"  {dimension_key}: slope={reg.coef_[0]:.3f}, intercept={reg.intercept_:.3f}, R²={r2:.3f}")

        print("Calibration fitting complete!\n")

    def calibrate_scores(self, scores, dimension_key):
        """
        Apply calibration to model scores

        Args:
            scores: List of model scores
            dimension_key: Which dimension

        Returns:
            Calibrated scores (clamped to 1-7 range)
        """
        if dimension_key not in self.calibration_params:
            return scores

        params = self.calibration_params[dimension_key]
        calibrated = []

        for score in scores:
            cal_score = params['slope'] * score + params['intercept']
            # Clamp to valid range [1, 7]
            cal_score = max(1.0, min(7.0, cal_score))
            calibrated.append(round(cal_score))

        return calibrated

    def format_fewshot_example(self, example_idx, dimension_key):
        """
        Format a single few-shot example for the prompt

        Args:
            example_idx: Index in MTurk dataframe
            dimension_key: Which dimension (I/You/We-Involvement)

        Returns:
            Formatted example string
        """
        row = self.mturk_df.iloc[example_idx]
        dimension_info = self.evaluation_dimensions[dimension_key]

        tweet_text = row['Tweet Text']
        comment = row[dimension_info['comment_col']]
        scores = [
            int(row[dimension_info['rating_cols'][0]]),
            int(row[dimension_info['rating_cols'][1]]),
            int(row[dimension_info['rating_cols'][2]])
        ]

        # Escape quotes in comment to prevent JSON issues
        comment_escaped = comment.replace('"', '\\"')

        example = f"""Example:
Tweet: "{tweet_text}"

{{
    "scores": {scores},
    "comment": "{comment_escaped}"
}}"""

        return example

    def create_fewshot_prompt(self, tweet_text, dimension, questions, example_indices):
        """
        Create a few-shot prompt with similar examples

        Args:
            tweet_text: Target tweet to evaluate
            dimension: Dimension name
            questions: List of evaluation questions
            example_indices: Indices of examples to include

        Returns:
            Formatted prompt string
        """
        # Extract dimension key from dimension title
        dimension_key = None
        for key, info in self.evaluation_dimensions.items():
            if info['title'] == dimension:
                dimension_key = key
                break

        if dimension_key is None:
            raise ValueError(f"Unknown dimension: {dimension}")

        prompt = f"""You are evaluating a tweet about a disaster or crisis situation.
Please evaluate how much you agree with the following statements about this tweet.

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

        prompt += "\n"

        # Add few-shot examples
        if example_indices:
            prompt += f"Here are {len(example_indices)} example(s) of how similar tweets were evaluated:\n\n"

            for idx in example_indices:
                example = self.format_fewshot_example(idx, dimension_key)
                prompt += example + "\n\n"

        # Add the target tweet
        prompt += f"""Now evaluate this tweet:
Tweet: "{tweet_text}"

Please respond in JSON format with scores and a brief explanation:
{{
    "scores": [score1, score2, score3],
    "comment": "brief explanation of why you gave these scores for this dimension"
}}

Provide exactly 3 scores (integers from 1-7) for the 3 statements above, and a brief comment (1-2 sentences) explaining your reasoning for this specific dimension.
Analyze the tweet carefully and provide your evaluation:"""

        return prompt

    def evaluate_single_tweet(self, tweet_text, tweet_id=None, apply_calibration=False):
        """Evaluate a single tweet across all dimensions using few-shot learning"""
        results = {}

        for dimension_key, dimension_info in self.evaluation_dimensions.items():
            # Select examples - either dimension-specific or general
            if self.use_dimension_specific:
                example_indices = self.select_dimension_specific_examples(
                    tweet_text,
                    target_tweet_id=tweet_id,
                    dimension_key=dimension_key,
                    k=self.k_shots
                )
            else:
                # Use same examples for all dimensions (original approach)
                example_indices = self.select_fewshot_examples(
                    tweet_text,
                    target_tweet_id=tweet_id,
                    k=self.k_shots,
                    exclude_same_text=True
                )

            prompt = self.create_fewshot_prompt(
                tweet_text,
                dimension_info["title"],
                dimension_info["questions"],
                example_indices
            )

            # Prepare messages for the model
            chat = [{"role": "user", "content": prompt}]

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
                max_length=4096  # Increased for few-shot examples
            ).to(self.model.device)

            # Generate response
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode response
            response = self.tokenizer.decode(
                generation[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

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
                    except:
                        evaluation = None

                # Strategy 1: Try to parse the entire response as JSON
                if evaluation is None:
                    try:
                        evaluation = json.loads(response)
                    except:
                        pass

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

                    # Try each candidate
                    for candidate in json_candidates:
                        try:
                            evaluation = json.loads(candidate)
                            break
                        except:
                            continue

                # Strategy 3: Simple regex fallback
                if evaluation is None:
                    json_match = re.search(r'\{[^}]+\}', response)
                    if json_match:
                        evaluation = json.loads(json_match.group())

                if not evaluation:
                    raise ValueError("Could not extract valid JSON from response")

                if "scores" in evaluation and isinstance(evaluation["scores"], list):
                    scores = []
                    for score in evaluation["scores"]:
                        try:
                            s = int(score)
                            scores.append(max(1, min(7, s)))  # Clamp to 1-7
                        except:
                            scores.append(4)

                    # Apply calibration if requested and available
                    if apply_calibration:
                        scores = self.calibrate_scores(scores[:3], dimension_key)
                    else:
                        scores = scores[:3]

                    results[dimension_key] = {
                        "scores": scores,
                        "average": sum(scores) / len(scores) if scores else 4.0,
                        "comment": evaluation.get("comment", f"Evaluated {dimension_key}"),
                        "example_indices": example_indices,  # Track which examples were used
                        "calibrated": apply_calibration
                    }
                else:
                    print(f"Warning: Invalid scores for {dimension_key}, using defaults")
                    results[dimension_key] = {
                        "scores": [4, 4, 4],
                        "average": 4.0,
                        "comment": "Unable to evaluate - using neutral scores",
                        "error": "Could not parse scores",
                        "example_indices": example_indices
                    }

            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Could not parse response for {dimension_key}: {str(e)}")
                results[dimension_key] = {
                    "scores": [4, 4, 4],
                    "average": 4.0,
                    "comment": f"Error in evaluation: {str(e)[:100]}",
                    "error": str(e),
                    "raw_response": response[:200] if response else "",
                    "example_indices": example_indices
                }

        return results

    def evaluate_dataset(self, csv_path, output_path=None, max_retries=2, apply_calibration=False):
        """Evaluate all tweets in the dataset with retry mechanism"""
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} tweets from {csv_path}")

        # Prepare output
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            calib_suffix = "_calibrated" if apply_calibration else ""
            output_path = f"gemma_evaluations_fewshot_k{self.k_shots}{calib_suffix}_{timestamp}.json"

        evaluations = []
        failed_tweets = []

        # Process each tweet
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating tweets"):
            tweet_id = row['Tweet ID']
            tweet_text = row['Tweet Text']

            print(f"\nEvaluating Tweet {idx+1}/{len(df)}: {tweet_text[:100]}...")

            try:
                evaluation = self.evaluate_single_tweet(tweet_text, tweet_id=tweet_id, apply_calibration=apply_calibration)

                has_error = any(
                    "error" in dim_eval
                    for dim_eval in evaluation.values()
                )

                result = {
                    "tweet_id": tweet_id,
                    "tweet_text": tweet_text,
                    "source_dataset": row.get('source_dataset', 'unknown'),
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                    "has_error": has_error,
                    "k_shots": self.k_shots
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
                    "has_error": True,
                    "k_shots": self.k_shots
                }
                evaluations.append(failed_result)
                failed_tweets.append((row, failed_result))

        # Retry logic (similar to zero-shot version)
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
                        evaluation = self.evaluate_single_tweet(tweet_text, tweet_id=tweet_id, apply_calibration=apply_calibration)

                        has_error = any(
                            "error" in dim_eval
                            for dim_eval in evaluation.values()
                        )

                        if not has_error:
                            for i, eval_item in enumerate(evaluations):
                                if eval_item["tweet_id"] == tweet_id:
                                    evaluations[i] = {
                                        "tweet_id": tweet_id,
                                        "tweet_text": tweet_text,
                                        "source_dataset": row_data.get('source_dataset', 'unknown'),
                                        "evaluation": evaluation,
                                        "timestamp": datetime.now().isoformat(),
                                        "has_error": False,
                                        "retry_count": retry_num,
                                        "k_shots": self.k_shots
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

        # Also save as CSV
        csv_output = output_path.replace('.json', '.csv')
        self.save_as_csv(evaluations, csv_output)

        return evaluations

    def save_as_csv(self, evaluations, csv_path):
        """Convert evaluations to CSV format for easier analysis"""
        rows = []

        for eval_item in evaluations:
            if "error" in eval_item and eval_item["error"]:
                continue

            row = {
                "tweet_id": eval_item["tweet_id"],
                "tweet_text": eval_item["tweet_text"],
                "source_dataset": eval_item.get("source_dataset", ""),
                "timestamp": eval_item.get("timestamp", ""),
                "k_shots": eval_item.get("k_shots", self.k_shots)
            }

            # Extract scores for each dimension
            for dimension in ["I-Involvement", "You-Involvement", "We-Involvement"]:
                if dimension in eval_item["evaluation"]:
                    dim_eval = eval_item["evaluation"][dimension]
                    if "scores" in dim_eval and isinstance(dim_eval["scores"], list):
                        scores = dim_eval["scores"]
                        for i in range(3):
                            if i < len(scores):
                                row[f"{dimension}_Q{i+1}"] = scores[i]
                            else:
                                row[f"{dimension}_Q{i+1}"] = 4

                        if "average" in dim_eval:
                            row[f"{dimension}_avg"] = dim_eval["average"]
                        else:
                            row[f"{dimension}_avg"] = sum(scores[:3]) / min(len(scores), 3) if scores else 4.0

                        if "comment" in dim_eval:
                            row[f"{dimension}_comment"] = dim_eval["comment"]
                    else:
                        for i in range(1, 4):
                            row[f"{dimension}_Q{i}"] = 4
                        row[f"{dimension}_avg"] = 4.0

            rows.append(row)

        if rows:
            df_results = pd.DataFrame(rows)

            # Reorder columns
            column_order = ["tweet_id", "tweet_text", "source_dataset", "k_shots"]

            for dim in ["I-Involvement", "You-Involvement", "We-Involvement"]:
                for i in range(1, 4):
                    column_order.append(f"{dim}_Q{i}")
                column_order.append(f"{dim}_avg")
                if f"{dim}_comment" in df_results.columns:
                    column_order.append(f"{dim}_comment")

            column_order.append("timestamp")

            column_order = [col for col in column_order if col in df_results.columns]
            df_results = df_results[column_order]

            df_results.to_csv(csv_path, index=False)
            print(f"CSV results saved to: {csv_path}")
        else:
            print("No valid evaluations to save to CSV")

    def create_train_val_split(self, val_ratio=0.2, random_seed=42):
        """
        Split MTurk data into training (for few-shot examples) and validation sets

        Args:
            val_ratio: Ratio of validation set
            random_seed: Random seed for reproducibility

        Returns:
            train_df, val_df
        """
        np.random.seed(random_seed)
        n = len(self.mturk_df)
        indices = np.random.permutation(n)

        val_size = int(n * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        val_df = self.mturk_df.iloc[val_indices].reset_index(drop=True)
        train_df = self.mturk_df.iloc[train_indices].reset_index(drop=True)

        print(f"Train/Val split: {len(train_df)} train, {len(val_df)} validation")

        return train_df, val_df

    def evaluate_on_validation(self, val_df):
        """
        Evaluate model on validation set and compare with human annotations

        Args:
            val_df: Validation dataframe with human annotations

        Returns:
            List of comparison results
        """
        validation_results = []

        print(f"\nEvaluating on {len(val_df)} validation examples...")

        for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validation"):
            tweet_text = row['Tweet Text']
            tweet_id = row.get('Tweet ID', None)

            # Get model predictions
            model_eval = self.evaluate_single_tweet(tweet_text, tweet_id=tweet_id, apply_calibration=False)

            # Extract human scores
            human_scores = {}
            model_scores = {}

            for dimension_key, dimension_info in self.evaluation_dimensions.items():
                rating_cols = dimension_info['rating_cols']

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
            def score_to_level(score):
                if score <= 3:
                    return 'low'
                elif score == 4:
                    return 'middle'
                else:  # 5-7
                    return 'high'

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

    def grid_search_optimization(self, val_df, param_grid=None, val_sample_size=None):
        """
        Grid search to find optimal hyperparameters

        Args:
            val_df: Validation dataframe
            param_grid: Dict of parameters to search
            val_sample_size: Number of validation samples to use (None = all)

        Returns:
            best_params, all_results
        """
        if param_grid is None:
            param_grid = {
                'k_shots': [2, 4, 8],
                'temperature': [0.0, 0.1, 0.3],
                'use_dimension_specific': [False, True]
            }

        # Sample validation set if needed for faster search
        if val_sample_size and val_sample_size < len(val_df):
            val_sample = val_df.sample(n=val_sample_size, random_state=42)
            print(f"Using {val_sample_size} validation samples for grid search")
        else:
            val_sample = val_df
            print(f"Using all {len(val_df)} validation samples for grid search")

        print("\n" + "="*70)
        print("GRID SEARCH OPTIMIZATION")
        print("="*70)
        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        print()

        results = []
        best_score = float('inf')  # We want to minimize MAE
        best_params = None

        # Generate all combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        total_combinations = np.prod([len(v) for v in param_values])
        print(f"Total combinations to try: {total_combinations}\n")

        for i, param_combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, param_combination))

            print(f"\n[{i+1}/{total_combinations}] Testing: {params}")

            # Temporarily update parameters
            original_k = self.k_shots
            original_temp = self.temperature
            original_dim_specific = self.use_dimension_specific

            self.k_shots = params.get('k_shots', self.k_shots)
            self.temperature = params.get('temperature', self.temperature)
            self.use_dimension_specific = params.get('use_dimension_specific', self.use_dimension_specific)

            # Also update top_p if in grid
            if 'top_p' in params:
                original_top_p = self.top_p
                self.top_p = params['top_p']

            try:
                # Evaluate on validation set
                val_results = self.evaluate_on_validation(val_sample)

                # Compute metrics
                metrics = self.compute_metrics(val_results)

                # Calculate average MAE (our primary metric)
                avg_mae = np.mean([m['mae'] for m in metrics.values()])
                avg_pearson = np.mean([m['pearson_r'] for m in metrics.values()])
                avg_within_1 = np.mean([m['within_1_acc'] for m in metrics.values()])

                print(f"  Results: MAE={avg_mae:.3f}, Pearson={avg_pearson:.3f}, Within-1={avg_within_1:.1%}")

                # Track results
                result_entry = {
                    'params': params.copy(),
                    'avg_mae': avg_mae,
                    'avg_pearson': avg_pearson,
                    'avg_within_1': avg_within_1,
                    'metrics': metrics
                }
                results.append(result_entry)

                # Update best
                if avg_mae < best_score:
                    best_score = avg_mae
                    best_params = params.copy()
                    print(f"  ✨ New best MAE: {best_score:.3f}")

            except Exception as e:
                print(f"  ❌ Error: {str(e)}")

            # Restore original parameters
            self.k_shots = original_k
            self.temperature = original_temp
            self.use_dimension_specific = original_dim_specific
            if 'top_p' in params:
                self.top_p = original_top_p

        # Print summary
        print("\n" + "="*70)
        print("GRID SEARCH COMPLETE")
        print("="*70)
        print(f"Best parameters: {best_params}")
        print(f"Best MAE: {best_score:.3f}")
        print()

        # Sort results by MAE
        results.sort(key=lambda x: x['avg_mae'])

        print("Top 5 configurations:")
        for i, r in enumerate(results[:5], 1):
            print(f"{i}. {r['params']}")
            print(f"   MAE={r['avg_mae']:.3f}, Pearson={r['avg_pearson']:.3f}, Within-1={r['avg_within_1']:.1%}")

        print("="*70 + "\n")

        return best_params, results


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Few-shot tweet evaluation with Gemma")
    parser.add_argument(
        "--mode",
        type=str,
        default="evaluate",
        choices=["evaluate", "optimize", "calibrate"],
        help="Mode: 'evaluate' (default), 'optimize' (grid search), or 'calibrate' (fit calibration)"
    )
    parser.add_argument(
        "--k-shots",
        type=int,
        default=4,
        choices=[0, 1, 2, 4, 8, 16],
        help="Number of few-shot examples to use (0 for zero-shot, default: 4)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter (default: 1.0)"
    )
    parser.add_argument(
        "--dimension-specific",
        action="store_true",
        help="Use dimension-specific example selection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemma model to use (default: from config)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help="GPU to use: 'auto' (default, select automatically), or specific GPU ID like '0', '1', etc."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/targeted_tweets.csv",
        help="Input CSV file with tweets to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: auto-generated)"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=50,
        help="Validation set size for optimization/calibration (default: 50)"
    )
    parser.add_argument(
        "--apply-calibration",
        action="store_true",
        help="Apply calibration to predictions (must run --mode calibrate first)"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = FewShotTweetEvaluator(
        model_name=args.model,
        k_shots=args.k_shots,
        temperature=args.temperature,
        top_p=args.top_p,
        use_dimension_specific=args.dimension_specific,
        gpu_id=args.gpu
    )

    if args.mode == "optimize":
        # Grid search optimization mode
        print("="*70)
        print("OPTIMIZATION MODE")
        print("="*70)

        # Create train/val split from MTurk data
        train_df, val_df = evaluator.create_train_val_split(val_ratio=0.2)

        # Use only subset for faster optimization
        val_subset = val_df.head(args.val_size) if args.val_size < len(val_df) else val_df

        # Run grid search
        best_params, results = evaluator.grid_search_optimization(
            val_subset,
            param_grid={
                'k_shots': [2, 4, 8],
                'temperature': [0.0, 0.1, 0.3],
                'use_dimension_specific': [False, True]
            }
        )

        # Save results
        results_file = "few_shot/data/grid_search_results.json"
        with open(results_file, 'w') as f:
            # Convert to JSON-serializable format
            json_results = []
            for r in results:
                json_results.append({
                    'params': r['params'],
                    'avg_mae': r['avg_mae'],
                    'avg_pearson': r['avg_pearson'],
                    'avg_within_1': r['avg_within_1']
                })
            json.dump({
                'best_params': best_params,
                'all_results': json_results
            }, f, indent=2)

        print(f"Results saved to: {results_file}")
        print(f"\nRecommended command:")
        print(f"python {__file__} --mode evaluate \\")
        print(f"  --k-shots {best_params['k_shots']} \\")
        print(f"  --temperature {best_params['temperature']} \\")
        if best_params.get('use_dimension_specific', False):
            print(f"  --dimension-specific \\")
        print(f"  --input {args.input}")

    elif args.mode == "calibrate":
        # Calibration mode
        print("="*70)
        print("CALIBRATION MODE")
        print("="*70)

        # Create train/val split
        train_df, val_df = evaluator.create_train_val_split(val_ratio=0.2)

        # Update MTurk data to only use training set for few-shot examples
        evaluator.mturk_df = train_df
        evaluator.mturk_tweets = train_df['Tweet Text'].tolist()
        evaluator.mturk_embeddings = evaluator.embedding_model.encode(
            evaluator.mturk_tweets,
            show_progress_bar=True,
            convert_to_tensor=True
        )

        # Use subset for calibration
        val_subset = val_df.head(args.val_size) if args.val_size < len(val_df) else val_df

        # Evaluate on validation set
        val_results = evaluator.evaluate_on_validation(val_subset)

        # Fit calibration
        evaluator.fit_calibration(val_results)

        # Save calibration parameters
        calib_file = "few_shot/data/calibration_params.json"
        with open(calib_file, 'w') as f:
            json.dump(evaluator.calibration_params, f, indent=2)

        print(f"Calibration parameters saved to: {calib_file}")

        # Evaluate with calibration
        print("\nEvaluating with calibration...")
        val_results_calibrated = []
        for result in val_results:
            model_scores_cal = {}
            for dim_key in result['model_scores']:
                model_scores_cal[dim_key] = evaluator.calibrate_scores(
                    result['model_scores'][dim_key],
                    dim_key
                )
            val_results_calibrated.append({
                'tweet_id': result['tweet_id'],
                'tweet_text': result['tweet_text'],
                'human_scores': result['human_scores'],
                'model_scores': model_scores_cal
            })

        # Compute metrics with calibration
        metrics_calibrated = evaluator.compute_metrics(val_results_calibrated)
        print("\nMETRICS AFTER CALIBRATION:")
        evaluator.print_metrics(metrics_calibrated)

    else:
        # Standard evaluation mode
        print("="*70)
        print("EVALUATION MODE")
        print("="*70)
        print(f"Parameters:")
        print(f"  k_shots: {args.k_shots}")
        print(f"  temperature: {args.temperature}")
        print(f"  top_p: {args.top_p}")
        print(f"  dimension_specific: {args.dimension_specific}")
        print(f"  apply_calibration: {args.apply_calibration}")
        print()

        # Load calibration if requested
        if args.apply_calibration:
            calib_file = "few_shot/data/calibration_params.json"
            try:
                with open(calib_file, 'r') as f:
                    evaluator.calibration_params = json.load(f)
                print(f"Loaded calibration parameters from: {calib_file}\n")
            except FileNotFoundError:
                print(f"Warning: Calibration file not found at {calib_file}")
                print("Run with --mode calibrate first to generate calibration parameters\n")
                args.apply_calibration = False

        # Run evaluation
        evaluator.evaluate_dataset(
            csv_path=args.input,
            output_path=args.output,
            max_retries=2,
            apply_calibration=args.apply_calibration
        )


if __name__ == "__main__":
    main()
