#!/usr/bin/env python3
"""
Debug script to see what the fine-tuned model actually generates
"""
import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

EVALUATION_DIMENSIONS = {
    "I-Involvement": {
        "title": "I-Involvement Recognition (Individual)",
        "questions": [
            "The tweet expresses concern about the issue's impact on the self",
            "The tweet indicates that the issue has personal consequences for the speaker",
            "The tweet emphasizes individual experience or personal situation regarding the issue"
        ],
        "rating_cols": ["I-Involvement-1", "I-Involvement-2", "I-Involvement-3"]
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

def load_model(model_path, base_model_name="google/gemma-2-9b-it"):
    """Load model"""
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

    print("Model loaded!\n")
    return model, tokenizer

def test_model_output(model, tokenizer, tweet_text):
    """Test what the model generates"""

    dim_info = EVALUATION_DIMENSIONS["I-Involvement"]

    prompt = create_evaluation_prompt(
        tweet_text,
        dim_info['title'],
        dim_info['questions']
    )

    print("="*70)
    print("PROMPT:")
    print("="*70)
    print(prompt)
    print("\n")

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    print("="*70)
    print("FORMATTED PROMPT (with chat template):")
    print("="*70)
    print(formatted_prompt)
    print("\n")

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    print("="*70)
    print("GENERATING RESPONSE...")
    print("="*70)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    print("MODEL RESPONSE:")
    print("-"*70)
    print(response)
    print("-"*70)
    print("\n")

    # Try to parse - extract FIRST complete JSON object only
    print("="*70)
    print("PARSING ATTEMPT (FIXED):")
    print("="*70)
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

            print(f"JSON start: {json_start}, JSON end: {json_end}")

            if json_end > json_start:
                json_str = response[json_start:json_end]
                print(f"Extracted JSON string:\n{json_str}")

                result = json.loads(json_str)
                print(f"Parsed result: {result}")

                scores = result.get('scores', [])
                print(f"Scores: {scores}")

                if len(scores) == 3 and all(isinstance(s, int) and 1 <= s <= 7 for s in scores):
                    print(f"✓ Valid scores: {scores}")
                else:
                    print(f"✗ Invalid scores: {scores}")
            else:
                print("✗ No matching closing brace found")
        else:
            print("✗ No JSON found in response")
    except Exception as e:
        print(f"✗ Parsing error: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str,
        default="finetuned/models/gemma-2-9b-tweet-eval/final",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--tweet", type=str,
        default="Stay home, stay safe. We're all in this together during this pandemic.",
        help="Tweet text to test"
    )

    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    test_model_output(model, tokenizer, args.tweet)
