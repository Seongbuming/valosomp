#!/usr/bin/env python3
"""
Fine-tune Gemma-2-9b-it model using QLoRA for tweet evaluation task.
Uses PEFT (Parameter-Efficient Fine-Tuning) for memory efficiency.
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import argparse

def load_jsonl_dataset(file_path):
    """Load dataset from JSONL file"""
    return load_dataset('json', data_files=file_path, split='train')

def format_instruction_response(example):
    """
    Format example as instruction-response pair for Gemma chat template
    """
    messages = [
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['response']}
    ]
    return {"messages": messages}

def main(args):
    print("="*70)
    print("GEMMA-2-9B-IT FINE-TUNING WITH QLORA")
    print("="*70)
    print(f"Base model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU: {args.gpu}")
    print()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Set GPU
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU {args.gpu}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16

    # QLoRA config (4-bit quantization)
    print("Configuring QLoRA (4-bit quantization)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with quantization
    print("Loading base model with 4-bit quantization...")
    print("Estimated VRAM: ~6-8GB")

    # Get HuggingFace token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment")
        hf_token = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load datasets
    print()
    print("Loading datasets...")
    train_dataset = load_jsonl_dataset(args.train_file)
    val_dataset = load_jsonl_dataset(args.val_file)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Format datasets
    print("Formatting datasets...")
    train_dataset = train_dataset.map(format_instruction_response)
    val_dataset = val_dataset.map(format_instruction_response)

    # Training configuration
    print()
    print("Training configuration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print()

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        report_to="none",  # Disable wandb/tensorboard
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        dataloader_num_workers=4,
        # SFT-specific parameters
        packing=False,
        max_length=2048,
        dataset_text_field="messages"
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer
    )

    # Train
    print()
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    trainer.train()

    # Save final model
    print()
    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    # Save training info
    training_info = {
        'base_model': args.model_name,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'train_examples': len(train_dataset),
        'val_examples': len(val_dataset),
        'lora_r': peft_config.r,
        'lora_alpha': peft_config.lora_alpha,
        'lora_dropout': peft_config.lora_dropout
    }

    info_path = os.path.join(args.output_dir, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"Training info saved to {info_path}")
    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model saved to: {os.path.join(args.output_dir, 'final')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Gemma model with QLoRA")

    # Model args
    parser.add_argument(
        "--model-name", type=str,
        default="google/gemma-2-9b-it",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="finetuned/models/gemma-2-9b-tweet-eval",
        help="Output directory for fine-tuned model"
    )

    # Data args
    parser.add_argument(
        "--train-file", type=str,
        default="finetuned/data/train.jsonl",
        help="Training data file"
    )
    parser.add_argument(
        "--val-file", type=str,
        default="finetuned/data/val.jsonl",
        help="Validation data file"
    )

    # Training args
    parser.add_argument(
        "--num-epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU ID to use (e.g., '0' or '1')"
    )

    args = parser.parse_args()

    main(args)
