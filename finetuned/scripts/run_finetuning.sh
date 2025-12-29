#!/bin/bash
# Run fine-tuning with QLoRA on Gemma-2-9b-it

echo "Starting fine-tuning..."
echo "================================"
echo "Model: google/gemma-2-9b-it"
echo "Method: QLoRA (4-bit quantization)"
echo "Training examples: 2604"
echo "Validation examples: 558"
echo "================================"
echo ""

python finetuned/src/finetune_gemma.py \
    --model-name google/gemma-2-9b-it \
    --output-dir finetuned/models/gemma-2-9b-tweet-eval \
    --train-file finetuned/data/train.jsonl \
    --val-file finetuned/data/val.jsonl \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --weight-decay 0.01 \
    --seed 42

echo ""
echo "================================"
echo "Fine-tuning complete!"
echo "Model saved to: finetuned/models/gemma-2-9b-tweet-eval/final"
echo "================================"
