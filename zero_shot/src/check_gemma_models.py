#!/usr/bin/env python3
"""
Check available Gemma models and their types
"""

from huggingface_hub import list_models
from dotenv import load_dotenv
import os

load_dotenv()

# Get token
token = os.getenv("HUGGINGFACE_TOKEN")

print("=== Available Gemma Models ===\n")

# Search for Gemma models
models = list(list_models(search="gemma", author="google", token=token))

print(f"Found {len(models)} Gemma models from Google:\n")

# Filter and categorize models
text_models = []
multimodal_models = []
other_models = []

for model in models:
    model_id = model.modelId
    tags = model.tags if hasattr(model, 'tags') else []

    # Check model type based on tags and name
    if 'text-generation' in tags or 'text2text-generation' in tags:
        if 'gemma-3' in model_id.lower():
            multimodal_models.append(model_id)
        else:
            text_models.append(model_id)
    elif 'image-text-to-text' in tags or 'multimodal' in tags:
        multimodal_models.append(model_id)
    else:
        # Check by name patterns
        if 'gemma-3' in model_id.lower():
            multimodal_models.append(model_id)
        elif 'gemma' in model_id.lower() and '-it' in model_id:
            text_models.append(model_id)
        else:
            other_models.append(model_id)

print("📝 Text-Only Models (Good for tweet evaluation):")
for model in sorted(text_models):
    print(f"  - {model}")

print("\n🖼️ Multimodal Models (Gemma-3 series):")
for model in sorted(multimodal_models):
    print(f"  - {model}")

print("\n❓ Other Models:")
for model in sorted(other_models)[:10]:  # Show first 10
    print(f"  - {model}")

print("\n=== Recommended Models for Tweet Evaluation ===")
print("1. google/gemma-2-2b-it (2B params, fast)")
print("2. google/gemma-2-9b-it (9B params, better quality)")
print("3. google/gemma-2-27b-it (27B params, best quality)")
print("\nNote: Gemma-3 models are multimodal and require image inputs.")
print("For text-only tasks like tweet evaluation, use Gemma-2 series.")