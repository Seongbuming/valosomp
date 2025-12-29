"""
Configuration for Gemma model selection
"""

# Available Gemma text models for tweet evaluation
GEMMA_MODELS = {
    "small": "google/gemma-2-2b-it",     # 2B params - Fast, good for testing
    "medium": "google/gemma-2-9b-it",    # 9B params - Better quality
    "large": "google/gemma-2-27b-it",    # 27B params - Best quality (requires more VRAM)
    "legacy_small": "google/gemma-1.1-2b-it",  # Older version, 2B
    "legacy_medium": "google/gemma-1.1-7b-it"  # Older version, 7B
}

# Default model selection
DEFAULT_MODEL = GEMMA_MODELS["medium"]  # Change to "medium" or "large" for better quality

# GPU Memory Requirements (approximate)
VRAM_REQUIREMENTS = {
    "google/gemma-2-2b-it": "~8GB",
    "google/gemma-2-9b-it": "~20GB",
    "google/gemma-2-27b-it": "~48GB",
    "google/gemma-1.1-2b-it": "~6GB",
    "google/gemma-1.1-7b-it": "~16GB"
}