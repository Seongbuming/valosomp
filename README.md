# Decoding I, You, and We in Crisis Communication: A Three-Phase LLM-Based Validation of Self-Oriented Model of Digital Publics (SOMP)

## Authors

- Dr. Hyelim Lee
- Seongbum Seo
- Somin Park
- Dr. Lisa Tam
- Dr. Soojin Kim
- Dr. Yun Jang

## Repository Structure

### `/data`
Survey response data and crisis-related tweet datasets. Contains anonymized worker responses, filtered tweet collections, and language detection results.

### `/zero_shot`
Zero-shot classification experiments using LLMs to evaluate I-, You-, and We-involvement in crisis communication without training examples.

### `/few_shot`
Few-shot classification experiments (k=1, 4, 8, 16) testing LLM performance with varying numbers of demonstration examples.

### `/finetuned`
Fine-tuning experiments with LoRA adapters for tweet evaluation. Includes training data, model checkpoints, and evaluation scripts.

### `/survey`
Survey analysis scripts and outputs, including statistical tests and visualization generation.

### `/scripts`
Data processing and analysis utilities for deduplication, aggregation methods comparison, and accuracy measurement.

### `/pre-survey`
Preliminary survey materials and pilot study data.

## Data Anonymization

All personally identifiable information has been anonymized:
- Worker IDs replaced with ANONYMOUS### format
- Email addresses replaced with email###@anonymous.com format

## Requirements

See `requirements.txt` for Python dependencies.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
