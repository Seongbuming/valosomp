# Validation of Self-Oriented Model of Digital Publics (SOMP)

This repository contains data, code, and analytical procedures for the following paper:
Lee, H., Seo, S., Park, S., Tam, L., Kim, S., & Jang, Y. (2026).
[Theory-driven LLM-assisted analysis in crisis communication: A case study of the self-oriented model of digital publics](https://doi.org/10.1016/j.pubrev.2026.102709).
*Public Relations Review*, *52*(3), 102709.

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

## Citation

```bibtex
@article{lee2026theory,
  title = {Theory-driven LLM-assisted analysis in crisis communication: A case study of the self-oriented model of digital publics},
  journal = {Public Relations Review},
  volume = {52},
  number = {3},
  pages = {102709},
  year = {2026},
  issn = {0363-8111},
  doi = {https://doi.org/10.1016/j.pubrev.2026.102709},
  url = {https://www.sciencedirect.com/science/article/pii/S0363811126000445},
  author = {Hyelim Lee and Seongbum Seo and Somin Park and Lisa Tam and Soojin Kim and Yun Jang},
  keywords = {Large Language Models, Human–AI Integration, Public Relations Methods, Crisis Communication, Self-Oriented Model of Digital Publics}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
