# Gemini Fine-Tuning for Price Prediction

Fine-tune Gemini to predict ETH/USD direction from encoded price patterns.

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **Vertex AI API** enabled
3. **GCS bucket** for training data
4. **Authentication** configured

## Setup

```bash
# Install dependencies
pip install google-cloud-aiplatform

# Authenticate
gcloud auth application-default login

# Set environment variables (add to .env)
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GCS_BUCKET="your-bucket-name"
```

## Usage

## Versioned pipelines

- v1 (`generate_data_v1.py`, `train_v1.py`, `test_v1.py`): coordinate encoding + binary UP/DOWN labels (`data/train.jsonl`, `tuned_model_v1.txt`).
- v2 (`generate_data_v2.py`, `train_v2.py`, `test_v2.py`): coordinate encoding + engineered features + rich multi-target outputs (`data/train_v2.jsonl`, `tuned_model_v2.txt`).
- v3 (`generate_data_v3.py`, `train_v3.py`, `test_v3.py`): SAX encoding + triple-barrier labels (UP/DOWN/same) aligned to best ICL config (`data/train_v3.jsonl`, `tuned_model_v3.txt`).

### Step 1: Generate Training Data

```bash
python -m trdr.strategy.llm_predict.finetune.generate_data_v1
```

Creates:

- `data/train.jsonl` - 1000 training examples
- `data/val.jsonl` - 200 validation examples

### Step 2: Fine-Tune Model

```bash
python -m trdr.strategy.llm_predict.finetune.train_v1
```

This will:

1. Upload data to GCS
2. Submit tuning job to Vertex AI
3. Poll until complete (~30-60 minutes)
4. Save tuned model config to `tuned_model.txt`

### Step 3: Test Fine-Tuned Model

```bash
python -m trdr.strategy.llm_predict.finetune.test_v1
```

Runs 50 predictions on held-out test data.

## Cost Estimate

- Training: ~$2-4 per million tokens
- 1000 examples × ~100 tokens = 100K tokens
- Estimated cost: **< $1** for training

## Expected Results

- **Baseline (ICL)**: ~50% accuracy (random)
- **Fine-tuned**: TBD - may or may not improve

The hypothesis is that fine-tuning allows the model to learn patterns
that in-context learning cannot capture. However, if no predictive
signal exists in the data, fine-tuning won't help either.

## Troubleshooting

**"Permission denied" errors:**

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**"API not enabled" errors:**

```bash
gcloud services enable aiplatform.googleapis.com
```

**"Quota exceeded" errors:**

- Check Vertex AI quotas in Cloud Console
- Request quota increase if needed
