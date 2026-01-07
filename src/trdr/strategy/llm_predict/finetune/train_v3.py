"""Submit fine-tuning job to Vertex AI.

Prerequisites:
1. Google Cloud project with Vertex AI enabled
2. GCS bucket with training data uploaded
3. GOOGLE_CLOUD_PROJECT env var set
4. Authenticated via: gcloud auth application-default login

Usage:
    python -m trdr.strategy.llm_predict.finetune.train_v3
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent.parent.parent / ".env")

# Config - UPDATE THESE
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = "us-central1"
GCS_BUCKET = os.environ.get("GCS_BUCKET", "your-bucket-name")

# Model config
SOURCE_MODEL = "gemini-2.5-flash-lite"  # Base model for SFT
TRAIN_DATA_URI = f"gs://{GCS_BUCKET}/finetune/train_v3.jsonl"
VAL_DATA_URI = f"gs://{GCS_BUCKET}/finetune/val_v3.jsonl"

# Training config
TUNED_MODEL_DISPLAY_NAME = "eth-direction-predictor-sax"
EPOCHS = 3  # 1-10, start low


def check_config():
    """Verify configuration."""
    print("=" * 60)
    print("CONFIGURATION CHECK")
    print("=" * 60)

    issues = []

    if PROJECT_ID == "your-project-id":
        issues.append("Set GOOGLE_CLOUD_PROJECT env var")

    if GCS_BUCKET == "your-bucket-name":
        issues.append("Set GCS_BUCKET env var")

    print(f"Project ID:    {PROJECT_ID}")
    print(f"Location:      {LOCATION}")
    print(f"GCS Bucket:    {GCS_BUCKET}")
    print(f"Source Model:  {SOURCE_MODEL}")
    print(f"Train Data:    {TRAIN_DATA_URI}")
    print(f"Val Data:      {VAL_DATA_URI}")
    print(f"Epochs:        {EPOCHS}")

    if issues:
        print("\n[ERRORS]")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True


def upload_data():
    """Upload local data to GCS."""
    import subprocess

    data_dir = Path(__file__).parent / "data"
    train_path = data_dir / "train_v3.jsonl"
    val_path = data_dir / "val_v3.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run generate_data_v3.py first.")
        return False

    print("\nUploading data to GCS...")

    # Upload train
    cmd = f"gsutil cp {train_path} {TRAIN_DATA_URI}"
    print(f"  {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False

    # Upload val
    cmd = f"gsutil cp {val_path} {VAL_DATA_URI}"
    print(f"  {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False

    print("  Upload complete.")
    return True


def start_tuning_job():
    """Start the fine-tuning job."""
    import vertexai
    from vertexai.tuning import sft

    print("\n" + "=" * 60)
    print("STARTING FINE-TUNING JOB")
    print("=" * 60)

    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Start tuning job
    print(f"\nSubmitting tuning job...")
    print(f"  Source model: {SOURCE_MODEL}")
    print(f"  Train data:   {TRAIN_DATA_URI}")
    print(f"  Val data:     {VAL_DATA_URI}")

    tuning_job = sft.train(
        source_model=SOURCE_MODEL,
        train_dataset=TRAIN_DATA_URI,
        validation_dataset=VAL_DATA_URI,
        tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME,
        epochs=EPOCHS,
    )

    print(f"\nJob submitted!")
    print(f"  Job name: {tuning_job.name}")

    # Poll for completion
    print("\nPolling for completion (this may take 30-60 minutes)...")
    poll_count = 0
    while not tuning_job.has_ended:
        poll_count += 1
        print(f"  [{poll_count}] Job still running... (checking every 60s)")
        time.sleep(60)
        tuning_job.refresh()

    # Results
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)

    if tuning_job.tuned_model_name:
        print(f"\nTuned model name:     {tuning_job.tuned_model_name}")
        print(f"Tuned model endpoint: {tuning_job.tuned_model_endpoint_name}")

        # Save for later use
        config_path = Path(__file__).parent / "tuned_model_v3.txt"
        with open(config_path, "w") as f:
            f.write(f"MODEL_NAME={tuning_job.tuned_model_name}\n")
            f.write(f"ENDPOINT_NAME={tuning_job.tuned_model_endpoint_name}\n")
        print(f"\nSaved config to {config_path}")
    else:
        print("\nERROR: Tuning failed. Check Vertex AI console for details.")

    return tuning_job


def main():
    # Check config
    if not check_config():
        print("\nFix configuration errors and re-run.")
        return

    # Check if data exists locally
    data_dir = Path(__file__).parent / "data"
    if not (data_dir / "train_v3.jsonl").exists():
        print("\nNo training data found. Generate it first:")
        print("  python -m trdr.strategy.llm_predict.finetune.generate_data_v3")
        return

    # Upload data
    if not upload_data():
        print("\nFailed to upload data. Check GCS permissions.")
        return

    # Start tuning
    try:
        start_tuning_job()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you're authenticated: gcloud auth application-default login")
        print("  2. Ensure Vertex AI API is enabled")
        print("  3. Ensure you have the vertexai package: pip install google-cloud-aiplatform")


if __name__ == "__main__":
    main()
