#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./finetune_v3.sh <v1|v2|v3>"
  exit 1
fi

version="$1"
bucket="gs://grant-test-vertex-456/finetune"

case "$version" in
  v1)
    gen="generate_data_v1"
    train="train_v1"
    test="test_v1"
    data_glob="data/*.jsonl"
    ;;
  v2)
    gen="generate_data_v2"
    train="train_v2"
    test="test_v2"
    data_glob="data/*_v2.jsonl"
    ;;
  v3)
    gen="generate_data_v3"
    train="train_v3"
    test="test_v3"
    data_glob="data/*_v3.jsonl"
    ;;
  *)
    echo "Unknown version: $version (expected v1|v2|v3)"
    exit 1
    ;;
esac

python -m trdr.strategy.llm_predict.finetune."$gen"
gsutil cp ./src/trdr/strategy/llm_predict/finetune/"$data_glob" "$bucket"
python -m trdr.strategy.llm_predict.finetune."$train"
python -m trdr.strategy.llm_predict.finetune."$test"
