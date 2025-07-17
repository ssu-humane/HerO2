#!/bin/bash

# System configuration
SPLIT="dev"  # Change this to "dev", or "test"
BASE_DIR="."  # Current directory

DATA_STORE="${BASE_DIR}/data_store"
KNOWLEDGE_STORE="${BASE_DIR}/knowledge_store"
EMBEDDING_STORE="${KNOWLEDGE_STORE}/${SPLIT}_embed"
SUMMARY_STORE="${KNOWLEDGE_STORE}/${SPLIT}_summary"
export HF_HOME="${BASE_DIR}/huggingface_cache"

mkdir -p "${EMBEDDING_STORE}"
mkdir -p "${SUMMARY_STORE}"
mkdir -p "${HF_HOME}"

export HUGGING_FACE_HUB_TOKEN="[YOUR TOKEN]"

python hero2/embed_script.py \
  --model_name Alibaba-NLP/gte-base-en-v1.5 \
  --knowledge_dir "${KNOWLEDGE_STORE}/${SPLIT}" \
  --output_dir "${EMBEDDING_STORE}" \
  --num_files 500 \
  --batch_size 32


python hero2/document_summary.py \
  --model Qwen/Qwen3-8B \
  --knowledge_store "${KNOWLEDGE_STORE}" \
  --save_path "${SUMMARY_STORE}" \
  --max_input_token_size 24000 \
  --num_files 500