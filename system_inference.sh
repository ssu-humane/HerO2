#!/bin/bash

# System configuration
SYSTEM_NAME="hero2"  # Change this to "HerO", "Baseline", etc.
SPLIT="dev"  # Change this to "dev", or "test"
BASE_DIR="."  # Current directory

DATA_STORE="${BASE_DIR}/data_store"
KNOWLEDGE_STORE="${BASE_DIR}/knowledge_store"
export HF_HOME="${BASE_DIR}/huggingface_cache"

# Create necessary directories
mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${DATA_STORE}/${SYSTEM_NAME}"
mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"
mkdir -p "${HF_HOME}"

export HUGGING_FACE_HUB_TOKEN="[YOUR TOKEN]"

# Execute each script from src directory
python hero2/hyde_fc_generation.py \
    --target_data "${DATA_STORE}/averitec/${SPLIT}.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" \
    --model "meta-llama/Llama-3.1-8B-Instruct" || exit 1

python hero2/retrieval.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" \
    --embedding_store_dir "${KNOWLEDGE_STORE}/${SPLIT}_embed" \
    --knowledge_store_dir "${KNOWLEDGE_STORE}/${SPLIT}_summary" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
    --model "Alibaba-NLP/gte-base-en-v1.5" --top_k 10 || exit 1

python hero2/question_generation.py \
    --reference_corpus "${DATA_STORE}/averitec/train.json" \
    --top_k_target_knowledge "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
    --output_questions "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
    --model "Qwen/Qwen3-8B" || exit 1

python hero2/answer_rewriting.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa_rewrite.json" \
    --model "Qwen/Qwen3-8B" || exit 1

python hero2/veracity_prediction.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa_rewrite.json" \
    --output_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
    --model "humane-lab/Qwen3-32B-AWQ-HerO" || exit 1