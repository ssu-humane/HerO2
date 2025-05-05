#!/bin/bash

# System configuration
SYSTEM_NAME="shero"  # Change this to "HerO", "Baseline", etc.
SPLIT="test_2025"  # Change this to "dev", or "test"
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
python shero/hyde_fc_generation.py \
    --target_data "${DATA_STORE}/averitec/${SPLIT}.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" \
    --model "meta-llama/Llama-3.1-8B-Instruct" || exit 1

python shero/retrieval.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" \
    --embedding_store_dir "${KNOWLEDGE_STORE}/test_2025_embed" \
    --knowledge_store_dir "${KNOWLEDGE_STORE}/test_2025_summary" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
    --model "Alibaba-NLP/gte-base-en-v1.5" --top_k 10 || exit 1

python shero/question_generation.py \
    --reference_corpus "${DATA_STORE}/averitec/train_dev.json" \
    --top_k_target_knowledge "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
    --output_questions "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
    --model "Qwen/Qwen3-8B" || exit 1

python shero/answer_rewriting.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa_rewrite.json" \
    --model "Qwen/Qwen3-8B" || exit 1

python shero/veracity_prediction.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa_rewrite.json" \
    --output_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
    --model "huggingface_cache/Qwen3-32B-HerO-bs-awq4bit" || exit 1

python prepare_leaderboard_submission.py --filename "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" || exit 1

python averitec_evaluate.py \
    --prediction_file "leaderboard_submission/submission.csv" \
    --label_file "leaderboard_submission/solution_test_2025.csv" || exit 1