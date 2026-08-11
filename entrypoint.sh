#!/bin/bash
set -e

echo "Checking Headroom model cache..."

# Ensure HF_HOME is set (inherited from environment)
export HF_HOME=${HF_HOME:-/data/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/models}

# Ensure HEADROOM_CCR_STORE_PATH is set for persistent CCR storage
export HEADROOM_CCR_STORE_PATH=${HEADROOM_CCR_STORE_PATH:-/data/headroom_ccr}

# Create cache directories if they don't exist (volumes may be empty)
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HEADROOM_CCR_STORE_PATH"

# Download Kompress model if not present
if [ ! -d "$HUGGINGFACE_HUB_CACHE/models--chopratejas--kompress-v2-base" ]; then
    echo "Downloading chopratejas/kompress-v2-base..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='chopratejas/kompress-v2-base', local_dir='$HUGGINGFACE_HUB_CACHE/models--chopratejas--kompress-v2-base')
print('Kompress model downloaded successfully.')
"
else
    echo "Kompress model already cached."
fi

# Download Qwen tokenizer if not present
if [ ! -d "$TRANSFORMERS_CACHE/models--Qwen--Qwen-7B" ]; then
    echo "Downloading Qwen/Qwen-7B tokenizer..."
    python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True)
print('Qwen tokenizer downloaded successfully.')
"
else
    echo "Qwen tokenizer already cached."
fi

# Download sentence-transformers model if not present
if [ ! -d "$TRANSFORMERS_CACHE/models--all-MiniLM-L6-v2" ]; then
    echo "Downloading all-MiniLM-L6-v2..."
    python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
print('Sentence-transformers model downloaded successfully.')
"
else
    echo "Sentence-transformers model already cached."
fi

echo "Cache check complete. Starting Uvicorn..."
exec uvicorn app:app --host 0.0.0.0 --port ${ROUTER_INTERNAL_PORT:-4000}
