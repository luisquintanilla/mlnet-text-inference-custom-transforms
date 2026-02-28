#!/bin/bash
# download-models.sh — Download HuggingFace models for all sample projects.
#
# Usage:
#   bash scripts/download-models.sh [task...]
#
# Tasks:
#   embeddings-core    all-MiniLM-L6-v2 + copies to shared samples (~86MB)
#   embeddings-all     all embedding models (MiniLM, BGE, E5, GTE, ~470MB)
#   classification     sentiment, emotion, zero-shot models (~1GB)
#   reranking          cross-encoder reranking models (~1.2GB)
#   ner                NER models via optimum-cli export (~500MB, needs Python)
#   qa                 QA models (~600MB)
#   textgen            Phi-3-mini for local generation (~2GB, needs huggingface-cli)
#   all                everything except textgen (~3.5GB)
#   all+textgen        everything including textgen (~5.5GB)
#
# Examples:
#   bash scripts/download-models.sh embeddings-core      # just the starter model
#   bash scripts/download-models.sh classification ner    # two tasks
#   bash scripts/download-models.sh all                   # everything (no textgen)

set -e

# ── helpers ───────────────────────────────────────────────────────────────

hf_download() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        echo "    ✓ $(basename "$dest") already exists, skipping"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo "    ↓ $(basename "$dest")"
    curl -fSL --progress-bar -o "$dest" "$url"
}

optimum_export() {
    local model_id="$1" dest_dir="$2"
    if [ -f "$dest_dir/model.onnx" ]; then
        echo "    ✓ model.onnx already exists, skipping"
        return
    fi
    if ! command -v optimum-cli &> /dev/null; then
        echo "    Installing optimum[exporters]..."
        pip install -q "optimum[exporters]"
    fi
    mkdir -p "$dest_dir"
    echo "    ↓ Exporting $model_id to ONNX..."
    optimum-cli export onnx --model "$model_id" "$dest_dir"
}

copy_if_missing() {
    local src="$1" dest="$2"
    if [ ! -f "$dest" ]; then
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
        echo "    ✓ Copied to $(dirname "$dest")"
    fi
}

# ── task functions ────────────────────────────────────────────────────────

download_embeddings_core() {
    echo ""
    echo "📦 Embeddings (core) — all-MiniLM-L6-v2 (~86MB)"
    echo "─────────────────────────────────────────────────"
    local base="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
    local dir="samples/BasicUsage/models"

    hf_download "$base/onnx/model.onnx" "$dir/model.onnx"
    hf_download "$base/vocab.txt" "$dir/vocab.txt"

    # Copy to samples that share this model
    for dest in samples/ComposablePoolingComparison samples/IntermediateInspection samples/MeaiProviderAgnostic; do
        copy_if_missing "$dir/model.onnx" "$dest/models/model.onnx"
        copy_if_missing "$dir/vocab.txt" "$dest/models/vocab.txt"
    done

    echo "  ✅ Ready: BasicUsage, ComposablePoolingComparison, IntermediateInspection, MeaiProviderAgnostic"
}

download_embeddings_extra() {
    echo ""
    echo "📦 Embeddings (extra) — BGE, E5, GTE (~380MB)"
    echo "───────────────────────────────────────────────"

    echo "  BGE-small-en-v1.5:"
    local bge="https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main"
    hf_download "$bge/onnx/model.onnx" "samples/BgeSmallEmbedding/models/model.onnx"
    hf_download "$bge/vocab.txt" "samples/BgeSmallEmbedding/models/vocab.txt"

    echo "  E5-small-v2:"
    local e5="https://huggingface.co/intfloat/e5-small-v2/resolve/main"
    hf_download "$e5/model.onnx" "samples/E5SmallEmbedding/models/model.onnx"
    hf_download "$e5/vocab.txt" "samples/E5SmallEmbedding/models/vocab.txt"

    echo "  GTE-small:"
    local gte="https://huggingface.co/thenlper/gte-small/resolve/main"
    hf_download "$gte/onnx/model.onnx" "samples/GteSmallEmbedding/models/model.onnx"
    hf_download "$gte/vocab.txt" "samples/GteSmallEmbedding/models/vocab.txt"

    echo "  ✅ Ready: BgeSmallEmbedding, E5SmallEmbedding, GteSmallEmbedding"
}

download_classification() {
    echo ""
    echo "📦 Classification — Sentiment, Emotion, Zero-shot (~1GB)"
    echo "─────────────────────────────────────────────────────────"

    echo "  DistilBERT-SST2 (sentiment):"
    local distilbert="https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main"
    hf_download "$distilbert/onnx/model.onnx" "samples/Classification/SentimentDistilBERT/models/model.onnx"
    hf_download "$distilbert/vocab.txt" "samples/Classification/SentimentDistilBERT/models/vocab.txt"

    echo "  RoBERTa-GoEmotions (emotion):"
    local roberta="https://huggingface.co/lquint/roberta-base-go_emotions-onnx/resolve/main"
    hf_download "$roberta/model.onnx" "samples/Classification/EmotionRoBERTa/models/model.onnx"
    hf_download "$roberta/vocab.json" "samples/Classification/EmotionRoBERTa/models/vocab.json"
    hf_download "$roberta/merges.txt" "samples/Classification/EmotionRoBERTa/models/merges.txt"
    hf_download "$roberta/tokenizer_config.json" "samples/Classification/EmotionRoBERTa/models/tokenizer_config.json"

    echo "  DeBERTa-v3-NLI (zero-shot):"
    local deberta="https://huggingface.co/lquint/DeBERTa-v3-base-mnli-fever-anli-onnx/resolve/main"
    hf_download "$deberta/model.onnx" "samples/Classification/ZeroShotDeBERTa/models/model.onnx"
    hf_download "$deberta/spm.model" "samples/Classification/ZeroShotDeBERTa/models/spm.model"
    hf_download "$deberta/tokenizer_config.json" "samples/Classification/ZeroShotDeBERTa/models/tokenizer_config.json"

    echo "  ✅ Ready: SentimentDistilBERT, EmotionRoBERTa, ZeroShotDeBERTa"
}

download_reranking() {
    echo ""
    echo "📦 Reranking — Cross-encoder models (~1.2GB)"
    echo "──────────────────────────────────────────────"

    echo "  MS MARCO MiniLM:"
    local marco="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main"
    hf_download "$marco/onnx/model.onnx" "samples/Reranking/MsMarcoMiniLM/models/model.onnx"
    hf_download "$marco/vocab.txt" "samples/Reranking/MsMarcoMiniLM/models/vocab.txt"
    hf_download "$marco/tokenizer_config.json" "samples/Reranking/MsMarcoMiniLM/models/tokenizer_config.json"

    echo "  BGE-reranker-base:"
    local bge="https://huggingface.co/BAAI/bge-reranker-base/resolve/main"
    hf_download "$bge/onnx/model.onnx" "samples/Reranking/BgeReranker/models/model.onnx"
    hf_download "$bge/vocab.txt" "samples/Reranking/BgeReranker/models/vocab.txt"
    hf_download "$bge/tokenizer_config.json" "samples/Reranking/BgeReranker/models/tokenizer_config.json"

    echo "  ✅ Ready: MsMarcoMiniLM, BgeReranker"
}

download_ner() {
    echo ""
    echo "📦 NER — Named Entity Recognition (~500MB, needs Python)"
    echo "──────────────────────────────────────────────────────────"

    echo "  BERT-base-NER:"
    optimum_export "dslim/bert-base-NER" "samples/NER/BertBaseNER/models"

    echo "  Multilingual NER:"
    optimum_export "Davlan/bert-base-multilingual-cased-ner-hrl" "samples/NER/MultilingualNER/models"

    echo "  ✅ Ready: BertBaseNER, MultilingualNER"
}

download_qa() {
    echo ""
    echo "📦 QA — Question Answering (~600MB)"
    echo "─────────────────────────────────────"

    echo "  RoBERTa-SQuAD2:"
    local roberta_qa="https://huggingface.co/lquint/roberta-base-squad2-onnx/resolve/main"
    hf_download "$roberta_qa/model.onnx" "samples/QA/RobertaSquad2/models/model.onnx"
    hf_download "$roberta_qa/vocab.json" "samples/QA/RobertaSquad2/models/vocab.json"
    hf_download "$roberta_qa/merges.txt" "samples/QA/RobertaSquad2/models/merges.txt"
    hf_download "$roberta_qa/tokenizer_config.json" "samples/QA/RobertaSquad2/models/tokenizer_config.json"

    echo "  MiniLM-SQuAD2:"
    local minilm_qa="https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main"
    hf_download "$minilm_qa/model.onnx" "samples/QA/MiniLMSquad2/models/model.onnx"
    hf_download "$minilm_qa/vocab.txt" "samples/QA/MiniLMSquad2/models/vocab.txt"
    hf_download "$minilm_qa/tokenizer_config.json" "samples/QA/MiniLMSquad2/models/tokenizer_config.json"

    echo "  ✅ Ready: RobertaSquad2, MiniLMSquad2"
}

download_textgen() {
    echo ""
    echo "📦 Text Generation — Phi-3-mini ONNX GenAI (~2GB)"
    echo "───────────────────────────────────────────────────"

    local dir="samples/TextGenerationLocal/models/phi-3-mini"
    if [ -d "$dir" ] && [ -f "$dir/model.onnx" -o -f "$dir/genai_config.json" ]; then
        echo "    ✓ Model directory already exists, skipping"
    else
        if ! command -v huggingface-cli &> /dev/null; then
            echo "    Installing huggingface-hub..."
            pip install -q huggingface-hub
        fi
        mkdir -p "$dir"
        echo "    ↓ Downloading Phi-3-mini (cpu-int4, ~2GB)..."
        huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx \
            --include "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*" \
            --local-dir "$dir"
    fi

    echo "  ✅ Ready: TextGenerationLocal"
    echo "  Note: TextGenerationMeai uses DemoChatClient and needs no model."
}

# ── usage / help ──────────────────────────────────────────────────────────

show_help() {
    echo "Usage: bash scripts/download-models.sh [task...]"
    echo ""
    echo "Tasks:"
    echo "  embeddings-core    all-MiniLM-L6-v2 + shared samples (~86MB)"
    echo "  embeddings-all     all embedding models (~470MB)"
    echo "  classification     sentiment, emotion, zero-shot (~1GB)"
    echo "  reranking          cross-encoder reranking (~1.2GB)"
    echo "  ner                NER via optimum-cli export (~500MB)"
    echo "  qa                 QA models (~600MB)"
    echo "  textgen            Phi-3-mini for local generation (~2GB)"
    echo "  all                everything except textgen (~3.5GB)"
    echo "  all+textgen        everything including textgen (~5.5GB)"
    echo ""
    echo "Examples:"
    echo "  bash scripts/download-models.sh embeddings-core"
    echo "  bash scripts/download-models.sh classification reranking"
    echo "  bash scripts/download-models.sh all"
}

# ── main ──────────────────────────────────────────────────────────────────

if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

for task in "$@"; do
    case "$task" in
        embeddings-core)
            download_embeddings_core
            ;;
        embeddings-all)
            download_embeddings_core
            download_embeddings_extra
            ;;
        classification)
            download_classification
            ;;
        reranking)
            download_reranking
            ;;
        ner)
            download_ner
            ;;
        qa)
            download_qa
            ;;
        textgen)
            download_textgen
            ;;
        all)
            download_embeddings_core
            download_embeddings_extra
            download_classification
            download_reranking
            download_ner
            download_qa
            ;;
        all+textgen)
            download_embeddings_core
            download_embeddings_extra
            download_classification
            download_reranking
            download_ner
            download_qa
            download_textgen
            ;;
        *)
            echo "Unknown task: $task"
            echo "Run with --help for available tasks."
            exit 1
            ;;
    esac
done

echo ""
echo "Done! Run any sample with: cd samples/<SampleName> && dotnet run"
