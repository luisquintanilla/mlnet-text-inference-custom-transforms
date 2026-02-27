#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║  ML.NET Text Inference Custom Transforms             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

echo "==> Restoring NuGet packages..."
dotnet restore

echo "==> Building solution..."
dotnet build --no-restore

echo ""
echo "==> Downloading starter model (all-MiniLM-L6-v2, ~86MB)..."
bash scripts/download-models.sh embeddings-core

echo ""

# Report GPU status
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "🟢 GPU detected: $GPU_NAME"
    echo "   Samples auto-use Microsoft.ML.OnnxRuntime.Gpu via Directory.Build.props."
else
    echo "⚪ No GPU detected — samples will use CPU inference."
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Ready! Try it now:"
echo ""
echo "    cd samples/BasicUsage && dotnet run"
echo ""
echo "  Download models for other tasks:"
echo ""
echo "    bash scripts/download-models.sh all             # everything (~3GB)"
echo "    bash scripts/download-models.sh classification  # sentiment, emotion, zero-shot"
echo "    bash scripts/download-models.sh reranking       # cross-encoder reranking"
echo "    bash scripts/download-models.sh ner             # named entity recognition"
echo "    bash scripts/download-models.sh qa              # question answering"
echo "    bash scripts/download-models.sh --help          # see all options"
echo ""
echo "════════════════════════════════════════════════════════"
