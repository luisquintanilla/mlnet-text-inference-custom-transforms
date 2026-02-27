# MLNet.Embeddings.Onnx

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/luisquintanilla/mlnet-embedding-custom-transforms?quickstart=1)

A custom ML.NET `IEstimator` / `ITransformer` that generates text embeddings using local HuggingFace ONNX models. Provides both a **convenience facade** (single estimator) and a **composable modular pipeline** (three independent transforms) for tokenization, ONNX inference, pooling, and normalization.

```
                                            ┌─ MeanPooling ─┐
Raw text → [TextTokenizer] → token IDs  →  [OnnxScorer] → [Pooling] → L2-normalized embedding
           (BPE / WordPiece)   + masks      (ONNX Runtime)  └─ CLS / Max ─┘
```

## Why This Exists

ML.NET has no built-in transform for modern HuggingFace embedding models (all-MiniLM-L6-v2, BGE, E5, etc.). Building one is hard because ML.NET's convenient internal base classes (`RowToRowTransformerBase`, `OneToOneTransformerBase`) have `private protected` constructors — they can't be subclassed from external projects.

This project implements a custom transform using direct `IEstimator<T>` / `ITransformer` interfaces (Approach C from the [ML.NET Custom Transformer Guide](https://github.com/luisquintanilla/mlnet-custom-transformer-guide)), enhanced with custom zip-based save/load for model persistence.

## Features

- **Composable modular pipeline** — three independent transforms (`TokenizeText → ScoreOnnxTextModel → PoolEmbedding`) that can be inspected, swapped, and reused
- **Convenience facade** — `OnnxTextEmbeddingEstimator` wraps all three transforms in a single call
- **Provider-agnostic MEAI integration** — `EmbeddingGeneratorEstimator` wraps any `IEmbeddingGenerator<string, Embedding<float>>` as an ML.NET transform
- **Smart tokenizer resolution** — point to a directory; auto-detects from `tokenizer_config.json` or known vocab files (BPE, SentencePiece, WordPiece)
- **ONNX auto-discovery** — automatically detects input/output tensor names, shapes, and embedding dimensions from model metadata
- **Self-contained save/load** — serializes to a portable `.mlnet` zip file containing the ONNX model, tokenizer, and config
- **SIMD-accelerated pooling** — mean pooling and L2 normalization use `TensorPrimitives` for hardware-vectorized math
- **Configurable batching** — process rows in configurable batch sizes to bound memory usage
- **Multiple pooling strategies** — Mean, CLS token, and Max pooling

## Quickstart

### 1. Get the model files

Download [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) ONNX model and vocabulary:

```powershell
mkdir models
# ONNX model (~86 MB)
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
# Vocabulary file
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### 2. Composable pipeline (modular)

```csharp
using Microsoft.ML;
using MLNet.Embeddings.Onnx;

var mlContext = new MLContext();
var data = mlContext.Data.LoadFromEnumerable(new[]
{
    new { Text = "What is machine learning?" },
    new { Text = "How to cook pasta" }
});

// Step-by-step: tokenize → score → pool
var tokenizer = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
{
    TokenizerPath = "models/",  // directory — auto-detects tokenizer
    InputColumnName = "Text"
}).Fit(data);

var scorer = mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions
{
    ModelPath = "models/model.onnx"
}).Fit(tokenizer.Transform(data));

var pooler = mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
{
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true,
    HiddenDim = scorer.HiddenDim,
    IsPrePooled = scorer.HasPooledOutput,
    SequenceLength = 128
}).Fit(scorer.Transform(tokenizer.Transform(data)));
```

Or chain with `.Append()` for the idiomatic ML.NET pattern:

```csharp
var pipeline = mlContext.Transforms.TokenizeText(tokenizerOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.PoolEmbedding(poolingOpts));
var model = pipeline.Fit(data);
```

### 3. Convenience facade (single-shot)

```csharp
var estimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
{
    ModelPath = "models/model.onnx",
    TokenizerPath = "models/",
});
var transformer = estimator.Fit(data);
var embeddings = transformer.Transform(data);
```

### 4. Provider-agnostic MEAI embedding

```csharp
using Microsoft.Extensions.AI;

// Works with ANY IEmbeddingGenerator — ONNX, OpenAI, Azure, Ollama...
IEmbeddingGenerator<string, Embedding<float>> generator =
    new OnnxEmbeddingGenerator(mlContext, transformer);

var estimator = mlContext.Transforms.TextEmbedding(generator);
```

### 5. Save and load

```csharp
// Save — bundles ONNX model + tokenizer + config into a portable zip
transformer.Save("my-embedding-model.mlnet");

// Load — fully self-contained, no external file dependencies
var loaded = OnnxTextEmbeddingTransformer.Load(mlContext, "my-embedding-model.mlnet");
```

## Project Structure

```
mlnet-embedding-custom-transforms/
├── src/MLNet.Embeddings.Onnx/
│   ├── TextTokenizerEstimator.cs         — Transform 1: tokenization (BPE/WordPiece/SentencePiece)
│   ├── OnnxTextModelScorerEstimator.cs   — Transform 2: ONNX inference with lookahead batching
│   ├── EmbeddingPoolingEstimator.cs      — Transform 3: pooling + L2 normalization
│   ├── OnnxTextEmbeddingEstimator.cs     — Convenience facade (chains all three)
│   ├── EmbeddingGeneratorEstimator.cs    — Provider-agnostic MEAI wrapper
│   ├── OnnxEmbeddingGenerator.cs         — MEAI IEmbeddingGenerator for ONNX models
│   ├── MLContextExtensions.cs            — Extension methods for fluent API
│   ├── EmbeddingPooling.cs               — SIMD-accelerated pooling via TensorPrimitives
│   ├── ModelPackager.cs                  — Save/load to self-contained zip
│   └── PoolingStrategy.cs               — Mean / CLS / Max pooling enum
├── samples/
│   ├── BasicUsage/                       — all-MiniLM-L6-v2: all API surfaces
│   ├── BgeSmallEmbedding/                — BGE-small: query prefix pattern
│   ├── E5SmallEmbedding/                 — E5-small: query/passage prefixes
│   ├── GteSmallEmbedding/                — GTE-small: semantic search (no prefix)
│   ├── ComposablePoolingComparison/      — 3 pooling strategies, shared inference
│   ├── IntermediateInspection/           — Inspect tokens, masks, raw output
│   └── MeaiProviderAgnostic/             — Provider-agnostic MEAI transform
├── docs/                                 — Detailed documentation
│   ├── design-decisions.md               — Why every choice was made
│   ├── architecture.md                   — Component walkthrough + pipeline stages
│   ├── tensor-deep-dive.md               — System.Numerics.Tensors for AI workloads
│   ├── extending.md                      — How to modify and extend
│   └── references.md                     — All sources and further reading
├── proposals/                            — Design proposals for the modular architecture
└── nuget.config                          — NuGet source (nuget.org only)
```

## Samples

| Sample | Model | Pattern Demonstrated |
|--------|-------|---------------------|
| [BasicUsage](samples/BasicUsage/) | all-MiniLM-L6-v2 | All API surfaces: facade, composable pipeline, `.Append()`, save/load, MEAI |
| [BgeSmallEmbedding](samples/BgeSmallEmbedding/) | BGE-small-en-v1.5 | Composable pipeline + BGE query prefix for asymmetric retrieval |
| [E5SmallEmbedding](samples/E5SmallEmbedding/) | E5-small-v2 | Composable pipeline + E5 dual query:/passage: prefix pattern |
| [GteSmallEmbedding](samples/GteSmallEmbedding/) | GTE-small | Composable pipeline + semantic search (no prefix needed) |
| [ComposablePoolingComparison](samples/ComposablePoolingComparison/) | all-MiniLM-L6-v2 | **3 pooling strategies**, shared tokenizer+scorer (key modularization demo) |
| [IntermediateInspection](samples/IntermediateInspection/) | all-MiniLM-L6-v2 | Inspect token IDs, attention masks, raw output at each pipeline stage |
| [MeaiProviderAgnostic](samples/MeaiProviderAgnostic/) | all-MiniLM-L6-v2 | `EmbeddingGeneratorEstimator` wrapping `IEmbeddingGenerator` |

## API at a Glance

| Class | Role | Key Methods |
|-------|------|-------------|
| `TextTokenizerEstimator` | Transform 1: Tokenization | `Fit(IDataView)` |
| `OnnxTextModelScorerEstimator` | Transform 2: ONNX Scoring | `Fit(IDataView)` → `.HiddenDim`, `.HasPooledOutput` |
| `EmbeddingPoolingEstimator` | Transform 3: Pooling | `Fit(IDataView)` |
| `OnnxTextEmbeddingEstimator` | Facade (chains 1→2→3) | `Fit(IDataView)`, `GetOutputSchema()` |
| `OnnxTextEmbeddingTransformer` | Facade transformer | `Transform(IDataView)`, `Save(path)`, `Load(ctx, path)` |
| `EmbeddingGeneratorEstimator` | MEAI wrapper transform | `Fit(IDataView)` |
| `OnnxEmbeddingGenerator` | MEAI `IEmbeddingGenerator` | `GenerateAsync(texts)` |

## Supported Models

Any sentence-transformer ONNX model that follows the standard input/output convention:

| Model | Dimensions | Size | Tested | Sample |
|-------|:----------:|:----:|:------:|--------|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | ~86 MB | ✅ | [BasicUsage](samples/BasicUsage/) |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | ~127 MB | ✅ | [BgeSmallEmbedding](samples/BgeSmallEmbedding/) |
| [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) | 384 | ~127 MB | ✅ | [E5SmallEmbedding](samples/E5SmallEmbedding/) |
| [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) | 384 | ~127 MB | ✅ | [GteSmallEmbedding](samples/GteSmallEmbedding/) |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | 384 | ~120 MB | — | |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 768 | ~420 MB | — | |

Models with `sentence_embedding` output (pre-pooled) are auto-detected and pooling is skipped.

## GPU Support (CUDA)

The library supports GPU-accelerated ONNX inference via CUDA. The library itself ships with no native binaries — you control the execution provider by choosing your OnnxRuntime package.

### GPU Prerequisites

GPU inference requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) installed on the host machine, plus an NVIDIA GPU with a compatible driver.

**Windows (winget + direct download):**

```powershell
# 1. Install CUDA Toolkit 12.6
winget install Nvidia.CUDA --version 12.6 --source winget

# 2. Download and install cuDNN 9.x for CUDA 12
#    Download the zip from NVIDIA's redistributable endpoint:
$url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.8.0.87_cuda12-archive.zip"
Invoke-WebRequest -Uri $url -OutFile "$env:TEMP\cudnn.zip"
Expand-Archive "$env:TEMP\cudnn.zip" -DestinationPath "$env:TEMP\cudnn"

# 3. Copy cuDNN DLLs to CUDA bin (requires admin)
Copy-Item "$env:TEMP\cudnn\cudnn-*\bin\*.dll" "$env:CUDA_PATH\bin" -Force
```

**Linux (apt):**

```bash
# CUDA Toolkit
sudo apt-get install -y nvidia-cuda-toolkit

# cuDNN (via NVIDIA's apt repository)
# See: https://docs.nvidia.com/deeplearning/cudnn/installation/linux.html
```

**Verify installation:**

```powershell
nvidia-smi                    # Should show GPU + driver version
nvcc --version                # Should show CUDA 12.x
```

### Package Setup

Replace `Microsoft.ML.OnnxRuntime` with `Microsoft.ML.OnnxRuntime.Gpu` in your application:

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.24.2" />
```

> **Samples auto-detect GPU:** The sample projects use a `Directory.Build.props` that checks for `CUDA_PATH` (set by the CUDA Toolkit installer) and automatically switches to `Microsoft.ML.OnnxRuntime.Gpu`. Override with `dotnet build -p:UseGpuRuntime=true` or `dotnet build -p:UseGpuRuntime=false`.

### Usage

```csharp
// Pattern 1: Context-level (applies to all ONNX estimators)
var mlContext = new MLContext();
mlContext.GpuDeviceId = 0; // Use first CUDA device

var estimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
{
    ModelPath = "models/model.onnx",
    TokenizerPath = "models/",
});

// Pattern 2: Per-estimator override
var scorerOptions = new OnnxTextModelScorerOptions
{
    ModelPath = "models/model.onnx",
    GpuDeviceId = 0,       // Override for this estimator only
    FallbackToCpu = true,  // Graceful degradation if CUDA unavailable
};
```

**Resolution order:** Per-estimator `GpuDeviceId` → `MLContext.GpuDeviceId` → `null` (CPU).

When `FallbackToCpu = true`, if CUDA initialization fails the estimator silently falls back to CPU execution.

## NuGet Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `Microsoft.ML` | 5.0.0 | IEstimator/ITransformer, IDataView, MLContext |
| `Microsoft.ML.OnnxRuntime.Managed` | 1.24.2 | InferenceSession, OrtValue (managed API; bring your own native runtime) |
| `Microsoft.ML.Tokenizers` | 2.0.0 | BertTokenizer (WordPiece), BPE, SentencePiece |
| `Microsoft.Extensions.AI.Abstractions` | 10.3.0 | IEmbeddingGenerator |
| `System.Numerics.Tensors` | 10.0.3 | Tensor\<T\>, TensorPrimitives |

## Documentation

For detailed documentation on the design, architecture, and implementation:

- **[Architecture Decision Record](docs/architecture-decision-record.md)** — Why this repo exists and the platform architecture
- **[Design Decisions](docs/design-decisions.md)** — Why every choice was made
- **[Architecture](docs/architecture.md)** — Component walkthrough and pipeline stages
- **[Tensor Deep Dive](docs/tensor-deep-dive.md)** — System.Numerics.Tensors for AI workloads
- **[Extending](docs/extending.md)** — How to modify, extend, and harden
- **[References](docs/references.md)** — All sources and further reading

## Target Framework

.NET 10 (LTS). Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0).

## License

This is a prototype / reference implementation for educational purposes.
