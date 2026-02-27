# ML.NET Text Inference Custom Transforms

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/luisquintanilla/mlnet-text-inference-custom-transforms?quickstart=1)

A **multi-task text inference platform** for ML.NET that runs local HuggingFace ONNX encoder models. Provides a shared foundation of tokenization + ONNX scoring, with task-specific post-processing transforms for embeddings, classification, NER, reranking, question answering, and more.

```
                      TextTokenizerTransformer
                               │
                               ▼
                    OnnxTextModelScorerTransformer          (task-agnostic)
                               │
        ┌──────────┬───────────┼───────────┬──────────┐
        │          │           │           │          │
  EmbeddingPool SoftmaxClass SigmoidScor NerDecoding QaSpanExtract
  Transformer   Transformer  Transformer Transformer Transformer

  ChatClientTransformer (text generation — provider-agnostic, separate pipeline)
```

## Task Status

| Task | Status | Post-processor | Facade |
|------|--------|---------------|--------|
| Embeddings | ✅ Implemented | `EmbeddingPoolingTransformer` | `OnnxTextEmbeddingEstimator` |
| Classification | ✅ Implemented | `SoftmaxClassificationTransformer` | `OnnxTextClassificationEstimator` |
| Reranking | ✅ Implemented | `SigmoidScorerTransformer` | `OnnxRerankerEstimator` |
| NER | ✅ Implemented | `NerDecodingTransformer` | `OnnxNerEstimator` |
| QA | ✅ Implemented | `QaSpanExtractionTransformer` | `OnnxQaEstimator` |
| Text Generation | ✅ Implemented | `ChatClientTransformer` | N/A (provider-agnostic) |
| Text Generation (local) | ✅ Implemented | `OnnxTextGenerationTransformer` | `OnnxTextGenerationEstimator` |

## Why This Exists

This project is forked from [`mlnet-embedding-custom-transforms`](https://github.com/luisquintanilla/mlnet-embedding-custom-transforms), which provides embedding generation only. This fork extends the platform to support **all encoder transformer tasks** — embeddings, classification, NER, reranking, and question answering — by sharing a task-agnostic tokenization and ONNX scoring foundation and adding task-specific post-processing transforms.

ML.NET has no built-in transform for modern HuggingFace encoder models (all-MiniLM-L6-v2, BGE, E5, DeBERTa, etc.). Building one is hard because ML.NET's convenient internal base classes (`RowToRowTransformerBase`, `OneToOneTransformerBase`) have `private protected` constructors — they can't be subclassed from external projects.

This project implements custom transforms using direct `IEstimator<T>` / `ITransformer` interfaces (Approach C from the [ML.NET Custom Transformer Guide](https://github.com/luisquintanilla/mlnet-custom-transformer-guide)), enhanced with custom zip-based save/load for model persistence.

## Features

- **Composable modular pipeline** — task-agnostic transforms (`TokenizeText → ScoreOnnxTextModel`) plus task-specific post-processing that can be inspected, swapped, and reused
- **Convenience facades** — `OnnxTextEmbeddingEstimator`, `OnnxTextClassificationEstimator`, `OnnxRerankerEstimator`, `OnnxNerEstimator`, `OnnxQaEstimator` each wrap all transforms for their task in a single call
- **Provider-agnostic MEAI integration** — `EmbeddingGeneratorEstimator` wraps any `IEmbeddingGenerator<string, Embedding<float>>` as an ML.NET transform; `ChatClientEstimator` wraps any `IChatClient` for text generation
- **Text-pair tokenization** — cross-encoder reranking uses `[CLS] A [SEP] B [SEP]` with token type IDs for query-document pairs
- **Token offset tracking** — NER tokenization preserves character offsets via `EncodeToTokens()` for mapping entities back to source text
- **Multi-output ONNX scoring** — QA models produce separate start/end logit tensors via `AdditionalOutputTensorNames`
- **Smart tokenizer resolution** — point to a directory; auto-detects from `tokenizer_config.json` or known vocab files (BPE, SentencePiece, WordPiece)
- **ONNX auto-discovery** — automatically detects input/output tensor names, shapes, and dimensions from model metadata
- **Self-contained save/load** — serializes to a portable `.mlnet` zip file containing the ONNX model, tokenizer, and config
- **SIMD-accelerated post-processing** — pooling and normalization use `TensorPrimitives` for hardware-vectorized math
- **Configurable batching** — process rows in configurable batch sizes to bound memory usage
- **Multiple pooling strategies** — Mean, CLS token, and Max pooling (for embeddings)

## Quick Start

### Option A: GitHub Codespaces (fastest)

Click **"Open in GitHub Codespaces"** above. The dev container automatically:
1. Installs .NET 10, Python 3.12, and CUDA toolkit
2. Restores packages and builds the solution
3. Downloads the starter model (all-MiniLM-L6-v2, ~86MB)

Once ready, run the first sample:

```bash
cd samples/BasicUsage && dotnet run
```

Download models for other tasks:

```bash
bash scripts/download-models.sh classification  # sentiment, emotion, zero-shot
bash scripts/download-models.sh reranking       # cross-encoder reranking
bash scripts/download-models.sh ner             # named entity recognition
bash scripts/download-models.sh qa              # question answering
bash scripts/download-models.sh all             # everything (~3.5GB)
bash scripts/download-models.sh --help          # see all options
```

### Option B: Local setup

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0). Python 3.x needed only for NER/QA model export.

```bash
dotnet restore && dotnet build
bash scripts/download-models.sh embeddings-core   # downloads ~86MB starter model
cd samples/BasicUsage && dotnet run
```

Or download manually:

```powershell
mkdir samples/BasicUsage/models
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" -OutFile "samples/BasicUsage/models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" -OutFile "samples/BasicUsage/models/vocab.txt"
```

## Code Examples (Embeddings)

> **Note:** Embeddings are the simplest task. Browse the [samples/](samples/) directory for classification, reranking, NER, QA, and text generation examples.

```csharp
using Microsoft.ML;
using MLNet.TextInference.Onnx;

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

### Convenience facade (single-shot)

```csharp
var estimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
{
    ModelPath = "models/model.onnx",
    TokenizerPath = "models/",
});
var transformer = estimator.Fit(data);
var embeddings = transformer.Transform(data);
```

### Provider-agnostic MEAI embedding

```csharp
using Microsoft.Extensions.AI;

// Works with ANY IEmbeddingGenerator — ONNX, OpenAI, Azure, Ollama...
IEmbeddingGenerator<string, Embedding<float>> generator =
    new OnnxEmbeddingGenerator(mlContext, transformer);

var estimator = mlContext.Transforms.TextEmbedding(generator);
```

### Save and load

```csharp
// Save — bundles ONNX model + tokenizer + config into a portable zip
transformer.Save("my-embedding-model.mlnet");

// Load — fully self-contained, no external file dependencies
var loaded = OnnxTextEmbeddingTransformer.Load(mlContext, "my-embedding-model.mlnet");
```

## Model Compatibility

The shared foundation supports any encoder transformer ONNX model. Compatible model architectures include:

- **BERT** and derivatives (all-MiniLM, all-mpnet)
- **RoBERTa** (including XLM-RoBERTa for multilingual)
- **DistilBERT**
- **DeBERTa** / DeBERTa-v2
- **MiniLM** (Microsoft)
- **MPNet** (Microsoft)
- **E5** (intfloat)
- **BGE** (BAAI)
- **GTE** (Alibaba)

### Tested Embedding Models

| Model | Dimensions | Size | Tested | Sample |
|-------|:----------:|:----:|:------:|--------|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | ~86 MB | ✅ | [BasicUsage](samples/BasicUsage/) |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | ~127 MB | ✅ | [BgeSmallEmbedding](samples/BgeSmallEmbedding/) |
| [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) | 384 | ~127 MB | ✅ | [E5SmallEmbedding](samples/E5SmallEmbedding/) |
| [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) | 384 | ~127 MB | ✅ | [GteSmallEmbedding](samples/GteSmallEmbedding/) |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | 384 | ~120 MB | — | |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 768 | ~420 MB | — | |

### Compatible Classification Models

Any encoder model fine-tuned for sequence classification (outputs logits over label classes):
- **Sentiment**: DistilBERT-SST2, BERT-SST2
- **Emotion**: RoBERTa-emotion, GoEmotions
- **NLI/Zero-shot**: DeBERTa-v3-NLI, BART-large-MNLI

### Compatible Reranking Models

Cross-encoder models that take text pairs and output a relevance score:
- **MS MARCO**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **BGE**: BAAI/bge-reranker-base, bge-reranker-v2-m3

### Compatible NER Models

Token classification models with BIO tagging:
- **BERT-NER**: dslim/bert-base-NER
- **Multilingual**: Davlan/xlm-roberta-base-ner-hrl

### Compatible QA Models

Extractive QA models outputting start/end logits:
- **RoBERTa**: deepset/roberta-base-squad2
- **MiniLM**: deepset/minilm-uncased-squad2

Models with `sentence_embedding` output (pre-pooled) are auto-detected and pooling is skipped.

## Project Structure

```
mlnet-text-inference-custom-transforms/
├── src/
│   ├── MLNet.TextInference.Onnx/
│   │   ├── TextTokenizerEstimator.cs         — Transform 1: tokenization (BPE/WordPiece/SentencePiece)
│   │   ├── OnnxTextModelScorerEstimator.cs   — Transform 2: ONNX inference with lookahead batching
│   │   ├── ChatClientEstimator.cs            — Provider-agnostic text generation (IChatClient)
│   │   ├── MLContextExtensions.cs            — Extension methods for fluent API
│   │   ├── Embeddings/
│   │   │   ├── EmbeddingPoolingEstimator.cs  — Pooling + L2 normalization
│   │   │   ├── OnnxTextEmbeddingEstimator.cs — Convenience facade
│   │   │   ├── EmbeddingGeneratorEstimator.cs— MEAI IEmbeddingGenerator wrapper
│   │   │   └── ...                           — OnnxEmbeddingGenerator, ModelPackager, PoolingStrategy
│   │   ├── Classification/
│   │   │   ├── SoftmaxClassificationEstimator.cs — Softmax post-processing
│   │   │   ├── OnnxTextClassificationEstimator.cs— Full pipeline facade
│   │   │   └── ...                              — Options, Transformer, ClassificationResult
│   │   ├── Reranking/
│   │   │   ├── SigmoidScorerEstimator.cs     — Sigmoid scoring transform
│   │   │   ├── OnnxRerankerEstimator.cs      — Cross-encoder facade
│   │   │   └── ...                           — Options, Transformer
│   │   ├── NER/
│   │   │   ├── NerDecodingEstimator.cs       — BIO tag decoding to entity spans
│   │   │   ├── OnnxNerEstimator.cs           — End-to-end NER facade
│   │   │   └── ...                           — Options, Transformer, NerEntity
│   │   └── QA/
│   │       ├── QaSpanExtractionEstimator.cs  — Span extraction from start/end logits
│   │       ├── OnnxQaEstimator.cs            — End-to-end QA facade
│   │       └── ...                           — Options, Transformer, QaResult
│   └── MLNet.TextGeneration.OnnxGenAI/
│       ├── OnnxTextGenerationEstimator.cs    — ORT GenAI local text generation
│       ├── MLContextExtensions.cs            — OnnxTextGeneration() extension method
│       └── ...                               — Options, Transformer
├── samples/
│   ├── BasicUsage/                           — all-MiniLM-L6-v2: all embedding API surfaces
│   ├── BgeSmallEmbedding/                    — BGE-small: query prefix pattern
│   ├── E5SmallEmbedding/                     — E5-small: query/passage prefixes
│   ├── GteSmallEmbedding/                    — GTE-small: semantic search (no prefix)
│   ├── ComposablePoolingComparison/          — 3 pooling strategies, shared inference
│   ├── IntermediateInspection/               — Inspect tokens, masks, raw output
│   ├── MeaiProviderAgnostic/                 — Provider-agnostic MEAI transform
│   ├── Classification/
│   │   ├── SentimentDistilBERT/              — Sentiment analysis with DistilBERT
│   │   ├── EmotionRoBERTa/                   — Multi-class emotion classification
│   │   └── ZeroShotDeBERTa/                  — Zero-shot NLI classification with DeBERTa
│   ├── Reranking/
│   │   ├── MsMarcoMiniLM/                    — MS MARCO cross-encoder reranking
│   │   └── BgeReranker/                      — BGE reranker for retrieval
│   ├── NER/
│   │   ├── BertBaseNER/                      — Named entity recognition with BERT-base
│   │   └── MultilingualNER/                  — Multilingual NER with XLM-RoBERTa
│   ├── QA/
│   │   ├── RobertaSquad2/                    — Extractive QA with RoBERTa on SQuAD 2.0
│   │   └── MiniLMSquad2/                     — Extractive QA with MiniLM on SQuAD 2.0
│   ├── TextGenerationMeai/                   — Provider-agnostic text generation (IChatClient)
│   └── TextGenerationLocal/                  — Local ORT GenAI text generation
├── docs/                                     — Detailed documentation
│   ├── architecture-decision-record.md       — ADR for multi-task platform decisions
│   ├── architecture.md                       — Component walkthrough + pipeline stages
│   ├── design-decisions.md                   — Why every choice was made
│   ├── extending.md                          — How to modify, extend, and add new tasks
│   ├── tensor-deep-dive.md                   — System.Numerics.Tensors for AI workloads
│   └── references.md                         — All sources and further reading
├── proposals/                                — Design proposals for the modular architecture
└── nuget.config                              — NuGet source (nuget.org only)
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
| [SentimentDistilBERT](samples/Classification/SentimentDistilBERT/) | DistilBERT-SST2 | Sentiment classification with softmax post-processing |
| [EmotionRoBERTa](samples/Classification/EmotionRoBERTa/) | RoBERTa-emotion | Multi-class emotion classification |
| [ZeroShotDeBERTa](samples/Classification/ZeroShotDeBERTa/) | DeBERTa-v3-NLI | Zero-shot classification via NLI entailment |
| [MsMarcoMiniLM](samples/Reranking/MsMarcoMiniLM/) | MS MARCO MiniLM | Cross-encoder reranking with text-pair tokenization |
| [BgeReranker](samples/Reranking/BgeReranker/) | BAAI/bge-reranker | BGE cross-encoder reranking for retrieval |
| [BertBaseNER](samples/NER/BertBaseNER/) | BERT-base-NER | Named entity recognition with BIO tag decoding |
| [MultilingualNER](samples/NER/MultilingualNER/) | XLM-RoBERTa-NER | Multilingual named entity recognition |
| [RobertaSquad2](samples/QA/RobertaSquad2/) | RoBERTa-SQuAD2 | Extractive question answering with span extraction |
| [MiniLMSquad2](samples/QA/MiniLMSquad2/) | MiniLM-SQuAD2 | Extractive question answering (lighter model) |
| [TextGenerationMeai](samples/TextGenerationMeai/) | Any IChatClient | Provider-agnostic text generation via MEAI |
| [TextGenerationLocal](samples/TextGenerationLocal/) | ORT GenAI model | Local text generation with ONNX Runtime GenAI |

## API at a Glance

### Shared Foundation

| Class | Role | Key Methods |
|-------|------|-------------|
| `TextTokenizerEstimator` | Transform 1: Tokenization | `Fit(IDataView)` |
| `OnnxTextModelScorerEstimator` | Transform 2: ONNX Scoring | `Fit(IDataView)` → `.HiddenDim`, `.HasPooledOutput` |

### Embeddings

| Class | Role | Key Methods |
|-------|------|-------------|
| `EmbeddingPoolingEstimator` | Transform 3: Pooling | `Fit(IDataView)` |
| `OnnxTextEmbeddingEstimator` | Facade (chains 1→2→3) | `Fit(IDataView)`, `GetOutputSchema()` |
| `OnnxTextEmbeddingTransformer` | Facade transformer | `Transform(IDataView)`, `Save(path)`, `Load(ctx, path)` |
| `EmbeddingGeneratorEstimator` | MEAI wrapper transform | `Fit(IDataView)` |
| `OnnxEmbeddingGenerator` | MEAI `IEmbeddingGenerator` | `GenerateAsync(texts)` |

### Classification

| Class | Role | Key Methods |
|-------|------|-------------|
| `SoftmaxClassificationEstimator` | Softmax post-processing | `Fit(IDataView)` |
| `OnnxTextClassificationEstimator` | Facade (tokenize→score→softmax) | `Fit(IDataView)` |
| `OnnxTextClassificationTransformer` | Facade transformer | `Transform(IDataView)`, `Classify(texts)` |

### Reranking

| Class | Role | Key Methods |
|-------|------|-------------|
| `SigmoidScorerEstimator` | Sigmoid scoring transform | `Fit(IDataView)` |
| `OnnxRerankerEstimator` | Facade (text-pair tokenize→score→sigmoid) | `Fit(IDataView)` |
| `OnnxRerankerTransformer` | Facade transformer | `Transform(IDataView)`, `Rerank(query, docs)` |

### Named Entity Recognition

| Class | Role | Key Methods |
|-------|------|-------------|
| `NerDecodingEstimator` | BIO tag decoding | `Fit(IDataView)` |
| `OnnxNerEstimator` | Facade (tokenize→score→decode) | `Fit(IDataView)` |
| `OnnxNerTransformer` | Facade transformer | `Transform(IDataView)`, `RecognizeEntities(texts)` |

### Question Answering

| Class | Role | Key Methods |
|-------|------|-------------|
| `QaSpanExtractionEstimator` | Span extraction from logits | `Fit(IDataView)` |
| `OnnxQaEstimator` | Facade (tokenize→multi-score→extract) | `Fit(IDataView)` |
| `OnnxQaTransformer` | Facade transformer | `Transform(IDataView)`, `Answer(questions)` |

### Text Generation

| Class | Role | Key Methods |
|-------|------|-------------|
| `ChatClientEstimator` | Provider-agnostic (wraps IChatClient) | `Fit(IDataView)` |
| `ChatClientTransformer` | Provider-agnostic transformer | `Transform(IDataView)` |
| `OnnxTextGenerationEstimator` | ORT GenAI local generation | `Fit(IDataView)` |
| `OnnxTextGenerationTransformer` | ORT GenAI transformer | `Transform(IDataView)` |

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

### MLNet.TextInference.Onnx (encoder tasks)

| Package | Version | Purpose |
|---------|---------|---------|
| `Microsoft.ML` | 5.0.0 | IEstimator/ITransformer, IDataView, MLContext |
| `Microsoft.ML.OnnxRuntime.Managed` | 1.24.2 | InferenceSession, OrtValue (managed API; bring your own native runtime) |
| `Microsoft.ML.Tokenizers` | 2.0.0 | BertTokenizer (WordPiece), BPE, SentencePiece |
| `Microsoft.Extensions.AI.Abstractions` | 10.3.0 | IEmbeddingGenerator, IChatClient |
| `System.Numerics.Tensors` | 10.0.3 | Tensor\<T\>, TensorPrimitives |

### MLNet.TextGeneration.OnnxGenAI (local text generation)

| Package | Version | Purpose |
|---------|---------|---------|
| `Microsoft.ML` | 5.0.0 | IEstimator/ITransformer, IDataView, MLContext |
| `Microsoft.ML.OnnxRuntimeGenAI` | 0.7.1 | ORT GenAI for local autoregressive generation |
| `Microsoft.Extensions.AI.Abstractions` | 10.3.0 | IChatClient |

## Documentation

- **[Architecture Decision Record](docs/architecture-decision-record.md)** — Why this repo exists and the platform architecture
- **[Architecture](docs/architecture.md)** — Component walkthrough and pipeline stages
- **[Design Decisions](docs/design-decisions.md)** — Why every choice was made
- **[Extending](docs/extending.md)** — How to modify, extend, and add new tasks
- **[Tensor Deep Dive](docs/tensor-deep-dive.md)** — System.Numerics.Tensors for AI workloads
- **[References](docs/references.md)** — All sources and further reading

## Target Framework

.NET 10 (LTS). Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0).

## License

This is a prototype / reference implementation for educational purposes.
