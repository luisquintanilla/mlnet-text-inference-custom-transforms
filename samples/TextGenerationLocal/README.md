# TextGenerationLocal Sample

Use `OnnxTextGenerationEstimator` for **local text generation** with ONNX Runtime GenAI — no cloud APIs required.

## Model Download

This sample requires a local ONNX GenAI model. We recommend [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx).

### Using Hugging Face CLI

```bash
pip install huggingface-hub
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*" --local-dir models/phi-3-mini
```

### Manual Download

1. Visit [microsoft/Phi-3-mini-4k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx)
2. Download the `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4` variant
3. Place all files in `models/phi-3-mini/`

## Run

```bash
# Default model path (models/phi-3-mini)
dotnet run

# Custom model path
dotnet run -- /path/to/your/model
```

## What This Sample Shows

1. **ML.NET Pipeline** — `OnnxTextGenerationEstimator` as a standard ML.NET transform with local ONNX inference
2. **Direct API** — Use `OnnxTextGenerationTransformer.Generate()` directly without IDataView overhead
3. **Prompt Formatting** — Automatic chat template formatting with system prompt support

## Key Code Pattern

```csharp
var options = new OnnxTextGenerationOptions
{
    ModelPath = "/path/to/model",
    MaxLength = 256,
    Temperature = 0.7f,
    SystemPrompt = "You are a helpful assistant."
};

var estimator = mlContext.Transforms.OnnxTextGeneration(options);
var transformer = estimator.Fit(dataView);
var results = transformer.Transform(dataView);

// Or use directly
var responses = transformer.Generate(new[] { "What is ML.NET?" });

// Don't forget to dispose (owns the ONNX model)
transformer.Dispose();
```
