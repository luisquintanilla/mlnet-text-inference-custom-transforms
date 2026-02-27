# Proposal 06: Post-Modularization Samples

## Summary

After the modular transform pipeline (proposals 01–05) merges, add samples that showcase patterns **only possible** with the composable architecture. These demonstrate the value of modularization and serve as acceptance tests for the new transforms.

**Dependency:** Proposals 01–05 must be implemented first. These samples use `TextTokenizerEstimator`, `OnnxTextModelScorerEstimator`, `EmbeddingPoolingEstimator`, and `EmbeddingGeneratorEstimator` directly.

## Samples

### Sample B1: `ComposablePoolingComparison`

**Purpose:** Same model, three pooling strategies — shows how the modular pipeline lets you swap post-processing without re-running inference.

**Model:** all-MiniLM-L6-v2 (reuse from BasicUsage — same model files).

**Pattern demonstrated:**
- Build a pipeline with explicit `TokenizeText → ScoreOnnxTextModel → PoolEmbedding` steps
- Run the tokenizer + scorer once, then apply three different pooling transforms (Mean, CLS, Max) to the same scored output
- Compare cosine similarity rankings across the three strategies
- Show how results differ: CLS tends to weight the [CLS] token's representation, Mean averages all tokens, Max takes the maximum activation per dimension

**Why this needs modularization:** With the monolith, changing pooling strategy requires re-fitting the entire estimator and re-running ONNX inference. With modular transforms, you swap only the pooler.

**Code sketch:**

```csharp
using System.Numerics.Tensors;
using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models")); // directory — auto-detects from tokenizer_config.json or known vocab files

Console.WriteLine("=== Composable Pooling Comparison ===\n");

var mlContext = new MLContext();

var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "ML.NET is a machine learning framework" },
    new TextData { Text = "How to bake sourdough bread" },
    new TextData { Text = "Deep learning uses neural networks" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

// Step 1: Tokenize (shared across all pooling strategies)
var tokenizerEstimator = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
{
    TokenizerPath = tokenizerPath,
    InputColumnName = "Text",
    MaxTokenLength = 128
});
var tokenizer = tokenizerEstimator.Fit(dataView);
var tokenized = tokenizer.Transform(dataView);

// Step 2: Score with ONNX (shared across all pooling strategies)
var scorerEstimator = mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions
{
    ModelPath = modelPath,
    MaxTokenLength = 128,
    BatchSize = 8
});
var scorer = scorerEstimator.Fit(tokenized);
var scored = scorer.Transform(tokenized);

// Step 3: Apply three different pooling strategies to the SAME scored output
var strategies = new[] { PoolingStrategy.MeanPooling, PoolingStrategy.ClsPooling, PoolingStrategy.MaxPooling };

foreach (var strategy in strategies)
{
    Console.WriteLine($"--- {strategy} ---");

    var poolerEstimator = mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
    {
        Pooling = strategy,
        Normalize = true,
        HiddenDim = scorer.HiddenDim,
        IsPrePooled = scorer.HasPooledOutput,
        SequenceLength = scorer.HasPooledOutput ? 0 : 128
    });
    var pooler = poolerEstimator.Fit(scored);
    var pooled = pooler.Transform(scored);

    var embeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(pooled, reuseRowObject: false).ToList();

    // Print pairwise similarities
    for (int i = 0; i < embeddings.Count; i++)
        for (int j = i + 1; j < embeddings.Count; j++)
        {
            float sim = TensorPrimitives.CosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding);
            Console.WriteLine($"  \"{sampleData[i].Text}\" vs \"{sampleData[j].Text}\": {sim:F4}");
        }
    Console.WriteLine();
}

// Cleanup
scorer.Dispose();
Console.WriteLine("Done!");

public class TextData { public string Text { get; set; } = ""; }
public class EmbeddingResult { public string Text { get; set; } = ""; public float[] Embedding { get; set; } = []; }
```

**Acceptance criteria:**
- [ ] Uses three separate transforms explicitly (not the facade)
- [ ] Tokenizer + scorer run once; only the pooler varies
- [ ] Output shows how different pooling strategies affect similarity rankings
- [ ] Builds and runs without errors

---

### Sample B2: `IntermediateInspection`

**Purpose:** Inspect intermediate pipeline outputs — token IDs, attention masks, raw model output — to understand what each transform does.

**Model:** all-MiniLM-L6-v2 (reuse from BasicUsage).

**Pattern demonstrated:**
- Build the composable pipeline step by step
- After tokenization: print token IDs, attention mask, token type IDs for each input text
- After scoring: print the raw output shape and a few values
- After pooling: print the final embedding
- Optionally decode token IDs back to tokens (if the tokenizer supports it) to show the tokenization process

**Why this needs modularization:** The monolith hides all intermediate state. You can't inspect what tokens were produced or what the raw ONNX output looks like.

**Code sketch:**

```csharp
using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models")); // directory — auto-detects tokenizer

Console.WriteLine("=== Intermediate Pipeline Inspection ===\n");

var mlContext = new MLContext();

var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "ML.NET is great" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

// Step 1: Tokenize
Console.WriteLine("=== Step 1: Tokenization ===\n");
var tokenizer = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
{
    TokenizerPath = tokenizerPath,
    InputColumnName = "Text",
    MaxTokenLength = 32  // short for readability
}).Fit(dataView);

var tokenized = tokenizer.Transform(dataView);

// Inspect token columns
using (var cursor = tokenized.GetRowCursor(tokenized.Schema))
{
    var textCol = tokenized.Schema["Text"];
    var tokenIdsCol = tokenized.Schema["TokenIds"];
    var attMaskCol = tokenized.Schema["AttentionMask"];

    var textGetter = cursor.GetGetter<ReadOnlyMemory<char>>(textCol);
    var tokenIdsGetter = cursor.GetGetter<VBuffer<long>>(tokenIdsCol);
    var attMaskGetter = cursor.GetGetter<VBuffer<long>>(attMaskCol);

    while (cursor.MoveNext())
    {
        ReadOnlyMemory<char> text = default;
        VBuffer<long> tokenIds = default;
        VBuffer<long> attMask = default;

        textGetter(ref text);
        tokenIdsGetter(ref tokenIds);
        attMaskGetter(ref attMask);

        Console.WriteLine($"  Text: \"{text}\"");
        Console.WriteLine($"  TokenIds ({tokenIds.Length}): [{string.Join(", ", tokenIds.DenseValues().Take(20))}...]");
        Console.WriteLine($"  AttentionMask: [{string.Join(", ", attMask.DenseValues().Take(20))}...]");
        Console.WriteLine($"  Real tokens: {attMask.DenseValues().Count(v => v == 1)}, Padding: {attMask.DenseValues().Count(v => v == 0)}");
        Console.WriteLine();
    }
}

// Step 2: Score
Console.WriteLine("=== Step 2: ONNX Scoring ===\n");
var scorer = mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions
{
    ModelPath = modelPath,
    MaxTokenLength = 32,
    BatchSize = 8
}).Fit(tokenized);

var scored = scorer.Transform(tokenized);
Console.WriteLine($"  Hidden dimension: {scorer.HiddenDim}");
Console.WriteLine($"  Has pre-pooled output: {scorer.HasPooledOutput}");

// Inspect raw output shape
var rawOutputCol = scored.Schema["RawOutput"];
Console.WriteLine($"  RawOutput column type: {rawOutputCol.Type}");
Console.WriteLine();

// Step 3: Pool
Console.WriteLine("=== Step 3: Pooling + Normalization ===\n");
var pooler = mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
{
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true,
    HiddenDim = scorer.HiddenDim,
    IsPrePooled = scorer.HasPooledOutput,
    SequenceLength = scorer.HasPooledOutput ? 0 : 32
}).Fit(scored);

var pooled = pooler.Transform(scored);

var embeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(pooled, reuseRowObject: false).ToList();
foreach (var (emb, idx) in embeddings.Select((e, i) => (e, i)))
{
    Console.WriteLine($"  [{idx}] \"{sampleData[idx].Text}\"");
    Console.WriteLine($"       Final embedding ({emb.Embedding.Length}d): [{string.Join(", ", emb.Embedding.Take(8).Select(f => f.ToString("F4")))}...]");
    var norm = MathF.Sqrt(emb.Embedding.Sum(x => x * x));
    Console.WriteLine($"       L2 norm: {norm:F4} (should be ~1.0 if normalized)");
    Console.WriteLine();
}

scorer.Dispose();
Console.WriteLine("Done!");

public class TextData { public string Text { get; set; } = ""; }
public class EmbeddingResult { public string Text { get; set; } = ""; public float[] Embedding { get; set; } = []; }
```

**Acceptance criteria:**
- [ ] Shows token IDs, attention masks, and padding for each input
- [ ] Shows raw ONNX output metadata (dimension, pre-pooled vs unpooled)
- [ ] Shows final embedding with L2 norm verification
- [ ] Demonstrates the value of inspectability
- [ ] Builds and runs without errors

---

### Sample B3: `MeaiProviderAgnostic`

**Purpose:** Use `EmbeddingGeneratorEstimator` to wrap an `IEmbeddingGenerator<string, Embedding<float>>` as an ML.NET transform — demonstrating provider-agnostic embedding generation.

**Model:** all-MiniLM-L6-v2 via `OnnxEmbeddingGenerator` (but the pattern works with any MEAI provider).

**Pattern demonstrated:**
- Create an `OnnxEmbeddingGenerator` (MEAI `IEmbeddingGenerator`)
- Wrap it as an ML.NET transform using `EmbeddingGeneratorEstimator`
- Use it in a standard ML.NET pipeline
- Show that the same code would work with an OpenAI, Azure OpenAI, or Ollama embedding generator — only the generator construction changes

**Why this needs modularization:** `EmbeddingGeneratorEstimator` is a new type introduced in proposal 05.

**Code sketch:**

```csharp
using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models")); // directory — auto-detects tokenizer

Console.WriteLine("=== Provider-Agnostic MEAI Embedding Transform ===\n");

var mlContext = new MLContext();

// --- Create an IEmbeddingGenerator (local ONNX in this case) ---
// This is the ONLY part that changes per provider.
// For OpenAI: new OpenAIClient(...).GetEmbeddingClient("text-embedding-3-small").AsEmbeddingGenerator()
// For Azure: new AzureOpenAIClient(...).GetEmbeddingClient("text-embedding-3-small").AsEmbeddingGenerator()
// For Ollama: new OllamaEmbeddingGenerator(new Uri("http://localhost:11434"), "all-minilm")

var onnxOptions = new OnnxTextEmbeddingOptions
{
    ModelPath = modelPath,
    TokenizerPath = tokenizerPath,
    MaxTokenLength = 128,
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true
};

var onnxEstimator = new OnnxTextEmbeddingEstimator(mlContext, onnxOptions);
var onnxTransformer = onnxEstimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<TextData>()));

IEmbeddingGenerator<string, Embedding<float>> generator =
    new OnnxEmbeddingGenerator(mlContext, onnxTransformer);

// --- Use the generator as an ML.NET transform ---
Console.WriteLine("1. Provider-Agnostic ML.NET Pipeline");
Console.WriteLine(new string('-', 40));

var estimator = mlContext.Transforms.TextEmbedding(generator);

var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "ML.NET is a machine learning framework" },
    new TextData { Text = "How to cook pasta" },
    new TextData { Text = "Deep learning and neural networks" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);
var transformer = estimator.Fit(dataView);
var transformed = transformer.Transform(dataView);

var embeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(transformed, reuseRowObject: false).ToList();

foreach (var (item, idx) in embeddings.Select((e, i) => (e, i)))
{
    Console.WriteLine($"  [{idx}] \"{sampleData[idx].Text}\"");
    Console.WriteLine($"       dims={item.Embedding.Length}, first 5: [{string.Join(", ", item.Embedding.Take(5).Select(f => f.ToString("F4")))}]");
}

// --- Cosine Similarity ---
Console.WriteLine($"\n2. Cosine Similarity");
Console.WriteLine(new string('-', 40));

for (int i = 0; i < embeddings.Count; i++)
    for (int j = i + 1; j < embeddings.Count; j++)
    {
        float sim = TensorPrimitives.CosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding);
        Console.WriteLine($"  \"{sampleData[i].Text}\" vs \"{sampleData[j].Text}\": {sim:F4}");
    }

Console.WriteLine("\n  Note: This same code works with ANY IEmbeddingGenerator<string, Embedding<float>>.");
Console.WriteLine("  Swap OnnxEmbeddingGenerator for OpenAI, Azure, Ollama, etc. — pipeline code stays identical.");

// Cleanup
onnxTransformer.Dispose();
Console.WriteLine("\nDone!");

public class TextData { public string Text { get; set; } = ""; }
public class EmbeddingResult { public string Text { get; set; } = ""; public float[] Embedding { get; set; } = []; }
```

**Acceptance criteria:**
- [ ] Uses `EmbeddingGeneratorEstimator` (not the ONNX-specific facade)
- [ ] Demonstrates the provider-agnostic pattern clearly
- [ ] Comments show how to swap to other MEAI providers
- [ ] Builds and runs without errors

---

## File Structure (Post-Modularization)

```
samples/
├── BasicUsage/                      ← existing
├── BgeSmallEmbedding/               ← from Category A issue
├── E5SmallEmbedding/                ← from Category A issue
├── GteSmallEmbedding/               ← from Category A issue
├── ComposablePoolingComparison/     ← NEW (B1)
│   ├── ComposablePoolingComparison.csproj
│   └── Program.cs
├── IntermediateInspection/          ← NEW (B2)
│   ├── IntermediateInspection.csproj
│   └── Program.cs
└── MeaiProviderAgnostic/            ← NEW (B3)
    ├── MeaiProviderAgnostic.csproj
    └── Program.cs
```

**Note:** B1 and B2 reuse the same all-MiniLM-L6-v2 model from `BasicUsage/models/`. They can either symlink/copy the models directory or reference it via relative path. The simplest approach is to have their `Program.cs` point to `../../BasicUsage/models/` or include a setup note to copy/download the model into their own `models/` directory.

## Implementation Order

These samples should be implemented **after** all of proposals 01–05 are merged and the composable pipeline works end-to-end.

1. **B1: ComposablePoolingComparison** — first, because it exercises the core three-transform pipeline (the main deliverable of modularization)
2. **B2: IntermediateInspection** — second, because it demonstrates inspectability (a key motivator for the refactor)
3. **B3: MeaiProviderAgnostic** — third, because it depends on `EmbeddingGeneratorEstimator` from proposal 05

## Relationship to Existing Proposals

| Proposal | What it provides | Which sample uses it |
|----------|-----------------|---------------------|
| [01-text-tokenizer-transform.md](01-text-tokenizer-transform.md) | `TextTokenizerEstimator` | B1, B2 |
| [02-onnx-text-model-scorer-transform.md](02-onnx-text-model-scorer-transform.md) | `OnnxTextModelScorerEstimator` | B1, B2 |
| [03-embedding-pooling-transform.md](03-embedding-pooling-transform.md) | `EmbeddingPoolingEstimator` | B1, B2 |
| [04-facade-refactor.md](04-facade-refactor.md) | Facade still works | B3 (creates transformer via facade) |
| [05-meai-integration.md](05-meai-integration.md) | `EmbeddingGeneratorEstimator` | B3 |

## Relationship to Future Tasks

These samples are distinct from the future-task expansions in [future-tasks.md](future-tasks.md). The future tasks (classification, NER, QA, cross-encoder) require **new post-processing transforms**. These samples use only the transforms defined in proposals 01–05 — no new transforms needed.

However, if a future text classification sample is added, it would follow the same composable pattern as B1/B2:

```csharp
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.SoftmaxClassify(classificationOpts));  // future transform
```
