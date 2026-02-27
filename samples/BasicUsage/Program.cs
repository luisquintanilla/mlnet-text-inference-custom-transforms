using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx");
var vocabPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "vocab.txt");

// Resolve paths
modelPath = Path.GetFullPath(modelPath);
vocabPath = Path.GetFullPath(vocabPath);

Console.WriteLine("=== ONNX Text Embedding Transform for ML.NET ===\n");

// --- 1. ML.NET Pipeline Usage ---
Console.WriteLine("1. ML.NET Pipeline Usage");
Console.WriteLine(new string('-', 40));

var mlContext = new MLContext();

var options = new OnnxTextEmbeddingOptions
{
    ModelPath = modelPath,
    TokenizerPath = vocabPath,
    InputColumnName = "Text",
    OutputColumnName = "Embedding",
    MaxTokenLength = 128,
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true,
    BatchSize = 8
};

var estimator = new OnnxTextEmbeddingEstimator(mlContext, options);

// Create sample data
var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "ML.NET is a machine learning framework for .NET" },
    new TextData { Text = "How to cook pasta" },
    new TextData { Text = "Deep learning and neural networks" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

// Fit (trivial — validates model, creates transformer)
Console.WriteLine("Fitting estimator (loading ONNX model + tokenizer)...");
var transformer = estimator.Fit(dataView);
Console.WriteLine($"  Embedding dimension: {transformer.EmbeddingDimension}");

// Transform
Console.WriteLine("Generating embeddings...");
var transformed = transformer.Transform(dataView);

// Read results
var embeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(transformed, reuseRowObject: false).ToList();

foreach (var (item, idx) in embeddings.Select((e, i) => (e, i)))
{
    Console.WriteLine($"  [{idx}] \"{sampleData[idx].Text}\"");
    Console.WriteLine($"       dims={item.Embedding.Length}, first 5: [{string.Join(", ", item.Embedding.Take(5).Select(f => f.ToString("F4")))}]");
}

// --- 2. Cosine Similarity ---
Console.WriteLine($"\n2. Cosine Similarity");
Console.WriteLine(new string('-', 40));

for (int i = 0; i < embeddings.Count; i++)
{
    for (int j = i + 1; j < embeddings.Count; j++)
    {
        float sim = TensorPrimitives.CosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding);
        Console.WriteLine($"  \"{sampleData[i].Text}\" vs \"{sampleData[j].Text}\": {sim:F4}");
    }
}

// --- 3. Save/Load Round-Trip ---
Console.WriteLine($"\n3. Save/Load Round-Trip");
Console.WriteLine(new string('-', 40));

var savePath = Path.Combine(Path.GetTempPath(), "embedding-model.mlnet");
Console.WriteLine($"  Saving to: {savePath}");
transformer.Save(savePath);
Console.WriteLine($"  File size: {new FileInfo(savePath).Length / 1024 / 1024} MB");

Console.WriteLine("  Loading from saved file...");
using var loaded = OnnxTextEmbeddingTransformer.Load(mlContext, savePath);
Console.WriteLine($"  Loaded embedding dimension: {loaded.EmbeddingDimension}");

// Verify loaded model produces same results
var loadedTransformed = loaded.Transform(dataView);
var loadedEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(loadedTransformed, reuseRowObject: false).ToList();

float maxDiff = 0;
for (int i = 0; i < embeddings.Count; i++)
{
    for (int d = 0; d < embeddings[i].Embedding.Length; d++)
    {
        float diff = MathF.Abs(embeddings[i].Embedding[d] - loadedEmbeddings[i].Embedding[d]);
        maxDiff = MathF.Max(maxDiff, diff);
    }
}
Console.WriteLine($"  Max difference after round-trip: {maxDiff:E2} (should be ~0)");

// Clean up
File.Delete(savePath);

// --- 4. MEAI IEmbeddingGenerator Usage ---
Console.WriteLine($"\n4. MEAI IEmbeddingGenerator Usage");
Console.WriteLine(new string('-', 40));

IEmbeddingGenerator<string, Embedding<float>> generator =
    new OnnxEmbeddingGenerator(mlContext, transformer);

Console.WriteLine($"  Provider: {(generator as OnnxEmbeddingGenerator)?.Metadata.ProviderName}");
Console.WriteLine($"  Model: {(generator as OnnxEmbeddingGenerator)?.Metadata.DefaultModelId}");

var meaiTexts = new[] { "What is .NET?", "Tell me about the .NET framework", "How to cook pasta" };
var meaiEmbeddings = await generator.GenerateAsync(meaiTexts);

Console.WriteLine($"  Generated {meaiEmbeddings.Count} embeddings");
Console.WriteLine($"  Vector dimensions: {meaiEmbeddings[0].Vector.Length}");

float sim01 = TensorPrimitives.CosineSimilarity(meaiEmbeddings[0].Vector.Span, meaiEmbeddings[1].Vector.Span);
float sim02 = TensorPrimitives.CosineSimilarity(meaiEmbeddings[0].Vector.Span, meaiEmbeddings[2].Vector.Span);
Console.WriteLine($"  \"{meaiTexts[0]}\" vs \"{meaiTexts[1]}\": {sim01:F4}");
Console.WriteLine($"  \"{meaiTexts[0]}\" vs \"{meaiTexts[2]}\": {sim02:F4}");
Console.WriteLine("  (.NET topics should be more similar to each other than to cooking)");

// --- 5. Composable Pipeline ---
Console.WriteLine($"\n5. Composable Pipeline");
Console.WriteLine(new string('-', 40));

var tokenizerEstimator = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
{
    TokenizerPath = vocabPath,
    InputColumnName = "Text",
    MaxTokenLength = 128
});
var tokenizerTransformer = tokenizerEstimator.Fit(dataView);
var tokenizedData = tokenizerTransformer.Transform(dataView);

var scorerEstimator = mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions
{
    ModelPath = modelPath,
    MaxTokenLength = 128,
    BatchSize = 8
});
var scorerTransformer = scorerEstimator.Fit(tokenizedData);
var scoredData = scorerTransformer.Transform(tokenizedData);

Console.WriteLine($"  Hidden dimension: {scorerTransformer.HiddenDim}");
Console.WriteLine($"  Pre-pooled output: {scorerTransformer.HasPooledOutput}");

var poolerEstimator = mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
{
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true,
    HiddenDim = scorerTransformer.HiddenDim,
    IsPrePooled = scorerTransformer.HasPooledOutput,
    SequenceLength = scorerTransformer.HasPooledOutput ? 1 : 128
});
var poolerTransformer = poolerEstimator.Fit(scoredData);
var composableResult = poolerTransformer.Transform(scoredData);

var composableEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(composableResult, reuseRowObject: false).ToList();

// Verify composable pipeline matches convenience API
float maxCompDiff = 0;
for (int i = 0; i < embeddings.Count; i++)
{
    for (int d = 0; d < embeddings[i].Embedding.Length; d++)
    {
        float diff = MathF.Abs(embeddings[i].Embedding[d] - composableEmbeddings[i].Embedding[d]);
        maxCompDiff = MathF.Max(maxCompDiff, diff);
    }
}
Console.WriteLine($"  Max difference vs convenience API: {maxCompDiff:E2} (should be ~0)");

// Cleanup
scorerTransformer.Dispose();

// --- 6. Chained Estimator Pipeline ---
Console.WriteLine($"\n6. Chained Estimator Pipeline (.Append)");
Console.WriteLine(new string('-', 40));

// The idiomatic ML.NET pattern: chain estimators with .Append(),
// then Fit + Transform the whole pipeline at once.
// Note: pooling options require model dimensions upfront (384 for MiniLM).
var chainedPipeline = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
    {
        TokenizerPath = vocabPath,
        InputColumnName = "Text",
        MaxTokenLength = 128
    })
    .Append(mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions
    {
        ModelPath = modelPath,
        MaxTokenLength = 128,
        BatchSize = 8
    }))
    .Append(mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
    {
        Pooling = PoolingStrategy.MeanPooling,
        Normalize = true,
        HiddenDim = 384,       // known from model architecture
        SequenceLength = 128,  // matches MaxTokenLength
        IsPrePooled = false
    }));

var chainedModel = chainedPipeline.Fit(dataView);
var chainedResult = chainedModel.Transform(dataView);
var chainedEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(chainedResult, reuseRowObject: false).ToList();

float maxChainDiff = 0;
for (int i = 0; i < embeddings.Count; i++)
{
    for (int d = 0; d < embeddings[i].Embedding.Length; d++)
    {
        float diff = MathF.Abs(embeddings[i].Embedding[d] - chainedEmbeddings[i].Embedding[d]);
        maxChainDiff = MathF.Max(maxChainDiff, diff);
    }
}
Console.WriteLine($"  Max difference vs convenience API: {maxChainDiff:E2} (should be ~0)");

Console.WriteLine("\nDone!");

// Cleanup
transformer.Dispose();

// --- Domain types ---
public class TextData
{
    public string Text { get; set; } = "";
}

public class EmbeddingResult
{
    public string Text { get; set; } = "";
    public float[] Embedding { get; set; } = [];
}
