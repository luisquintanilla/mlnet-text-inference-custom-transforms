using System.Numerics.Tensors;
using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models"));

Console.WriteLine("=== GTE-Small Embedding Sample ===\n");
Console.WriteLine("This sample demonstrates the composable modular pipeline with GTE-Small");
Console.WriteLine("for semantic search — no prefix engineering needed.\n");

var mlContext = new MLContext();

var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "ML.NET is a machine learning framework" },
    new TextData { Text = "How to bake sourdough bread" },
    new TextData { Text = "Deep learning uses neural networks" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

// --- 1. Composable Pipeline: TokenizeText → ScoreOnnxTextModel → PoolEmbedding ---
Console.WriteLine("1. Composable Modular Pipeline");
Console.WriteLine(new string('-', 40));
Console.WriteLine("  GTE-Small works well without any prefix.");

// Step 1: Tokenize
var tokenizer = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
{
    TokenizerPath = tokenizerPath, // directory — auto-detects tokenizer
    InputColumnName = "Text",
    MaxTokenLength = 128
}).Fit(dataView);
var tokenized = tokenizer.Transform(dataView);

// Step 2: Score with ONNX
var scorer = mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions
{
    ModelPath = modelPath,
    MaxTokenLength = 128,
    BatchSize = 8
}).Fit(tokenized);
var scored = scorer.Transform(tokenized);

Console.WriteLine($"  Hidden dimension: {scorer.HiddenDim}");
Console.WriteLine($"  Has pre-pooled output: {scorer.HasPooledOutput}");

// Step 3: Pool with mean pooling + L2 normalization
var pooler = mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
{
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true,
    HiddenDim = scorer.HiddenDim,
    IsPrePooled = scorer.HasPooledOutput,
    SequenceLength = scorer.HasPooledOutput ? 0 : 128
}).Fit(scored);
var pooled = pooler.Transform(scored);

var embeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(pooled, reuseRowObject: false).ToList();
Console.WriteLine($"  Embedding dimension: {embeddings[0].Embedding.Length}\n");

for (int i = 0; i < embeddings.Count; i++)
    for (int j = i + 1; j < embeddings.Count; j++)
    {
        float sim = TensorPrimitives.CosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding);
        Console.WriteLine($"  \"{sampleData[i].Text}\" vs \"{sampleData[j].Text}\": {sim:F4}");
    }

// --- 2. Semantic search demo ---
Console.WriteLine("\n2. Semantic Search");
Console.WriteLine(new string('-', 40));

// Helper: embed a batch of texts through the shared pipeline
IList<EmbeddingResult> Embed(TextData[] texts)
{
    var dv = mlContext.Data.LoadFromEnumerable(texts);
    return mlContext.Data.CreateEnumerable<EmbeddingResult>(
        pooler.Transform(scorer.Transform(tokenizer.Transform(dv))),
        reuseRowObject: false).ToList();
}

var corpus = new[]
{
    new TextData { Text = "Machine learning is a subset of artificial intelligence." },
    new TextData { Text = "Bread baking requires flour, water, yeast, and salt." },
    new TextData { Text = "Neural networks are inspired by biological neurons." },
    new TextData { Text = "The stock market fluctuates based on economic indicators." },
    new TextData { Text = "Python and C# are popular programming languages." }
};
var corpusEmbeddings = Embed(corpus);

var queries = new[] { "What is deep learning?", "How do I write code?", "Tell me about bread" };
foreach (var query in queries)
{
    var queryEmbedding = Embed([new TextData { Text = query }]);

    Console.WriteLine($"  Query: \"{query}\"");
    var ranked = corpusEmbeddings
        .Select((e, i) => (Text: corpus[i].Text, Sim: TensorPrimitives.CosineSimilarity(queryEmbedding[0].Embedding, e.Embedding)))
        .OrderByDescending(x => x.Sim)
        .ToList();
    foreach (var (text, sim) in ranked)
        Console.WriteLine($"    {sim:F4}  {text}");
    Console.WriteLine();
}

// --- 3. Chained Estimator Pipeline (.Append) ---
Console.WriteLine("3. Chained Estimator Pipeline (.Append)");
Console.WriteLine(new string('-', 40));

var chainedPipeline = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
    {
        TokenizerPath = tokenizerPath,
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
        HiddenDim = 384,       // known from GTE-small architecture
        SequenceLength = 128,
        IsPrePooled = false
    }));

var chainedModel = chainedPipeline.Fit(dataView);
var chainedResult = chainedModel.Transform(dataView);
var chainedEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(chainedResult, reuseRowObject: false).ToList();

float maxChainDiff = 0;
for (int i = 0; i < embeddings.Count; i++)
    for (int d = 0; d < embeddings[i].Embedding.Length; d++)
        maxChainDiff = MathF.Max(maxChainDiff, MathF.Abs(embeddings[i].Embedding[d] - chainedEmbeddings[i].Embedding[d]));
Console.WriteLine($"  Max difference vs step-by-step pipeline: {maxChainDiff:E2} (should be ~0)");

// --- 4. Convenience Facade (single-shot) ---
Console.WriteLine($"\n4. Convenience Facade (OnnxTextEmbeddingEstimator)");
Console.WriteLine(new string('-', 40));

var facadeEstimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
{
    ModelPath = modelPath,
    TokenizerPath = tokenizerPath,
    MaxTokenLength = 128,
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true,
    BatchSize = 8
});
var facadeTransformer = facadeEstimator.Fit(dataView);
var facadeResult = facadeTransformer.Transform(dataView);
var facadeEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(facadeResult, reuseRowObject: false).ToList();

float maxFacadeDiff = 0;
for (int i = 0; i < embeddings.Count; i++)
    for (int d = 0; d < embeddings[i].Embedding.Length; d++)
        maxFacadeDiff = MathF.Max(maxFacadeDiff, MathF.Abs(embeddings[i].Embedding[d] - facadeEmbeddings[i].Embedding[d]));
Console.WriteLine($"  Max difference vs composable pipeline: {maxFacadeDiff:E2} (should be ~0)");
Console.WriteLine("  The facade wraps the same three transforms internally.");

// Cleanup
scorer.Dispose();
facadeTransformer.Dispose();
Console.WriteLine("\nDone!");

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
