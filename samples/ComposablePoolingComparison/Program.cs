using System.Numerics.Tensors;
using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models"));

Console.WriteLine("=== Composable Pooling Comparison ===\n");
Console.WriteLine("Same model, three pooling strategies — shows how the modular pipeline");
Console.WriteLine("lets you swap post-processing without re-running inference.\n");

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
Console.WriteLine("Step 1: Tokenizing (shared)...");
var tokenizer = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
{
    TokenizerPath = tokenizerPath,
    InputColumnName = "Text",
    MaxTokenLength = 128
}).Fit(dataView);
var tokenized = tokenizer.Transform(dataView);

// Step 2: Score with ONNX (shared across all pooling strategies)
Console.WriteLine("Step 2: ONNX scoring (shared)...");
var scorer = mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions
{
    ModelPath = modelPath,
    MaxTokenLength = 128,
    BatchSize = 8
}).Fit(tokenized);
var scored = scorer.Transform(tokenized);

Console.WriteLine($"  Hidden dimension: {scorer.HiddenDim}");
Console.WriteLine($"  Has pre-pooled output: {scorer.HasPooledOutput}");
Console.WriteLine();

// Step 3: Apply three different pooling strategies to the SAME scored output
Console.WriteLine("Step 3: Applying three pooling strategies to the same scored output...\n");

var strategies = new[] { PoolingStrategy.MeanPooling, PoolingStrategy.ClsToken, PoolingStrategy.MaxPooling };

foreach (var strategy in strategies)
{
    Console.WriteLine($"--- {strategy} ---");

    var pooler = mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
    {
        Pooling = strategy,
        Normalize = true,
        HiddenDim = scorer.HiddenDim,
        IsPrePooled = scorer.HasPooledOutput,
        SequenceLength = scorer.HasPooledOutput ? 0 : 128
    }).Fit(scored);
    var pooled = pooler.Transform(scored);

    var embeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(pooled, reuseRowObject: false).ToList();

    for (int i = 0; i < embeddings.Count; i++)
        for (int j = i + 1; j < embeddings.Count; j++)
        {
            float sim = TensorPrimitives.CosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding);
            Console.WriteLine($"  \"{sampleData[i].Text}\" vs \"{sampleData[j].Text}\": {sim:F4}");
        }
    Console.WriteLine();
}

Console.WriteLine("Key insight: tokenizer + scorer ran ONCE. Only the pooler changed.");
Console.WriteLine("With the monolithic API, you'd re-run ONNX inference for each strategy.\n");

// Cleanup
scorer.Dispose();
Console.WriteLine("Done!");

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
