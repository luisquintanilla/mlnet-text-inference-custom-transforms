using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.TextInference.Onnx;

var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models"));

Console.WriteLine("=== Intermediate Pipeline Inspection ===\n");
Console.WriteLine("Inspect token IDs, attention masks, raw ONNX output, and final");
Console.WriteLine("embeddings at each stage of the modular pipeline.\n");

var mlContext = new MLContext();

var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "ML.NET is great" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

// === Step 1: Tokenization ===
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

// === Step 2: ONNX Scoring ===
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

// === Step 3: Pooling + Normalization ===
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

Console.WriteLine("Key insight: with the modular pipeline you can inspect every intermediate");
Console.WriteLine("stage. The monolithic API hides all of this internal state.");

// Cleanup
scorer.Dispose();
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
