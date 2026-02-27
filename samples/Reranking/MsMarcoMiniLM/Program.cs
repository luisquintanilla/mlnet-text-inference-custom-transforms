using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx");
var tokenizerPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models");

modelPath = Path.GetFullPath(modelPath);
tokenizerPath = Path.GetFullPath(tokenizerPath);

Console.WriteLine("=== Cross-Encoder Reranking: MS MARCO MiniLM ===\n");

var mlContext = new MLContext();

// Sample query and candidate passages
var query = "What is machine learning?";
var documents = new[]
{
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The weather forecast predicts rain tomorrow in Seattle.",
    "Deep learning uses neural networks with many layers to model complex patterns.",
    "How to bake chocolate chip cookies at home.",
    "ML.NET is a cross-platform machine learning framework for .NET developers.",
};

// Create input data — each row is a query-document pair
var inputData = documents.Select(doc => new QueryDocument { Query = query, Document = doc }).ToArray();
var dataView = mlContext.Data.LoadFromEnumerable(inputData);

// Option 1: Convenience facade
Console.WriteLine("1. Convenience Facade (OnnxRerankerEstimator)");
Console.WriteLine(new string('-', 50));

var options = new OnnxRerankerOptions
{
    ModelPath = modelPath,
    TokenizerPath = tokenizerPath,
    QueryColumnName = "Query",
    DocumentColumnName = "Document",
    OutputColumnName = "Score",
    MaxTokenLength = 512,
    BatchSize = 8,
};

var estimator = new OnnxRerankerEstimator(mlContext, options);
var transformer = estimator.Fit(dataView);
var results = transformer.Transform(dataView);

var scores = mlContext.Data.CreateEnumerable<RerankResult>(results, reuseRowObject: false).ToList();

Console.WriteLine($"Query: \"{query}\"\n");

// Sort by score descending
var ranked = scores
    .Select((s, i) => (Score: s.Score, Index: i, Document: documents[i]))
    .OrderByDescending(x => x.Score)
    .ToList();

foreach (var (score, index, document) in ranked)
{
    Console.WriteLine($"  [{score:F4}] {document}");
}

// Option 2: Composable pipeline
Console.WriteLine($"\n2. Composable Pipeline");
Console.WriteLine(new string('-', 50));

var tokenizerEstimator = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
{
    TokenizerPath = tokenizerPath,
    InputColumnName = "Query",
    SecondInputColumnName = "Document",
    MaxTokenLength = 512,
});
var tokenizerTransformer = tokenizerEstimator.Fit(dataView);
var tokenizedData = tokenizerTransformer.Transform(dataView);

var scorerEstimator = mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions
{
    ModelPath = modelPath,
    MaxTokenLength = 512,
    BatchSize = 8,
    PreferredOutputNames = ["logits", "output"],
});
var scorerTransformer = scorerEstimator.Fit(tokenizedData);
var scoredData = scorerTransformer.Transform(tokenizedData);

var sigmoidEstimator = mlContext.Transforms.SigmoidScore(new SigmoidScorerOptions
{
    OutputColumnName = "Score",
});
var sigmoidTransformer = sigmoidEstimator.Fit(scoredData);
var composableResult = sigmoidTransformer.Transform(scoredData);

var composableScores = mlContext.Data.CreateEnumerable<RerankResult>(composableResult, reuseRowObject: false).ToList();

for (int i = 0; i < composableScores.Count; i++)
{
    Console.WriteLine($"  [{composableScores[i].Score:F4}] {documents[i]}");
}

// Cleanup
transformer.Dispose();
scorerTransformer.Dispose();

Console.WriteLine("\nDone!");

public class QueryDocument
{
    public string Query { get; set; } = "";
    public string Document { get; set; } = "";
}

public class RerankResult
{
    public float Score { get; set; }
}
