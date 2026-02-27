using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx");
var tokenizerPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models");

modelPath = Path.GetFullPath(modelPath);
tokenizerPath = Path.GetFullPath(tokenizerPath);

Console.WriteLine("=== Cross-Encoder Reranking: BGE Reranker ===\n");

var mlContext = new MLContext();

// Sample query and longer document passages
var query = "How does photosynthesis work?";
var documents = new[]
{
    "Photosynthesis is the process by which green plants and certain other organisms transform light energy into chemical energy. During photosynthesis, plants capture light energy with chlorophyll and use it to convert carbon dioxide and water into glucose and oxygen.",
    "The mitochondria is the powerhouse of the cell, responsible for generating most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.",
    "Plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar. This process, known as photosynthesis, occurs primarily in the leaves of the plant through organelles called chloroplasts.",
    "Nuclear fusion is the process that powers the sun and stars, where hydrogen atoms combine under extreme pressure and temperature to form helium, releasing enormous amounts of energy.",
    "Chlorophyll is the green pigment found in plants that absorbs light energy for photosynthesis. It is found in chloroplasts, primarily in the mesophyll cells of leaves.",
};

var inputData = documents.Select(doc => new QueryDocument { Query = query, Document = doc }).ToArray();
var dataView = mlContext.Data.LoadFromEnumerable(inputData);

Console.WriteLine("Reranking with BGE Reranker");
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

var ranked = scores
    .Select((s, i) => (Score: s.Score, Index: i, Document: documents[i]))
    .OrderByDescending(x => x.Score)
    .ToList();

foreach (var (score, index, document) in ranked)
{
    var preview = document.Length > 100 ? document[..100] + "..." : document;
    Console.WriteLine($"  [{score:F4}] {preview}");
}

transformer.Dispose();
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
