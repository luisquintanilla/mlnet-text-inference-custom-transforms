using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models"));

Console.WriteLine("=== Provider-Agnostic MEAI Embedding Transform ===\n");
Console.WriteLine("Use EmbeddingGeneratorEstimator to wrap any IEmbeddingGenerator<string, Embedding<float>>");
Console.WriteLine("as an ML.NET transform — demonstrating provider-agnostic embedding generation.\n");

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

// --- Direct MEAI usage ---
Console.WriteLine($"\n3. Direct MEAI IEmbeddingGenerator Usage");
Console.WriteLine(new string('-', 40));

var meaiTexts = new[] { "What is .NET?", "Tell me about the .NET framework", "How to cook pasta" };
var meaiEmbeddings = await generator.GenerateAsync(meaiTexts);

Console.WriteLine($"  Generated {meaiEmbeddings.Count} embeddings");
Console.WriteLine($"  Vector dimensions: {meaiEmbeddings[0].Vector.Length}");

float sim01 = TensorPrimitives.CosineSimilarity(meaiEmbeddings[0].Vector.Span, meaiEmbeddings[1].Vector.Span);
float sim02 = TensorPrimitives.CosineSimilarity(meaiEmbeddings[0].Vector.Span, meaiEmbeddings[2].Vector.Span);
Console.WriteLine($"  \"{meaiTexts[0]}\" vs \"{meaiTexts[1]}\": {sim01:F4}");
Console.WriteLine($"  \"{meaiTexts[0]}\" vs \"{meaiTexts[2]}\": {sim02:F4}");

Console.WriteLine("\n  Note: This same code works with ANY IEmbeddingGenerator<string, Embedding<float>>.");
Console.WriteLine("  Swap OnnxEmbeddingGenerator for OpenAI, Azure, Ollama, etc. — pipeline code stays identical.");

// Cleanup
onnxTransformer.Dispose();
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
