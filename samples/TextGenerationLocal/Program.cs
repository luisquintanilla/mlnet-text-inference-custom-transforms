using Microsoft.ML;
using MLNet.TextGeneration.OnnxGenAI;

Console.WriteLine("=== Local ONNX Text Generation ===\n");
Console.WriteLine("Use OnnxTextGenerationEstimator for local text generation with ONNX Runtime GenAI.\n");

// --- Configuration ---
// Download a model first (see README.md for instructions).
var modelPath = args.Length > 0
    ? args[0]
    : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "phi-3-mini"));

if (!Directory.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine("Please download a model first. See README.md for instructions.");
    Console.WriteLine("\nUsage: dotnet run [model-path]");
    return;
}

var mlContext = new MLContext();

var options = new OnnxTextGenerationOptions
{
    ModelPath = modelPath,
    MaxLength = 256,
    Temperature = 0.7f,
    TopP = 0.9f,
    SystemPrompt = "You are a helpful assistant. Be concise."
};

// --- ML.NET Pipeline ---
Console.WriteLine("1. ML.NET Pipeline with OnnxTextGenerationEstimator");
Console.WriteLine(new string('-', 55));

var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "Explain .NET in one sentence." },
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);
var estimator = mlContext.Transforms.OnnxTextGeneration(options);
var transformer = estimator.Fit(dataView);
var transformed = transformer.Transform(dataView);

var results = mlContext.Data.CreateEnumerable<TextGenerationResult>(transformed, reuseRowObject: false).ToList();

foreach (var (item, idx) in results.Select((r, i) => (r, i)))
{
    Console.WriteLine($"  Prompt:   \"{sampleData[idx].Text}\"");
    Console.WriteLine($"  Response: \"{item.GeneratedText}\"");
    Console.WriteLine();
}

// --- Direct Generate() API ---
Console.WriteLine("2. Direct Generate() API");
Console.WriteLine(new string('-', 55));

var prompts = new[] { "What is 2+2?", "Name three colors." };
var responses = transformer.Generate(prompts);

for (int i = 0; i < prompts.Length; i++)
{
    Console.WriteLine($"  Prompt:   \"{prompts[i]}\"");
    Console.WriteLine($"  Response: \"{responses[i]}\"");
    Console.WriteLine();
}

// Cleanup
transformer.Dispose();
Console.WriteLine("Done!");

// --- Domain types ---
public class TextData
{
    public string Text { get; set; } = "";
}

public class TextGenerationResult
{
    public string Text { get; set; } = "";
    public string GeneratedText { get; set; } = "";
}
