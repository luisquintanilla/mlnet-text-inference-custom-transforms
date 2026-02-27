using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx");
var tokenizerPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models");

modelPath = Path.GetFullPath(modelPath);
tokenizerPath = Path.GetFullPath(tokenizerPath);

Console.WriteLine("=== Zero-Shot Classification with DeBERTa (NLI) ===\n");

var mlContext = new MLContext();

var options = new OnnxTextClassificationOptions
{
    ModelPath = modelPath,
    TokenizerPath = tokenizerPath,
    InputColumnName = "Text",
    Labels = ["contradiction", "neutral", "entailment"],
    MaxTokenLength = 256,
    BatchSize = 8,
};

var estimator = new OnnxTextClassificationEstimator(mlContext, options);

// NLI-based zero-shot: input is "premise [SEP] hypothesis"
var sampleData = new[]
{
    new TextData { Text = "The weather is great today. [SEP] The weather is good." },
    new TextData { Text = "The cat is on the mat. [SEP] There are no animals in the house." },
    new TextData { Text = "She went to the store. [SEP] She bought groceries." },
    new TextData { Text = "He is a doctor. [SEP] He works in a hospital." },
    new TextData { Text = "The movie was terrible. [SEP] The movie was excellent." },
    new TextData { Text = "It is raining outside. [SEP] The ground is wet." },
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

Console.WriteLine("Fitting estimator (loading ONNX model + tokenizer)...");
var transformer = estimator.Fit(dataView);
Console.WriteLine($"  Number of classes: {transformer.NumClasses}");
Console.WriteLine($"  Labels: [{string.Join(", ", transformer.Labels ?? [])}]\n");

// Direct API
Console.WriteLine("NLI Classification Results");
Console.WriteLine(new string('-', 60));

var texts = sampleData.Select(s => s.Text).ToList();
var results = transformer.Classify(texts);

foreach (var (result, idx) in results.Select((r, i) => (r, i)))
{
    Console.WriteLine($"  \"{texts[idx]}\"");
    Console.WriteLine($"    → {result.PredictedLabel} (confidence: {result.Confidence:P1})");
    Console.WriteLine($"      Probabilities: [{string.Join(", ", options.Labels!.Zip(result.Probabilities, (l, p) => $"{l}={p:F3}"))}]");
}

Console.WriteLine("\nDone!");
transformer.Dispose();

public class TextData
{
    public string Text { get; set; } = "";
}
