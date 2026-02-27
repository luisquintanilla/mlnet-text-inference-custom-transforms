using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx");
var tokenizerPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models");

modelPath = Path.GetFullPath(modelPath);
tokenizerPath = Path.GetFullPath(tokenizerPath);

Console.WriteLine("=== Emotion Classification with RoBERTa (GoEmotions) ===\n");

var mlContext = new MLContext();

var options = new OnnxTextClassificationOptions
{
    ModelPath = modelPath,
    TokenizerPath = tokenizerPath,
    InputColumnName = "Text",
    Labels =
    [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral"
    ],
    MaxTokenLength = 128,
    BatchSize = 8,
};

var estimator = new OnnxTextClassificationEstimator(mlContext, options);

var sampleData = new[]
{
    new TextData { Text = "I just got promoted at work!" },
    new TextData { Text = "My dog passed away yesterday." },
    new TextData { Text = "That joke was hilarious!" },
    new TextData { Text = "I can't believe they did that to me." },
    new TextData { Text = "Thank you so much for your help!" },
    new TextData { Text = "I'm not sure what to think about this." },
    new TextData { Text = "I love spending time with my family." },
    new TextData { Text = "This is so frustrating and annoying." },
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

Console.WriteLine("Fitting estimator (loading ONNX model + tokenizer)...");
var transformer = estimator.Fit(dataView);
Console.WriteLine($"  Number of classes: {transformer.NumClasses}");
Console.WriteLine($"  Labels: [{string.Join(", ", transformer.Labels ?? [])}]\n");

// Direct API
Console.WriteLine("Classification Results");
Console.WriteLine(new string('-', 60));

var texts = sampleData.Select(s => s.Text).ToList();
var results = transformer.Classify(texts);

foreach (var (result, idx) in results.Select((r, i) => (r, i)))
{
    Console.WriteLine($"  \"{texts[idx]}\"");
    Console.WriteLine($"    → {result.PredictedLabel} (confidence: {result.Confidence:P1})");

    // Show top 3 emotions
    var top3 = result.Probabilities
        .Select((p, i) => (Prob: p, Label: options.Labels![i]))
        .OrderByDescending(x => x.Prob)
        .Take(3);
    Console.WriteLine($"      Top 3: {string.Join(", ", top3.Select(x => $"{x.Label}={x.Prob:F3}"))}");
}

Console.WriteLine("\nDone!");
transformer.Dispose();

public class TextData
{
    public string Text { get; set; } = "";
}
