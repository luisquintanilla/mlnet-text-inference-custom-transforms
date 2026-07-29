using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx");
var tokenizerPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models");

modelPath = Path.GetFullPath(modelPath);
tokenizerPath = Path.GetFullPath(tokenizerPath);

Console.WriteLine("=== Emotion Classification with RoBERTa (GoEmotions) ===\n");

var mlContext = new MLContext();

/*
 * https://huggingface.co/lquint/roberta-base-go_emotions-onnx/blob/main/config.json
 
  "id2label": {
    "0": "admiration",
    "1": "amusement",
    "2": "anger",
    "3": "annoyance",
    "4": "approval",
    "5": "caring",
    "6": "confusion",
    "7": "curiosity",
    "8": "desire",
    "9": "disappointment",
    "10": "disapproval",
    "11": "disgust",
    "12": "embarrassment",
    "13": "excitement",
    "14": "fear",
    "15": "gratitude",
    "16": "grief",
    "17": "joy",
    "18": "love",
    "19": "nervousness",
    "20": "optimism",
    "21": "pride",
    "22": "realization",
    "23": "relief",
    "24": "remorse",
    "25": "sadness",
    "26": "surprise",
    "27": "neutral"
  }, 
 */

string[] labels = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral"
    ];

var options = new OnnxTextClassificationOptions
{
    ModelPath = modelPath,
    TokenizerPath = tokenizerPath,
    InputColumnName = "Text",
    Labels = labels,
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
    new TextData { Text = "I feel so proud of my accomplishments." },
    new TextData { Text = "I am so sad that this example don't work with french text." },
    new TextData { Text = "Je suis vraiment déçu que cet exemple ne fonctionne pas avec du texte en français." },
    new TextData { Text = "I admire the C# developers for their work on ML.NET." },
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
