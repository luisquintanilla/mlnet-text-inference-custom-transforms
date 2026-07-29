using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx");
var vocabPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "vocab.txt");

modelPath = Path.GetFullPath(modelPath);
vocabPath = Path.GetFullPath(vocabPath);

Console.WriteLine("=== Sentiment Classification with DistilBERT ===\n");

var mlContext = new MLContext();

/*
 * https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blob/main/config.json
 
"id2label": {
    "0": "NEGATIVE",
    "1": "POSITIVE"
  },
*/
string[] labels = ["NEGATIVE", "POSITIVE"];

var options = new OnnxTextClassificationOptions
{
    ModelPath = modelPath,
    TokenizerPath = vocabPath,
    InputColumnName = "Text",
    Labels = labels,
    MaxTokenLength = 128,
    BatchSize = 8,
};

var estimator = new OnnxTextClassificationEstimator(mlContext, options);

var sampleData = new[]
{
    new TextData { Text = "I absolutely loved this movie! The acting was superb." },
    new TextData { Text = "This was the worst experience of my life." },
    new TextData { Text = "The food was okay, nothing special." },
    new TextData { Text = "What an amazing concert! Best night ever!" },
    new TextData { Text = "I'm really disappointed with the quality." },
    new TextData { Text = "The service was friendly and the atmosphere was great." },
    // French examples (seems not to work)
    new TextData { Text = "Il était très mécontent." },
    new TextData { Text = "Le repas était décevant." },
    new TextData { Text = "Le repas était excellent." },
    new TextData { Text = "Le menu était très bon avec un dessert délicieux." },
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

Console.WriteLine("Fitting estimator (loading ONNX model + tokenizer)...");
var transformer = estimator.Fit(dataView);
Console.WriteLine($"  Number of classes: {transformer.NumClasses}");
Console.WriteLine($"  Labels: [{string.Join(", ", transformer.Labels ?? [])}]\n");

// --- 1. ML.NET Pipeline ---
Console.WriteLine("1. ML.NET Pipeline Results");
Console.WriteLine(new string('-', 60));

var transformed = transformer.Transform(dataView);
var results = mlContext.Data.CreateEnumerable<ClassificationOutput>(transformed, reuseRowObject: false).ToList();

for (int i = 0; i < results.Count; i++)
{
    Console.WriteLine($"  \"{sampleData[i].Text}\"");
    Console.WriteLine($"    → {results[i].PredictedLabel} (confidence: {results[i].Probabilities.Max():P1})");
    Console.WriteLine($"      Probabilities: [{string.Join(", ", results[i].Probabilities.Select(p => p.ToString("F4")))}]");
}

// --- 2. Direct API ---
Console.WriteLine($"\n2. Direct API (bypass IDataView)");
Console.WriteLine(new string('-', 60));

var texts = sampleData.Select(s => s.Text).ToList();
var directResults = transformer.Classify(texts);

foreach (var (result, idx) in directResults.Select((r, i) => (r, i)))
{
    Console.WriteLine($"  \"{texts[idx]}\"");
    Console.WriteLine($"    → {result.PredictedLabel} (confidence: {result.Confidence:P1})");
}

Console.WriteLine("\nDone!");
transformer.Dispose();

// Domain types
public class TextData
{
    public string Text { get; set; } = "";
}

public class ClassificationOutput
{
    public string Text { get; set; } = "";
    public string PredictedLabel { get; set; } = "";
    public float[] Probabilities { get; set; } = [];
}
