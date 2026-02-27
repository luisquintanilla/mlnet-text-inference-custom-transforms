using Microsoft.ML;
using MLNet.TextInference.Onnx;

// Paths — download model from https://huggingface.co/dslim/bert-base-NER (ONNX export)
var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models"));

Console.WriteLine("=== BERT-base NER (dslim/bert-base-NER) ===\n");

var mlContext = new MLContext();

// BIO labels for dslim/bert-base-NER
string[] labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"];

// --- 1. End-to-end facade ---
Console.WriteLine("1. End-to-end OnnxNer facade");
Console.WriteLine(new string('-', 40));

var nerOptions = new OnnxNerOptions
{
    ModelPath = modelPath,
    TokenizerPath = tokenizerPath,
    Labels = labels,
    InputColumnName = "Text",
    OutputColumnName = "Entities",
    MaxTokenLength = 128,
    BatchSize = 8
};

var estimator = mlContext.Transforms.OnnxNer(nerOptions);

var sampleData = new[]
{
    new TextData { Text = "John Smith works at Microsoft in Seattle." },
    new TextData { Text = "Angela Merkel met with Emmanuel Macron in Berlin." },
    new TextData { Text = "The United Nations headquarters is in New York." }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

Console.WriteLine("Fitting NER pipeline...");
var transformer = estimator.Fit(dataView);

// Direct API
Console.WriteLine("\nDirect API results:");
var texts = sampleData.Select(s => s.Text).ToList();
var entities = transformer.ExtractEntities(texts);

for (int i = 0; i < texts.Count; i++)
{
    Console.WriteLine($"\n  Text: \"{texts[i]}\"");
    foreach (var e in entities[i])
    {
        Console.WriteLine($"    {e.EntityType}: \"{e.Word}\" [{e.StartChar}..{e.EndChar}] (score: {e.Score:F4})");
    }
}

// --- 2. ML.NET Pipeline ---
Console.WriteLine("\n\n2. ML.NET Pipeline (IDataView)");
Console.WriteLine(new string('-', 40));

var result = transformer.Transform(dataView);
var rows = mlContext.Data.CreateEnumerable<NerResult>(result, reuseRowObject: false).ToList();

for (int i = 0; i < rows.Count; i++)
{
    Console.WriteLine($"  Text: \"{sampleData[i].Text}\"");
    Console.WriteLine($"  Entities JSON: {rows[i].Entities}");
}

Console.WriteLine("\nDone!");
transformer.Dispose();

public class TextData
{
    public string Text { get; set; } = "";
}

public class NerResult
{
    public string Text { get; set; } = "";
    public string Entities { get; set; } = "";
}
