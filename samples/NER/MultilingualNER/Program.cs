using Microsoft.ML;
using MLNet.TextInference.Onnx;

// Paths — download model from https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl (ONNX export)
var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models"));

Console.WriteLine("=== Multilingual NER (Davlan/bert-base-multilingual-cased-ner-hrl) ===\n");

var mlContext = new MLContext();

// BIO labels for Davlan/bert-base-multilingual-cased-ner-hrl
string[] labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"];

/*
bert-base-multilingual-cased-ner-hrl is a Named Entity Recognition model created by Davlan that identifies 
three entity types—person (PER), organization (ORG), and location (LOC)—across 10 languages: 
Arabic, German, English, Spanish, French, Italian, Latvian, Dutch, Portuguese, and Chinese.

Abbreviation	Description
O	    Outside of a named entity
B-PER	Beginning of a person’s name right after another person’s name
I-PER	Person’s name
B-ORG	Beginning of an organisation right after another organisation
I-ORG	Organisation
B-LOC	Beginning of a location right after another location
I-LOC	Location
 */

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

var sampleData = new[]
{
    new TextData { Text = "John Smith works at Microsoft in Seattle." },
    new TextData { Text = "Marie Curie a travaillé à l'Université de Paris." },
    new TextData { Text = "Angela Merkel war Bundeskanzlerin von Deutschland." },
    new TextData { Text = "東京タワーは日本の東京都港区にあります。" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

Console.WriteLine("Fitting multilingual NER pipeline...");
var estimator = mlContext.Transforms.OnnxNer(nerOptions);
var transformer = estimator.Fit(dataView);

var texts = sampleData.Select(s => s.Text).ToList();
var entities = transformer.ExtractEntities(texts);

for (int i = 0; i < texts.Count; i++)
{
    Console.WriteLine($"\nText: \"{texts[i]}\"");
    if (entities[i].Length == 0)
    {
        Console.WriteLine("  (no entities found)");
    }
    else
    {
        foreach (var e in entities[i])
        {
            Console.WriteLine($"  {e.EntityType}: \"{e.Word}\" [{e.StartChar}..{e.EndChar}] (score: {e.Score:F4})");
        }
    }
}

Console.WriteLine("\nDone!");
transformer.Dispose();

public class TextData
{
    public string Text { get; set; } = "";
}
