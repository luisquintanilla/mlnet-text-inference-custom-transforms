using Microsoft.ML;
using MLNet.TextInference.Onnx;

var modelPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx");
var tokenizerPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models");

modelPath = Path.GetFullPath(modelPath);
tokenizerPath = Path.GetFullPath(tokenizerPath);

Console.WriteLine("=== Zero-Shot Classification with DeBERTa (NLI) ===\n");

var mlContext = new MLContext();

/*
 * https://huggingface.co/lquint/DeBERTa-v3-base-mnli-fever-anli-onnx/blob/main/config.json
 * 
 "label2id": {
    "contradiction": 2,
    "entailment": 0,
    "neutral": 1
  }
 */
string[] labels = ["entailment", "neutral", "contradiction"];

var options = new OnnxTextClassificationOptions
{
    ModelPath = modelPath,
    TokenizerPath = tokenizerPath,
    InputColumnName = "Text",
    Labels = labels,
    MaxTokenLength = 256,
    BatchSize = 8,
};

var estimator = new OnnxTextClassificationEstimator(mlContext, options);

// NLI-based zero-shot: input is "premise [SEP] hypothesis"
var sampleData = new[]
{
    new TextData { Text = "The weather is great today. [SEP] It is raining and the temperature is very low." },
    new TextData { Text = "The quote: \"The weather is great today.\" - end of the quote. It is raining and the temperature is very low." },
    new TextData { Text = "The cat is on the mat. [SEP] There are no animals in the house." },
    new TextData { Text = "She went to the store. [SEP] She bought groceries." },
    new TextData { Text = "He is a doctor. [SEP] He works in a hospital." },
    new TextData { Text = "The movie was terrible. [SEP] The movie was excellent." },
    new TextData { Text = "It is raining outside. [SEP] The ground is wet." },
    new TextData { Text = "The quote: \"One dog is in the room\" - end of the quote. The quote indicates that there are no dog in the room." },
    new TextData { Text = "WASHINGTON. The nation’s governors said Saturday that passage of a $787 billion bill to stimulate the economy might help them avert draconian budget cuts, but that they did not expect to see signs of an economic recovery until late this year or early 2010. The officials, arriving here for the winter meeting of the National Governors Association, said that state revenues were coming in far below their projections and that the new federal measure, while helpful, would not be a panacea. Gov. Jon Huntsman Jr. of Utah, where the economy is better than in most states, said the revenue figures were \"\"still dismal.\"\" Asked when the recovery would start, Mr. Huntsman, a Republican, said: \"\"We were hoping in the fourth quarter of this year. Gov. Steven L. Beshear of Kentucky, a Democrat, said: “If the experts are correct, next year may be even worse than this year. I think very probably they are correct. [SEP] The text is positive" },
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
