using Microsoft.ML;
using MLNet.TextInference.Onnx;

// Paths — download model from https://huggingface.co/deepset/minilm-uncased-squad2 (ONNX export)
var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models"));

Console.WriteLine("=== Extractive QA (deepset/minilm-uncased-squad2) ===\n");
Console.WriteLine("Lightweight fast QA model — ideal for low-latency scenarios.\n");

var mlContext = new MLContext();

var qaOptions = new OnnxQaOptions
{
    ModelPath = modelPath,
    TokenizerPath = tokenizerPath,
    MaxTokenLength = 384,
    MaxAnswerLength = 30,
    BatchSize = 8
};

var estimator = mlContext.Transforms.OnnxQa(qaOptions);

var sampleData = new[]
{
    new QaInput
    {
        Question = "When was the Eiffel Tower built?",
        Context = "The Eiffel Tower is a wrought-iron lattice tower in Paris. It was constructed from 1887 to 1889."
    },
    new QaInput
    {
        Question = "What programming language is Python named after?",
        Context = "Python is a high-level programming language. It was named after the BBC comedy show Monty Python's Flying Circus."
    },
    // Unanswerable
    new QaInput
    {
        Question = "What color is the sky on Venus?",
        Context = "Venus is the second planet from the Sun. It is the hottest planet in our solar system."
    }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

Console.WriteLine("Fitting QA pipeline...");
var transformer = estimator.Fit(dataView);

// Direct API
Console.WriteLine("\nDirect API results:");
Console.WriteLine(new string('-', 40));

var questions = sampleData.Select(s => s.Question).ToList();
var contexts = sampleData.Select(s => s.Context).ToList();
var answers = transformer.Answer(questions, contexts);

for (int i = 0; i < questions.Count; i++)
{
    Console.WriteLine($"\n  Q: \"{questions[i]}\"");
    Console.WriteLine($"  Context: \"{contexts[i]}\"");
    if (answers[i].Answer.Length > 0)
        Console.WriteLine($"  Answer: \"{answers[i].Answer}\" (score: {answers[i].Score:F4})");
    else
        Console.WriteLine($"  Answer: <unanswerable> (score: {answers[i].Score:F4})");
}

Console.WriteLine("\nDone!");
transformer.Dispose();

public class QaInput
{
    public string Question { get; set; } = "";
    public string Context { get; set; } = "";
}
