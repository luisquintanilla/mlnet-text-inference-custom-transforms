using Microsoft.ML;
using MLNet.TextInference.Onnx;

// Paths — download model from https://huggingface.co/deepset/roberta-base-squad2 (ONNX export)
var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models"));

Console.WriteLine("=== Extractive QA (deepset/roberta-base-squad2) ===\n");

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

// Sample data: answerable + unanswerable questions
var sampleData = new[]
{
    new QaInput
    {
        Question = "What is the capital of France?",
        Context = "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower."
    },
    new QaInput
    {
        Question = "Who founded Microsoft?",
        Context = "Microsoft was founded by Bill Gates and Paul Allen in 1975 in Albuquerque, New Mexico."
    },
    // Unanswerable question (SQuAD 2.0)
    new QaInput
    {
        Question = "What is the population of Mars?",
        Context = "Mars is the fourth planet from the Sun. It has a thin atmosphere composed mostly of carbon dioxide."
    }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

Console.WriteLine("Fitting QA pipeline...");
var transformer = estimator.Fit(dataView);

// --- Direct API ---
Console.WriteLine("\n1. Direct API results:");
Console.WriteLine(new string('-', 40));

var questions = sampleData.Select(s => s.Question).ToList();
var contexts = sampleData.Select(s => s.Context).ToList();
var answers = transformer.Answer(questions, contexts);

for (int i = 0; i < questions.Count; i++)
{
    Console.WriteLine($"\n  Q: \"{questions[i]}\"");
    Console.WriteLine($"  Context: \"{contexts[i]}\"");
    if (answers[i].Answer.Length > 0)
        Console.WriteLine($"  Answer: \"{answers[i].Answer}\" (score: {answers[i].Score:F4}, chars [{answers[i].StartChar}..{answers[i].EndChar}])");
    else
        Console.WriteLine($"  Answer: <unanswerable> (score: {answers[i].Score:F4})");
}

// --- ML.NET Pipeline ---
Console.WriteLine("\n\n2. ML.NET Pipeline (IDataView):");
Console.WriteLine(new string('-', 40));

var result = transformer.Transform(dataView);
var rows = mlContext.Data.CreateEnumerable<QaOutput>(result, reuseRowObject: false).ToList();

for (int i = 0; i < rows.Count; i++)
{
    Console.WriteLine($"  Q: \"{sampleData[i].Question}\"");
    Console.WriteLine($"  Answer: \"{rows[i].Answer}\" (score: {rows[i].AnswerScore:F4})");
}

Console.WriteLine("\nDone!");
transformer.Dispose();

public class QaInput
{
    public string Question { get; set; } = "";
    public string Context { get; set; } = "";
}

public class QaOutput
{
    public string Question { get; set; } = "";
    public string Context { get; set; } = "";
    public string Answer { get; set; } = "";
    public float AnswerScore { get; set; }
}
