using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.TextInference.Onnx;

var mode = GetOption(args, "--mode") ?? "facade";
var bundlePath = GetOption(args, "--bundle");
if (string.IsNullOrWhiteSpace(bundlePath))
{
    Console.Error.WriteLine("A local --bundle path is required. Inference never downloads model assets.");
    return 2;
}

var questions = new[]
{
    MLNet.TextInference.TypedDecisions.DecisionQuestion.Choice(
        "priority", "How urgent is the request?", new[] { "low", "high" }),
    MLNet.TextInference.TypedDecisions.DecisionQuestion.Score(
        "quality", "How strong is the evidence?", new[] { "weak", "moderate", "strong" }),
    MLNet.TextInference.TypedDecisions.DecisionQuestion.Noul(
        "actionable", "Can the request be acted on now?")
};
var options = new OnnxTypedDecisionsOptions
{
    BundlePath = bundlePath,
    Questions = questions,
    BatchSize = 16
};

var ml = new MLContext(seed: 1);
var data = ml.Data.LoadFromEnumerable(new[]
{
    new StateRow
    {
        State = "The customer supplied reproducible steps and requested an urgent fix."
    },
    new StateRow
    {
        State = "The report is missing logs and has no clear requested action."
    }
});

IDataView output;
switch (mode.ToLowerInvariant())
{
    case "facade":
        output = ml.Transforms.OnnxTypedDecisions(options).Fit(data).Transform(data);
        break;

    case "stages":
        var prepared = new PrepareDecisionInputsOptions
        {
            BundlePath = bundlePath,
            Questions = questions
        };
        var scored = new ScoreOnnxDecisionModelOptions { BundlePath = bundlePath };
        var decoded = new DecodeDecisionsOptions { BundlePath = bundlePath };
        var stages = ml.Transforms.PrepareDecisionInputs(prepared)
            .Append(ml.Transforms.ScoreOnnxDecisionModel(scored))
            .Append(ml.Transforms.DecodeDecisions(decoded));
        output = stages.Fit(data).Transform(data);
        break;

    case "composed":
        var appendedOptions = new OnnxTypedDecisionsOptions
        {
            BundlePath = options.BundlePath,
            Questions = options.Questions,
            StateColumnName = options.StateColumnName,
            ResultsColumnName = "AppendedDecisionResults",
            ChoiceColumnName = "AppendedDecisionChoice",
            ScoreColumnName = "AppendedDecisionScore",
            ProbabilityTrueColumnName = "AppendedDecisionProbabilityTrue",
            ConfidenceColumnName = "AppendedDecisionConfidence",
            ActionProbabilityColumnName = "AppendedDecisionActionProbability",
            BatchSize = options.BatchSize
        };
        IEstimator<ITransformer> pipeline = ml.Transforms.OnnxTypedDecisions(options);
        output = pipeline.AppendOnnxTypedDecisions(ml, appendedOptions).Fit(data).Transform(data);
        break;

    default:
        Console.Error.WriteLine($"Unknown mode '{mode}'. Use facade, stages, or composed.");
        return 2;
}

foreach (var row in ml.Data.CreateEnumerable<DecisionRow>(output, reuseRowObject: false))
{
    Console.WriteLine($"choice={row.DecisionChoice}; score={row.DecisionScore}; " +
        $"true_probability={row.DecisionProbabilityTrue}; confidence={row.DecisionConfidence}");
    Console.WriteLine(row.DecisionResults);
}

return 0;

static string? GetOption(string[] args, string name)
{
    var index = Array.IndexOf(args, name);
    return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
}

public sealed class StateRow
{
    public string State { get; set; } = string.Empty;
}

public sealed class DecisionRow
{
    public string DecisionResults { get; set; } = string.Empty;
    public string DecisionChoice { get; set; } = string.Empty;
    public float DecisionScore { get; set; }
    public float DecisionProbabilityTrue { get; set; }
    public float DecisionConfidence { get; set; }
    public float DecisionActionProbability { get; set; }
}
