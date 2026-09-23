// This is a .NET 10 file-based app. The project directive keeps the sample on the
// same source and package surface as the solution without a sample-only project file.
#:project ../../../src/MLNet.TextInference.Onnx/MLNet.TextInference.Onnx.csproj
#:package Microsoft.ML@5.0.0
#:property PublishAot=false

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.TextInference.Onnx;
using MLNet.TextInference.TypedDecisions;
using static MLNet.TextInference.TypedDecisions.DecisionQuestion;

if (args is ["--help"] or ["-h"])
{
    PrintUsage();
    return 0;
}

var mode = GetOption(args, "--mode") ?? "facade";
var modelAssetsPath = GetOption(args, "--model-assets") ?? GetOption(args, "--bundle");
if (string.IsNullOrWhiteSpace(modelAssetsPath))
{
    Console.Error.WriteLine("A local --model-assets path is required. Inference never downloads model assets.");
    return 2;
}

var questions = new[]
{
    Choice("priority", "How urgent is the request?", ["low", "high"]),
    Score("quality", "How strong is the evidence?", ["weak", "moderate", "strong"]),
    Noul("actionable", "Can the request be acted on now?")
};
var states = new[]
{
    "The customer supplied reproducible steps and requested an urgent fix.",
    "The report is missing logs and has no clear requested action."
};
var options = new OnnxTypedDecisionsOptions
{
    ModelAssetsPath = modelAssetsPath,
    Questions = questions,
    BatchSize = 16
};

var ml = new MLContext(seed: 1);
var data = ml.Data.LoadFromEnumerable(states.Select(static state => new StateRow { State = state }));

switch (mode.ToLowerInvariant())
{
    case "direct":
        using (var transformer = ml.Transforms.OnnxTypedDecisions(options).Fit(data))
        {
            foreach (var response in transformer.Infer(states))
                PrintJson(DecisionResponseJson.Serialize(response));
        }
        break;

    case "facade":
        using (var transformer = ml.Transforms.OnnxTypedDecisions(options).Fit(data))
        {
            PrintRows(
                ml.Data.CreateEnumerable<DecisionRow>(
                    transformer.Transform(data),
                    reuseRowObject: false));
        }
        break;

    case "stages":
        var prepared = new DecisionInputPreparationOptions
        {
            ModelAssetsPath = modelAssetsPath,
            Questions = questions
        };
        var scored = new OnnxDecisionModelScorerOptions { ModelAssetsPath = modelAssetsPath };
        var decoded = new DecisionDecodingOptions
        {
            ModelAssetsPath = modelAssetsPath,
            Questions = questions
        };
        var stages = ml.Transforms.PrepareDecisionInputs(prepared)
            .Append(ml.Transforms.ScoreOnnxDecisionModel(scored))
            .Append(ml.Transforms.DecodeDecisions(decoded));
        var stageTransformer = stages.Fit(data);
        try
        {
            PrintStageRows(
                ml.Data.CreateEnumerable<StageRow>(
                    stageTransformer.Transform(data),
                    reuseRowObject: false));
        }
        finally
        {
            (stageTransformer as IDisposable)?.Dispose();
        }
        break;

    case "composed":
        var appendedOptions = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = options.ModelAssetsPath,
            Questions = options.Questions,
            StateColumnName = options.StateColumnName,
            ResultsColumnName = "AppendedDecisionResults",
            OutputPrefix = "AppendedDecision_",
            BatchSize = options.BatchSize
        };
        var composedTransformer = ml.Transforms.OnnxTypedDecisions(options)
            .AppendOnnxTypedDecisions(ml, appendedOptions)
            .Fit(data);
        try
        {
            PrintComposedRows(
                ml.Data.CreateEnumerable<ComposedDecisionRow>(
                    composedTransformer.Transform(data),
                    reuseRowObject: false));
        }
        finally
        {
            (composedTransformer as IDisposable)?.Dispose();
        }
        break;

    default:
        Console.Error.WriteLine($"Unknown mode '{mode}'. Use direct, facade, stages, or composed.");
        return 2;
}

return 0;

static string? GetOption(string[] args, string name)
{
    var index = Array.IndexOf(args, name);
    return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
}

static void PrintRows(IEnumerable<DecisionRow> rows)
{
    foreach (var row in rows)
    {
        PrintScalars(row);
        Console.WriteLine(row.DecisionResults);
    }
}

static void PrintStageRows(IEnumerable<StageRow> rows)
{
    foreach (var row in rows)
    {
        Console.WriteLine(
            $"native_batch={row.DecisionBatchSize}; sequence_length={row.DecisionSequenceLength}; " +
            $"marker_width={row.DecisionMarkerWidth}; input_ids={row.DecisionInputIds.Length}; " +
            $"logits={row.DecisionLogits.Length}; action_probabilities={row.DecisionActionProbabilities.Length}");
        PrintScalars(row);
        Console.WriteLine(row.DecisionResults);
    }
}

static void PrintComposedRows(IEnumerable<ComposedDecisionRow> rows)
{
    foreach (var row in rows)
    {
        PrintScalars(row);
        Console.WriteLine(row.DecisionResults);
        Console.WriteLine(
            $"appended_priority={row.AppendedDecision_priority_PredictedLabel}; " +
            $"appended_quality_score={row.AppendedDecision_quality_Score}; " +
            $"appended_actionable={row.AppendedDecision_actionable_PredictedLabel}; " +
            $"appended_actionable_true_probability={row.AppendedDecision_actionable_Probability}");
        Console.WriteLine(row.AppendedDecisionResults);
    }
}

static void PrintScalars(DecisionRow row)
{
    Console.WriteLine(
        $"priority={row.Decision_priority_PredictedLabel}; " +
        $"quality_score={row.Decision_quality_Score}; " +
        $"actionable={row.Decision_actionable_PredictedLabel}; " +
        $"actionable_true_probability={row.Decision_actionable_Probability}; " +
        $"priority_confidence={row.Decision_priority_Confidence}; " +
        $"quality_confidence={row.Decision_quality_Confidence}; " +
        $"actionable_confidence={row.Decision_actionable_Confidence}; " +
        $"actionable_action_probability={row.Decision_actionable_ActionProbability}");
}

static void PrintJson(string json) => Console.WriteLine(json);

static void PrintUsage()
{
    Console.WriteLine("""
        Usage: dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- --mode <direct|facade|stages|composed> --model-assets <directory-or-zip>
        """);
}

public sealed class StateRow
{
    public string State { get; set; } = string.Empty;
}

public class DecisionRow
{
    public string DecisionResults { get; set; } = string.Empty;
    public string Decision_priority_PredictedLabel { get; set; } = string.Empty;
    public VBuffer<float> Decision_priority_Probabilities { get; set; }
    public float Decision_priority_Confidence { get; set; }
    public float Decision_priority_ActionProbability { get; set; }
    public float Decision_quality_Score { get; set; }
    public VBuffer<float> Decision_quality_Probabilities { get; set; }
    public float Decision_quality_Confidence { get; set; }
    public float Decision_quality_ActionProbability { get; set; }
    public bool Decision_actionable_PredictedLabel { get; set; }
    public float Decision_actionable_Probability { get; set; }
    public float Decision_actionable_Confidence { get; set; }
    public float Decision_actionable_ActionProbability { get; set; }
}

public sealed class StageRow : DecisionRow
{
    public VBuffer<long> DecisionInputIds { get; set; }
    public VBuffer<long> DecisionAttentionMask { get; set; }
    public VBuffer<long> DecisionMarkerPositions { get; set; }
    public VBuffer<bool> DecisionMarkerMask { get; set; }
    public VBuffer<long> DecisionQuestionTypes { get; set; }
    public int DecisionBatchSize { get; set; }
    public int DecisionSequenceLength { get; set; }
    public int DecisionMarkerWidth { get; set; }
    public VBuffer<float> DecisionLogits { get; set; }
    public VBuffer<float> DecisionActionProbabilities { get; set; }
}

public sealed class ComposedDecisionRow : DecisionRow
{
    public string AppendedDecisionResults { get; set; } = string.Empty;
    public string AppendedDecision_priority_PredictedLabel { get; set; } = string.Empty;
    public VBuffer<float> AppendedDecision_priority_Probabilities { get; set; }
    public float AppendedDecision_priority_Confidence { get; set; }
    public float AppendedDecision_priority_ActionProbability { get; set; }
    public float AppendedDecision_quality_Score { get; set; }
    public VBuffer<float> AppendedDecision_quality_Probabilities { get; set; }
    public float AppendedDecision_quality_Confidence { get; set; }
    public float AppendedDecision_quality_ActionProbability { get; set; }
    public bool AppendedDecision_actionable_PredictedLabel { get; set; }
    public float AppendedDecision_actionable_Probability { get; set; }
    public float AppendedDecision_actionable_Confidence { get; set; }
    public float AppendedDecision_actionable_ActionProbability { get; set; }
}
