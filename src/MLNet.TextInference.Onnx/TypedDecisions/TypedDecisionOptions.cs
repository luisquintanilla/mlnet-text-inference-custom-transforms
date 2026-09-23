using MLNet.TextInference.TypedDecisions;

namespace MLNet.TextInference.Onnx;

public class OnnxTypedDecisionsOptions
{
    public required string BundlePath { get; init; }
    public required IReadOnlyList<DecisionQuestion> Questions { get; init; }
    public string StateColumnName { get; init; } = "State";
    public string ResultsColumnName { get; init; } = "DecisionResults";
    public string ChoiceColumnName { get; init; } = "DecisionChoice";
    public string ScoreColumnName { get; init; } = "DecisionScore";
    public string ProbabilityTrueColumnName { get; init; } = "DecisionProbabilityTrue";
    public string ConfidenceColumnName { get; init; } = "DecisionConfidence";
    public string ActionProbabilityColumnName { get; init; } = "DecisionActionProbability";
    public int BatchSize { get; init; } = 32;

    internal void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(BundlePath);
        ArgumentNullException.ThrowIfNull(Questions);
        if (Questions.Count == 0)
            throw new ArgumentException("At least one question must be configured.", nameof(Questions));
        if (BatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(BatchSize));
        TypedDecisionOptionsValidation.ValidateColumnName(StateColumnName, nameof(StateColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ResultsColumnName, nameof(ResultsColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ChoiceColumnName, nameof(ChoiceColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ScoreColumnName, nameof(ScoreColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ProbabilityTrueColumnName, nameof(ProbabilityTrueColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ConfidenceColumnName, nameof(ConfidenceColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ActionProbabilityColumnName, nameof(ActionProbabilityColumnName));
        foreach (var question in Questions)
            question.Validate();
    }
}

public sealed class DecisionInputPreparationOptions
{
    public required string BundlePath { get; init; }
    public required IReadOnlyList<DecisionQuestion> Questions { get; init; }
    public string StateColumnName { get; init; } = "State";
    public string OutputColumnName { get; init; } = "PreparedDecisionInputs";
    public int BatchSize { get; init; } = 32;

    internal void Validate()
    {
        TypedDecisionOptionsValidation.ValidateBundlePath(BundlePath);
        ArgumentNullException.ThrowIfNull(Questions);
        if (Questions.Count == 0)
            throw new ArgumentException("At least one question must be configured.", nameof(Questions));
        if (BatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(BatchSize));
        TypedDecisionOptionsValidation.ValidateColumnName(StateColumnName, nameof(StateColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(OutputColumnName, nameof(OutputColumnName));
        foreach (var question in Questions)
            question.Validate();
    }
}

public sealed class OnnxDecisionModelScorerOptions
{
    public required string BundlePath { get; init; }
    public string InputColumnName { get; init; } = "PreparedDecisionInputs";
    public string OutputColumnName { get; init; } = "ScoredDecisionOutputs";
    public int BatchSize { get; init; } = 32;

    internal void Validate()
    {
        TypedDecisionOptionsValidation.ValidateBundlePath(BundlePath);
        if (BatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(BatchSize));
        TypedDecisionOptionsValidation.ValidateColumnName(InputColumnName, nameof(InputColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(OutputColumnName, nameof(OutputColumnName));
    }
}

public sealed class DecisionDecodingOptions
{
    public required string BundlePath { get; init; }
    public string InputColumnName { get; init; } = "ScoredDecisionOutputs";
    public string ResultsColumnName { get; init; } = "DecisionResults";
    public string ChoiceColumnName { get; init; } = "DecisionChoice";
    public string ScoreColumnName { get; init; } = "DecisionScore";
    public string ProbabilityTrueColumnName { get; init; } = "DecisionProbabilityTrue";
    public string ConfidenceColumnName { get; init; } = "DecisionConfidence";
    public string ActionProbabilityColumnName { get; init; } = "DecisionActionProbability";

    internal void Validate()
    {
        TypedDecisionOptionsValidation.ValidateBundlePath(BundlePath);
        TypedDecisionOptionsValidation.ValidateColumnName(InputColumnName, nameof(InputColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ResultsColumnName, nameof(ResultsColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ChoiceColumnName, nameof(ChoiceColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ScoreColumnName, nameof(ScoreColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ProbabilityTrueColumnName, nameof(ProbabilityTrueColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ConfidenceColumnName, nameof(ConfidenceColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(ActionProbabilityColumnName, nameof(ActionProbabilityColumnName));
    }
}

internal static class TypedDecisionOptionsValidation
{
    internal static void ValidateColumnName(string value, string parameterName)
        => ArgumentException.ThrowIfNullOrWhiteSpace(value, parameterName);

    internal static void ValidateBundlePath(string value)
        => ArgumentException.ThrowIfNullOrWhiteSpace(value, nameof(value));
}
