using MLNet.TextInference.TypedDecisions;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the ML.NET typed-decision transform.
/// </summary>
public sealed class OnnxTypedDecisionsOptions
{
    /// <summary>
    /// Path to a local Laya model-assets directory or optional archive. A directory
    /// does not require a generated manifest.
    /// </summary>
    public required string ModelAssetsPath { get; init; }

    public required IReadOnlyList<DecisionQuestion> Questions { get; init; }
    public string StateColumnName { get; init; } = "State";
    public string OutputPrefix { get; init; } = "Decision_";
    public string ResultsColumnName { get; init; } = "DecisionResults";
    public int BatchSize { get; init; } = 32;

    internal DecisionOutputNames OutputNames => DecisionOutputNames.Create(OutputPrefix, Questions);

    internal void Validate()
    {
        TypedDecisionOptionsValidation.ValidateModelAssetsPath(ModelAssetsPath);
        ArgumentNullException.ThrowIfNull(Questions);
        if (Questions.Count == 0)
            throw new ArgumentException("At least one question must be configured.", nameof(Questions));
        TypedDecisionOptionsValidation.ValidateQuestionIds(Questions);
        if (BatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(BatchSize));
        TypedDecisionOptionsValidation.ValidateColumnName(StateColumnName, nameof(StateColumnName));
        TypedDecisionOptionsValidation.ValidateColumnName(OutputPrefix, nameof(OutputPrefix));
        TypedDecisionOptionsValidation.ValidateColumnName(ResultsColumnName, nameof(ResultsColumnName));
        foreach (var question in Questions)
            question.Validate();
        OutputNames.ValidateUnique(ResultsColumnName);
    }
}

/// <summary>Native tensor columns emitted by the preparation stage.</summary>
public sealed class DecisionInputPreparationOptions
{
    /// <summary>Path to a local Laya model-assets directory or optional archive.</summary>
    public required string ModelAssetsPath { get; init; }
    public required IReadOnlyList<DecisionQuestion> Questions { get; init; }
    public string StateColumnName { get; init; } = "State";
    public string InputIdsColumnName { get; init; } = "DecisionInputIds";
    public string AttentionMaskColumnName { get; init; } = "DecisionAttentionMask";
    public string MarkerPositionsColumnName { get; init; } = "DecisionMarkerPositions";
    public string MarkerMaskColumnName { get; init; } = "DecisionMarkerMask";
    public string QuestionTypesColumnName { get; init; } = "DecisionQuestionTypes";
    public string BatchSizeColumnName { get; init; } = "DecisionBatchSize";
    public string SequenceLengthColumnName { get; init; } = "DecisionSequenceLength";
    public string MarkerWidthColumnName { get; init; } = "DecisionMarkerWidth";

    internal void Validate()
    {
        TypedDecisionOptionsValidation.ValidateModelAssetsPath(ModelAssetsPath);
        ArgumentNullException.ThrowIfNull(Questions);
        if (Questions.Count == 0)
            throw new ArgumentException("At least one question must be configured.", nameof(Questions));
        TypedDecisionOptionsValidation.ValidateQuestionIds(Questions);
        TypedDecisionOptionsValidation.ValidateColumnName(StateColumnName, nameof(StateColumnName));
        foreach (var (name, parameter) in ColumnNames())
            TypedDecisionOptionsValidation.ValidateColumnName(name, parameter);
        TypedDecisionOptionsValidation.ValidateDistinctColumnNames(
            ColumnNames().Select(static item => item.Name).Append(StateColumnName));
        foreach (var question in Questions)
            question.Validate();
    }

    internal IEnumerable<(string Name, string Parameter)> ColumnNames()
    {
        yield return (InputIdsColumnName, nameof(InputIdsColumnName));
        yield return (AttentionMaskColumnName, nameof(AttentionMaskColumnName));
        yield return (MarkerPositionsColumnName, nameof(MarkerPositionsColumnName));
        yield return (MarkerMaskColumnName, nameof(MarkerMaskColumnName));
        yield return (QuestionTypesColumnName, nameof(QuestionTypesColumnName));
        yield return (BatchSizeColumnName, nameof(BatchSizeColumnName));
        yield return (SequenceLengthColumnName, nameof(SequenceLengthColumnName));
        yield return (MarkerWidthColumnName, nameof(MarkerWidthColumnName));
    }
}

/// <summary>Native tensor columns consumed and produced by the scoring stage.</summary>
public sealed class OnnxDecisionModelScorerOptions
{
    /// <summary>Path to a local Laya model-assets directory or optional archive.</summary>
    public required string ModelAssetsPath { get; init; }
    public string InputIdsColumnName { get; init; } = "DecisionInputIds";
    public string AttentionMaskColumnName { get; init; } = "DecisionAttentionMask";
    public string MarkerPositionsColumnName { get; init; } = "DecisionMarkerPositions";
    public string MarkerMaskColumnName { get; init; } = "DecisionMarkerMask";
    public string QuestionTypesColumnName { get; init; } = "DecisionQuestionTypes";
    public string BatchSizeColumnName { get; init; } = "DecisionBatchSize";
    public string SequenceLengthColumnName { get; init; } = "DecisionSequenceLength";
    public string MarkerWidthColumnName { get; init; } = "DecisionMarkerWidth";
    public string LogitsColumnName { get; init; } = "DecisionLogits";
    public string ActionProbabilitiesColumnName { get; init; } = "DecisionActionProbabilities";
    public int BatchSize { get; init; } = 32;

    internal void Validate()
    {
        TypedDecisionOptionsValidation.ValidateModelAssetsPath(ModelAssetsPath);
        if (BatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(BatchSize));
        foreach (var (name, parameter) in ColumnNames())
            TypedDecisionOptionsValidation.ValidateColumnName(name, parameter);
    }

    internal IEnumerable<(string Name, string Parameter)> ColumnNames()
    {
        yield return (InputIdsColumnName, nameof(InputIdsColumnName));
        yield return (AttentionMaskColumnName, nameof(AttentionMaskColumnName));
        yield return (MarkerPositionsColumnName, nameof(MarkerPositionsColumnName));
        yield return (MarkerMaskColumnName, nameof(MarkerMaskColumnName));
        yield return (QuestionTypesColumnName, nameof(QuestionTypesColumnName));
        yield return (BatchSizeColumnName, nameof(BatchSizeColumnName));
        yield return (SequenceLengthColumnName, nameof(SequenceLengthColumnName));
        yield return (MarkerWidthColumnName, nameof(MarkerWidthColumnName));
        yield return (LogitsColumnName, nameof(LogitsColumnName));
        yield return (ActionProbabilitiesColumnName, nameof(ActionProbabilitiesColumnName));
    }
}

/// <summary>Native scored columns and per-question decoded output configuration.</summary>
public sealed class DecisionDecodingOptions
{
    /// <summary>Path to a local Laya model-assets directory or optional archive.</summary>
    public required string ModelAssetsPath { get; init; }
    public required IReadOnlyList<DecisionQuestion> Questions { get; init; }
    public string InputIdsColumnName { get; init; } = "DecisionInputIds";
    public string AttentionMaskColumnName { get; init; } = "DecisionAttentionMask";
    public string MarkerPositionsColumnName { get; init; } = "DecisionMarkerPositions";
    public string MarkerMaskColumnName { get; init; } = "DecisionMarkerMask";
    public string QuestionTypesColumnName { get; init; } = "DecisionQuestionTypes";
    public string BatchSizeColumnName { get; init; } = "DecisionBatchSize";
    public string SequenceLengthColumnName { get; init; } = "DecisionSequenceLength";
    public string MarkerWidthColumnName { get; init; } = "DecisionMarkerWidth";
    public string LogitsColumnName { get; init; } = "DecisionLogits";
    public string ActionProbabilitiesColumnName { get; init; } = "DecisionActionProbabilities";
    public string OutputPrefix { get; init; } = "Decision_";
    public string ResultsColumnName { get; init; } = "DecisionResults";

    internal DecisionOutputNames OutputNames => DecisionOutputNames.Create(OutputPrefix, Questions);

    internal void Validate()
    {
        TypedDecisionOptionsValidation.ValidateModelAssetsPath(ModelAssetsPath);
        ArgumentNullException.ThrowIfNull(Questions);
        if (Questions.Count == 0)
            throw new ArgumentException("At least one question must be configured.", nameof(Questions));
        TypedDecisionOptionsValidation.ValidateQuestionIds(Questions);
        foreach (var question in Questions)
            question.Validate();
        TypedDecisionOptionsValidation.ValidateColumnName(OutputPrefix, nameof(OutputPrefix));
        TypedDecisionOptionsValidation.ValidateColumnName(ResultsColumnName, nameof(ResultsColumnName));
        foreach (var (name, parameter) in InputColumnNames())
            TypedDecisionOptionsValidation.ValidateColumnName(name, parameter);
        TypedDecisionOptionsValidation.ValidateDistinctColumnNames(
            InputColumnNames().Select(static item => item.Name)
                .Append(OutputPrefix)
                .Append(ResultsColumnName));
        OutputNames.ValidateUnique(ResultsColumnName);
    }

    internal IEnumerable<(string Name, string Parameter)> InputColumnNames()
    {
        yield return (InputIdsColumnName, nameof(InputIdsColumnName));
        yield return (AttentionMaskColumnName, nameof(AttentionMaskColumnName));
        yield return (MarkerPositionsColumnName, nameof(MarkerPositionsColumnName));
        yield return (MarkerMaskColumnName, nameof(MarkerMaskColumnName));
        yield return (QuestionTypesColumnName, nameof(QuestionTypesColumnName));
        yield return (BatchSizeColumnName, nameof(BatchSizeColumnName));
        yield return (SequenceLengthColumnName, nameof(SequenceLengthColumnName));
        yield return (MarkerWidthColumnName, nameof(MarkerWidthColumnName));
        yield return (LogitsColumnName, nameof(LogitsColumnName));
        yield return (ActionProbabilitiesColumnName, nameof(ActionProbabilitiesColumnName));
    }
}

internal sealed record DecisionOutputNames(
    IReadOnlyDictionary<string, QuestionOutputNames> ById)
{
    internal static DecisionOutputNames Create(
        string prefix,
        IReadOnlyList<DecisionQuestion> questions)
    {
        var result = new Dictionary<string, QuestionOutputNames>(StringComparer.Ordinal);
        foreach (var question in questions)
        {
            var name = $"{prefix}{question.Id}_";
            result.Add(question.Id, question.Type switch
            {
                DecisionQuestionType.Choice => new QuestionOutputNames(
                    $"{name}PredictedLabel",
                    null,
                    null,
                    $"{name}Probabilities",
                    $"{name}Confidence",
                    $"{name}ActionProbability"),
                DecisionQuestionType.Score => new QuestionOutputNames(
                    null,
                    $"{name}Score",
                    null,
                    $"{name}Probabilities",
                    $"{name}Confidence",
                    $"{name}ActionProbability"),
                DecisionQuestionType.Noul => new QuestionOutputNames(
                    $"{name}PredictedLabel",
                    null,
                    $"{name}Probability",
                    null,
                    $"{name}Confidence",
                    $"{name}ActionProbability"),
                _ => throw new ArgumentOutOfRangeException()
            });
        }

        return new DecisionOutputNames(result);
    }

    internal void ValidateUnique(string resultsColumn)
    {
        var names = ById.Values.SelectMany(static names => names.All).ToArray();
        if (names.Distinct(StringComparer.Ordinal).Count() != names.Length)
            throw new ArgumentException("Typed-decision output names must be unique.");
        if (names.Contains(resultsColumn, StringComparer.Ordinal))
            throw new ArgumentException(
                $"The results column '{resultsColumn}' conflicts with a per-question output column.");
    }
}

internal sealed record QuestionOutputNames(
    string? PredictedLabel,
    string? Score,
    string? Probability,
    string? Probabilities,
    string Confidence,
    string ActionProbability)
{
    internal IEnumerable<string> All =>
        new[] { PredictedLabel, Score, Probability, Probabilities, Confidence, ActionProbability }
            .OfType<string>();
}

internal static class TypedDecisionOptionsValidation
{
    internal static void ValidateColumnName(string value, string parameterName)
        => ArgumentException.ThrowIfNullOrWhiteSpace(value, parameterName);

    internal static void ValidateModelAssetsPath(string value)
        => ArgumentException.ThrowIfNullOrWhiteSpace(value, nameof(value));

    internal static void ValidateQuestionIds(IReadOnlyList<DecisionQuestion> questions)
    {
        var ids = questions.Select(static question => question.Id).ToArray();
        if (ids.Distinct(StringComparer.Ordinal).Count() != ids.Length)
            throw new ArgumentException("Question IDs must be unique.", nameof(questions));
    }

    internal static void ValidateDistinctColumnNames(IEnumerable<string> names)
    {
        var values = names.ToArray();
        if (values.Distinct(StringComparer.Ordinal).Count() != values.Length)
            throw new ArgumentException("Configured typed-decision column names must be unique.");
    }
}
