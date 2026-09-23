using System.Text.Json.Serialization;

namespace MLNet.TextInference.TypedDecisions;

/// <summary>Question kinds understood by the Laya decision head.</summary>
public enum DecisionQuestionType
{
    Choice = 0,
    Score = 1,
    Noul = 2
}

/// <summary>Optional text used to explain the true and false sides of a noul question.</summary>
public sealed record NoulCriteria(
    string? True = null,
    string? False = null);

/// <summary>
/// A typed question. State and instructions are deliberately strings so callers can provide
/// plain text or pre-serialized JSON without an implicit, Python-specific object serializer.
/// </summary>
public sealed record DecisionQuestion
{
    public required string Id { get; init; }
    public required DecisionQuestionType Type { get; init; }
    public required string Instructions { get; init; }
    public IReadOnlyDictionary<string, string?>? Choices { get; init; }
    public IReadOnlyList<string>? ScoreLevels { get; init; }
    public NoulCriteria? NoulCriteria { get; init; }

    public static DecisionQuestion Choice(
        string id,
        string instructions,
        IReadOnlyDictionary<string, string?> choices)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        ArgumentNullException.ThrowIfNull(choices);
        return new DecisionQuestion
        {
            Id = id,
            Type = DecisionQuestionType.Choice,
            Instructions = instructions ?? throw new ArgumentNullException(nameof(instructions)),
            Choices = choices
        };
    }

    public static DecisionQuestion Choice(
        string id,
        string instructions,
        IEnumerable<string> choices)
    {
        ArgumentNullException.ThrowIfNull(choices);
        return Choice(id, instructions,
            choices.ToDictionary(static c => c, static _ => (string?)null, StringComparer.Ordinal));
    }

    public static DecisionQuestion Score(
        string id,
        string instructions,
        IReadOnlyList<string> levels)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        ArgumentNullException.ThrowIfNull(levels);
        return new DecisionQuestion
        {
            Id = id,
            Type = DecisionQuestionType.Score,
            Instructions = instructions ?? throw new ArgumentNullException(nameof(instructions)),
            ScoreLevels = levels
        };
    }

    public static DecisionQuestion Noul(
        string id,
        string instructions,
        NoulCriteria? criteria = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        return new DecisionQuestion
        {
            Id = id,
            Type = DecisionQuestionType.Noul,
            Instructions = instructions ?? throw new ArgumentNullException(nameof(instructions)),
            NoulCriteria = criteria
        };
    }

    internal IReadOnlyList<string> RenderOptions()
    {
        return Type switch
        {
            DecisionQuestionType.Choice => (Choices ?? throw new InvalidOperationException(
                $"Choice question '{Id}' must define Choices.")).Select(static p =>
                    string.IsNullOrEmpty(p.Value) ? p.Key : $"{p.Key}: {p.Value}").ToArray(),
            DecisionQuestionType.Score => (ScoreLevels ?? throw new InvalidOperationException(
                $"Score question '{Id}' must define ScoreLevels.")).Select(
                    static (level, index) => $"level {index}: {level}").ToArray(),
            DecisionQuestionType.Noul =>
                [$"false: {NoulCriteria?.False ?? "no, the statement does not hold"}",
                 $"true: {NoulCriteria?.True ?? "yes, the statement holds"}"],
            _ => throw new ArgumentOutOfRangeException()
        };
    }

    internal IReadOnlyList<string> OptionLabels()
    {
        return Type switch
        {
            DecisionQuestionType.Choice => (Choices ?? throw new InvalidOperationException(
                $"Choice question '{Id}' must define Choices.")).Keys.ToArray(),
            DecisionQuestionType.Score => Enumerable.Range(0, (ScoreLevels ?? throw new InvalidOperationException(
                $"Score question '{Id}' must define ScoreLevels.")).Count).Select(static i => i.ToString()).ToArray(),
            DecisionQuestionType.Noul => ["false", "true"],
            _ => throw new ArgumentOutOfRangeException()
        };
    }

    public void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(Id);
        ArgumentNullException.ThrowIfNull(Instructions);
        var count = RenderOptions().Count;
        if (Type is DecisionQuestionType.Choice or DecisionQuestionType.Score && count < 2)
            throw new ArgumentException(
                $"Question '{Id}' must have at least two options in v1; single-option {Type} questions are not supported.");
    }
}

/// <summary>A state and one or more typed questions to evaluate.</summary>
public sealed record DecisionRequest
{
    public required string State { get; init; }
    public required IReadOnlyList<DecisionQuestion> Questions { get; init; }

    public static DecisionRequest Create(string state, IReadOnlyList<DecisionQuestion> questions)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(questions);
        if (questions.Count == 0)
            throw new ArgumentException("At least one question is required.", nameof(questions));
        foreach (var question in questions)
            question.Validate();
        return new DecisionRequest { State = state, Questions = questions };
    }
}

/// <summary>Per-question inputs after tokenization and marker placement.</summary>
public sealed record DecisionInputItem(
    int RequestIndex,
    DecisionQuestion Question,
    int[] MarkerPositions,
    string[] OptionLabels,
    long QuestionType);

/// <summary>
/// Dense, fixed-shape inputs for one ONNX call. Arrays are row-major and match the documented
/// model shapes: [B,L], [B,L], [B,K], [B,K], and [B].
/// </summary>
public sealed class DecisionInputBatch
{
    public required int BatchSize { get; init; }
    public required int SequenceLength { get; init; }
    public required int MarkerWidth { get; init; }
    public required long[] InputIds { get; init; }
    public required long[] AttentionMask { get; init; }
    public required long[] MarkerPositions { get; init; }
    public required bool[] MarkerMask { get; init; }
    public required long[] QuestionTypes { get; init; }
    public required IReadOnlyList<DecisionInputItem> Items { get; init; }

    public void Validate()
    {
        if (BatchSize <= 0 || SequenceLength <= 0 || MarkerWidth <= 0)
            throw new InvalidOperationException("Decision input dimensions must be positive.");
        if (InputIds is null || AttentionMask is null ||
            MarkerPositions is null || MarkerMask is null ||
            QuestionTypes is null || Items is null ||
            InputIds.Length != BatchSize * SequenceLength ||
            AttentionMask.Length != BatchSize * SequenceLength ||
            MarkerPositions.Length != BatchSize * MarkerWidth ||
            MarkerMask.Length != BatchSize * MarkerWidth ||
            QuestionTypes.Length != BatchSize ||
            Items.Count != BatchSize)
        {
            throw new InvalidOperationException("Decision input arrays do not match their declared shapes.");
        }
    }
}

/// <summary>Outputs from the decision ONNX graph.</summary>
public sealed class DecisionModelOutputs
{
    public required int BatchSize { get; init; }
    public required int MarkerWidth { get; init; }
    public required float[] Logits { get; init; }
    public required float[] ActionProbabilities { get; init; }

    public void Validate()
    {
        if (BatchSize <= 0 || MarkerWidth <= 0 ||
            Logits is null || ActionProbabilities is null ||
            Logits.Length != BatchSize * MarkerWidth ||
            ActionProbabilities.Length != BatchSize * 2)
        {
            throw new InvalidOperationException(
                "Decision model outputs must have logits [B,K] and act_probs [B,2].");
        }

        if (Logits.Any(static value => !float.IsFinite(value)))
            throw new InvalidOperationException("Decision model logits must be finite.");
        if (ActionProbabilities.Any(static value =>
                !float.IsFinite(value) || value < 0 || value > 1))
        {
            throw new InvalidOperationException(
                "Decision model act_probs values must be finite probabilities in [0,1].");
        }
    }
}

public abstract record DecisionResult(
    string Id,
    DecisionQuestionType Type,
    DecisionDistribution Distribution,
    float Confidence,
    float ActionProbability);

public sealed record ChoiceDecisionResult(
    string Id,
    string Choice,
    DecisionDistribution Distribution,
    float Confidence,
    float ActionProbability)
    : DecisionResult(Id, DecisionQuestionType.Choice, Distribution, Confidence, ActionProbability);

public sealed record ScoreDecisionResult(
    string Id,
    float Score,
    IReadOnlyDictionary<string, string> Legend,
    DecisionDistribution Distribution,
    float Confidence,
    float ActionProbability)
    : DecisionResult(Id, DecisionQuestionType.Score, Distribution, Confidence, ActionProbability);

public sealed record NoulDecisionResult(
    string Id,
    bool Value,
    float ProbabilityTrue,
    DecisionDistribution Distribution,
    float Confidence,
    float ActionProbability)
    : DecisionResult(Id, DecisionQuestionType.Noul, Distribution, Confidence, ActionProbability);

public sealed record DecisionDistribution(
    IReadOnlyList<string> Labels,
    IReadOnlyList<float> Probabilities);

public sealed class DecisionResponse
{
    public required IReadOnlyList<DecisionResult> Results { get; init; }
    public required int InputTokenCount { get; init; }
}

public sealed record TypedDecisionDiagnostic(
    string Code,
    string Message,
    TypedDecisionDiagnosticSeverity Severity = TypedDecisionDiagnosticSeverity.Warning);

public enum TypedDecisionDiagnosticSeverity
{
    Warning,
    Information
}
