namespace MLNet.TextInference.Onnx;

/// <summary>
/// Represents a single named entity extracted from text.
/// </summary>
public sealed class NerEntity
{
    /// <summary>Entity type (e.g. "PER", "ORG", "LOC").</summary>
    public string EntityType { get; init; } = "";

    /// <summary>Surface form of the entity in the original text.</summary>
    public string Word { get; init; } = "";

    /// <summary>Start character offset in the original text.</summary>
    public int StartChar { get; init; }

    /// <summary>End character offset in the original text (exclusive).</summary>
    public int EndChar { get; init; }

    /// <summary>Confidence score (softmax probability).</summary>
    public float Score { get; init; }
}
