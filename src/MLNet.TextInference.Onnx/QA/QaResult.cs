namespace MLNet.TextInference.Onnx;

/// <summary>
/// Represents a single answer extracted from a QA model.
/// </summary>
public sealed class QaResult
{
    /// <summary>The answer text extracted from the context. Empty if unanswerable.</summary>
    public string Answer { get; init; } = "";

    /// <summary>Combined start + end logit score for this answer span.</summary>
    public float Score { get; init; }

    /// <summary>Start character offset in the context text.</summary>
    public int StartChar { get; init; }

    /// <summary>End character offset in the context text (exclusive).</summary>
    public int EndChar { get; init; }
}
