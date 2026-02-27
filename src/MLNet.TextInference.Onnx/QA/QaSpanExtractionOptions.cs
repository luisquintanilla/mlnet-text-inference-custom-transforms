namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the QA span extraction post-processing transform.
/// Finds best answer spans from start/end logits produced by a QA model.
/// </summary>
public class QaSpanExtractionOptions
{
    /// <summary>Name of the start logits column. Default: "StartLogits".</summary>
    public string StartLogitsColumnName { get; set; } = "StartLogits";

    /// <summary>Name of the end logits column. Default: "EndLogits".</summary>
    public string EndLogitsColumnName { get; set; } = "EndLogits";

    /// <summary>Name of the output answer text column. Default: "Answer".</summary>
    public string OutputColumnName { get; set; } = "Answer";

    /// <summary>Name of the output answer score column. Default: "AnswerScore".</summary>
    public string ScoreColumnName { get; set; } = "AnswerScore";

    /// <summary>Name of the original context text column. Default: "Text".</summary>
    public string TextColumnName { get; set; } = "Text";

    /// <summary>Name of the token start offsets column. Default: "TokenStartOffsets".</summary>
    public string TokenStartOffsetsColumnName { get; set; } = "TokenStartOffsets";

    /// <summary>Name of the token end offsets column. Default: "TokenEndOffsets".</summary>
    public string TokenEndOffsetsColumnName { get; set; } = "TokenEndOffsets";

    /// <summary>Name of the attention mask column. Default: "AttentionMask".</summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>Maximum number of tokens in an answer span. Default: 30.</summary>
    public int MaxAnswerLength { get; set; } = 30;

    /// <summary>Number of top answer candidates to consider. Default: 1.</summary>
    public int TopK { get; set; } = 1;
}
