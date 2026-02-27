namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the NER decoding transform.
/// Decodes BIO-tagged model output into entity spans.
/// </summary>
public class NerDecodingOptions
{
    /// <summary>Name of the input column containing raw model logits. Default: "RawOutput".</summary>
    public string InputColumnName { get; set; } = "RawOutput";

    /// <summary>Name of the output column for JSON-serialized entities. Default: "Entities".</summary>
    public string OutputColumnName { get; set; } = "Entities";

    /// <summary>Name of the attention mask column. Default: "AttentionMask".</summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>Name of the original text column. Default: "Text".</summary>
    public string TextColumnName { get; set; } = "Text";

    /// <summary>Name of the token start offsets column. Default: "TokenStartOffsets".</summary>
    public string TokenStartOffsetsColumnName { get; set; } = "TokenStartOffsets";

    /// <summary>Name of the token end offsets column. Default: "TokenEndOffsets".</summary>
    public string TokenEndOffsetsColumnName { get; set; } = "TokenEndOffsets";

    /// <summary>BIO label list matching the model's output layer (e.g. ["O","B-PER","I-PER",...]).</summary>
    public required string[] Labels { get; set; }

    /// <summary>Number of labels. If null, inferred from Labels.Length.</summary>
    public int? NumLabels { get; set; }
}
