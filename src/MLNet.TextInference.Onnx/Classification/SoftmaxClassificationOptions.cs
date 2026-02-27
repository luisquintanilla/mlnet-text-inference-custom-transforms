namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the softmax classification post-processing transform.
/// </summary>
public class SoftmaxClassificationOptions
{
    /// <summary>Name of the input column containing raw model logits. Default: "RawOutput".</summary>
    public string InputColumnName { get; set; } = "RawOutput";

    /// <summary>Name of the output column for class probabilities. Default: "Probabilities".</summary>
    public string ProbabilitiesColumnName { get; set; } = "Probabilities";

    /// <summary>Name of the output column for the predicted label. Default: "PredictedLabel".</summary>
    public string PredictedLabelColumnName { get; set; } = "PredictedLabel";

    /// <summary>Optional class labels. If null, predicted labels are numeric indices.</summary>
    public string[]? Labels { get; set; }

    /// <summary>
    /// Number of classes. If null and Labels is provided, inferred from Labels.Length.
    /// Must be set (or inferable) before the transformer can operate.
    /// </summary>
    public int? NumClasses { get; set; }
}
