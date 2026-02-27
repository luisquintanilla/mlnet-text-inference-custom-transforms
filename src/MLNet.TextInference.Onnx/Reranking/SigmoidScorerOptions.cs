namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the sigmoid scorer post-processing transform.
/// Applies sigmoid to raw logits from a cross-encoder model.
/// </summary>
public class SigmoidScorerOptions
{
    /// <summary>Name of the input column containing raw model output (float[]). Default: "RawOutput".</summary>
    public string InputColumnName { get; set; } = "RawOutput";

    /// <summary>Name of the output score column (single float). Default: "Score".</summary>
    public string OutputColumnName { get; set; } = "Score";
}
