namespace MLNet.TextInference.Onnx;

/// <summary>
/// Result of classifying a single text input.
/// </summary>
public sealed class ClassificationResult
{
    /// <summary>The predicted class label (or numeric index if no labels were provided).</summary>
    public string PredictedLabel { get; init; } = "";

    /// <summary>Softmax probabilities for each class.</summary>
    public float[] Probabilities { get; init; } = [];

    /// <summary>Confidence score (max probability).</summary>
    public float Confidence => Probabilities.Length > 0 ? Probabilities.Max() : 0;
}
