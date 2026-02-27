using Microsoft.ML;

namespace MLNet.TextGeneration.OnnxGenAI;

/// <summary>
/// Extension methods for MLContext to provide a convenient API for ONNX GenAI text generation.
/// </summary>
public static class MLContextExtensions
{
    /// <summary>
    /// Creates an estimator that generates text using a local ONNX Runtime GenAI model.
    /// </summary>
    public static OnnxTextGenerationEstimator OnnxTextGeneration(
        this TransformsCatalog catalog,
        OnnxTextGenerationOptions options)
    {
        return new OnnxTextGenerationEstimator(GetMLContext(catalog), options);
    }

    private static MLContext GetMLContext(TransformsCatalog catalog)
    {
        var envProperty = typeof(TransformsCatalog)
            .GetProperty("Environment", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (envProperty?.GetValue(catalog) is MLContext mlContext)
            return mlContext;
        return new MLContext();
    }
}
