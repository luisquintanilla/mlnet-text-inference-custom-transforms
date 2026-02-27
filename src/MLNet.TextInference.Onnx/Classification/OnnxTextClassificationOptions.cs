namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the full ONNX text classification pipeline.
/// </summary>
public class OnnxTextClassificationOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; set; }

    /// <summary>
    /// Path to tokenizer artifacts. Can be a directory containing tokenizer_config.json,
    /// a tokenizer_config.json file, or a direct vocab file (.txt, .model).
    /// </summary>
    public required string TokenizerPath { get; set; }

    /// <summary>Name of the input text column in the IDataView. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output column for class probabilities. Default: "Probabilities".</summary>
    public string ProbabilitiesColumnName { get; set; } = "Probabilities";

    /// <summary>Name of the output column for the predicted label. Default: "PredictedLabel".</summary>
    public string PredictedLabelColumnName { get; set; } = "PredictedLabel";

    /// <summary>Optional class labels. If null, predicted labels are numeric indices.</summary>
    public string[]? Labels { get; set; }

    /// <summary>Maximum number of tokens per input text. Default: 128.</summary>
    public int MaxTokenLength { get; set; } = 128;

    /// <summary>Batch size for ONNX inference. Default: 32.</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Optional GPU device ID to run execution on. Null = CPU.
    /// Requires the consuming application to reference Microsoft.ML.OnnxRuntime.Gpu.
    /// </summary>
    public int? GpuDeviceId { get; set; }

    /// <summary>
    /// If true and GPU initialization fails, fall back to CPU instead of throwing.
    /// Default: false.
    /// </summary>
    public bool FallbackToCpu { get; set; }
}
