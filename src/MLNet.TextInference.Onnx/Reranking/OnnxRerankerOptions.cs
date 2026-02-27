namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the ONNX cross-encoder reranker facade.
/// Chains tokenization (text-pair) → ONNX inference → sigmoid scoring.
/// </summary>
public class OnnxRerankerOptions
{
    /// <summary>Path to the ONNX cross-encoder model file.</summary>
    public required string ModelPath { get; set; }

    /// <summary>Path to tokenizer artifacts (directory or vocab file).</summary>
    public required string TokenizerPath { get; set; }

    /// <summary>Name of the query text input column. Default: "Query".</summary>
    public string QueryColumnName { get; set; } = "Query";

    /// <summary>Name of the document text input column. Default: "Document".</summary>
    public string DocumentColumnName { get; set; } = "Document";

    /// <summary>Name of the output score column. Default: "Score".</summary>
    public string OutputColumnName { get; set; } = "Score";

    /// <summary>Maximum token length for the combined query + document. Default: 512.</summary>
    public int MaxTokenLength { get; set; } = 512;

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
