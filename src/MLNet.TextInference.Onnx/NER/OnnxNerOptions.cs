namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the end-to-end ONNX NER pipeline.
/// Chains tokenization → ONNX inference → NER decoding.
/// </summary>
public class OnnxNerOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; set; }

    /// <summary>Path to the tokenizer artifacts.</summary>
    public required string TokenizerPath { get; set; }

    /// <summary>Name of the input text column. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output entities column. Default: "Entities".</summary>
    public string OutputColumnName { get; set; } = "Entities";

    /// <summary>BIO label list matching the model's output layer.</summary>
    public required string[] Labels { get; set; }

    /// <summary>Maximum token sequence length. Default: 128.</summary>
    public int MaxTokenLength { get; set; } = 128;

    /// <summary>Batch size for ONNX inference. Default: 32.</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>GPU device ID. Null = CPU only.</summary>
    public int? GpuDeviceId { get; set; }

    /// <summary>If true, fall back to CPU when GPU initialization fails.</summary>
    public bool FallbackToCpu { get; set; }
}
