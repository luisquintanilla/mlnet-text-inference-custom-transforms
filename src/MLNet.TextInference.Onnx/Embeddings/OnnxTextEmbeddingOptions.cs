namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the ONNX text embedding transform.
/// </summary>
public class OnnxTextEmbeddingOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; set; }

    /// <summary>
    /// Path to tokenizer artifacts. Can be a directory containing tokenizer_config.json,
    /// a tokenizer_config.json file, or a direct vocab file (.txt, .model).
    /// See <see cref="TextTokenizerOptions.TokenizerPath"/> for full resolution rules.
    /// </summary>
    public required string TokenizerPath { get; set; }

    /// <summary>Name of the input text column in the IDataView. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output embedding column in the IDataView. Default: "Embedding".</summary>
    public string OutputColumnName { get; set; } = "Embedding";

    /// <summary>Maximum number of tokens per input text. Texts are padded/truncated to this length. Default: 128.</summary>
    public int MaxTokenLength { get; set; } = 128;

    /// <summary>Pooling strategy for reducing per-token outputs to a single vector. Default: MeanPooling.</summary>
    public PoolingStrategy Pooling { get; set; } = PoolingStrategy.MeanPooling;

    /// <summary>Whether to L2-normalize the output embeddings. Default: true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Batch size for ONNX inference. Default: 32.</summary>
    public int BatchSize { get; set; } = 32;

    // --- Auto-discovery overrides (null = auto-detect from ONNX metadata) ---

    /// <summary>ONNX input tensor name for token IDs. Null = auto-detect (expects "input_ids").</summary>
    public string? InputIdsName { get; set; }

    /// <summary>ONNX input tensor name for attention mask. Null = auto-detect (expects "attention_mask").</summary>
    public string? AttentionMaskName { get; set; }

    /// <summary>ONNX input tensor name for token type IDs. Null = auto-detect (uses "token_type_ids" if present).</summary>
    public string? TokenTypeIdsName { get; set; }

    /// <summary>ONNX output tensor name for embeddings. Null = auto-detect (prefers "sentence_embedding", falls back to "last_hidden_state").</summary>
    public string? OutputTensorName { get; set; }

    /// <summary>
    /// Optional GPU device ID to run execution on. Null = use MLContext.GpuDeviceId.
    /// Set to a non-negative integer (e.g. 0) to target a specific CUDA device.
    /// Requires the consuming application to reference Microsoft.ML.OnnxRuntime.Gpu.
    /// </summary>
    public int? GpuDeviceId { get; set; }

    /// <summary>
    /// If true and GPU initialization fails, fall back to CPU instead of throwing.
    /// Default: false.
    /// </summary>
    public bool FallbackToCpu { get; set; }
}
