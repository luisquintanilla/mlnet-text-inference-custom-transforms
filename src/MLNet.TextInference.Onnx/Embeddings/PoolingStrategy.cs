namespace MLNet.TextInference.Onnx;

/// <summary>
/// Pooling strategy for converting per-token hidden states into a single embedding vector.
/// </summary>
public enum PoolingStrategy
{
    /// <summary>Average of all non-padding token embeddings (most common for sentence-transformers).</summary>
    MeanPooling,

    /// <summary>Use the [CLS] token's embedding (first token position).</summary>
    ClsToken,

    /// <summary>Element-wise max across all non-padding token positions.</summary>
    MaxPooling
}
