using System.Numerics.Tensors;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Applies pooling and normalization to ONNX model outputs to produce final embeddings.
/// Uses TensorPrimitives for SIMD-accelerated math.
/// </summary>
internal static class EmbeddingPooling
{
    /// <summary>
    /// Pools per-token hidden states into a single embedding vector per batch item.
    /// </summary>
    /// <param name="hiddenStates">Flat output from ONNX model, shape [batchSize, seqLen, hiddenDim].</param>
    /// <param name="attentionMask">Flat attention mask, shape [batchSize, seqLen].</param>
    /// <param name="batchSize">Number of items in the batch.</param>
    /// <param name="seqLen">Sequence length (padded).</param>
    /// <param name="hiddenDim">Hidden dimension of the model.</param>
    /// <param name="strategy">Pooling strategy to apply.</param>
    /// <param name="normalize">Whether to L2-normalize the result.</param>
    /// <returns>Array of embedding vectors, one per batch item.</returns>
    public static float[][] Pool(
        ReadOnlySpan<float> hiddenStates,
        ReadOnlySpan<long> attentionMask,
        int batchSize,
        int seqLen,
        int hiddenDim,
        PoolingStrategy strategy,
        bool normalize)
    {
        var embeddings = new float[batchSize][];

        for (int b = 0; b < batchSize; b++)
        {
            embeddings[b] = strategy switch
            {
                PoolingStrategy.MeanPooling => MeanPool(hiddenStates, attentionMask, b, seqLen, hiddenDim),
                PoolingStrategy.ClsToken => ClsPool(hiddenStates, b, seqLen, hiddenDim),
                PoolingStrategy.MaxPooling => MaxPool(hiddenStates, attentionMask, b, seqLen, hiddenDim),
                _ => throw new ArgumentOutOfRangeException(nameof(strategy))
            };

            if (normalize)
                L2Normalize(embeddings[b]);
        }

        return embeddings;
    }

    /// <summary>
    /// Extracts a pre-pooled embedding directly (for models that output sentence_embedding).
    /// </summary>
    public static float[][] ExtractPooled(ReadOnlySpan<float> pooledOutput, int batchSize, int hiddenDim, bool normalize)
    {
        var embeddings = new float[batchSize][];

        for (int b = 0; b < batchSize; b++)
        {
            embeddings[b] = pooledOutput.Slice(b * hiddenDim, hiddenDim).ToArray();

            if (normalize)
                L2Normalize(embeddings[b]);
        }

        return embeddings;
    }

    private static float[] MeanPool(
        ReadOnlySpan<float> hiddenStates,
        ReadOnlySpan<long> attentionMask,
        int batchIdx, int seqLen, int hiddenDim)
    {
        var embedding = new float[hiddenDim];
        float tokenCount = 0;

        for (int s = 0; s < seqLen; s++)
        {
            if (attentionMask[batchIdx * seqLen + s] > 0)
            {
                int offset = (batchIdx * seqLen + s) * hiddenDim;
                ReadOnlySpan<float> tokenEmbed = hiddenStates.Slice(offset, hiddenDim);
                TensorPrimitives.Add(embedding, tokenEmbed, embedding);
                tokenCount++;
            }
        }

        if (tokenCount > 0)
            TensorPrimitives.Divide(embedding, tokenCount, embedding);

        return embedding;
    }

    private static float[] ClsPool(
        ReadOnlySpan<float> hiddenStates,
        int batchIdx, int seqLen, int hiddenDim)
    {
        // [CLS] is always position 0
        int offset = batchIdx * seqLen * hiddenDim;
        return hiddenStates.Slice(offset, hiddenDim).ToArray();
    }

    private static float[] MaxPool(
        ReadOnlySpan<float> hiddenStates,
        ReadOnlySpan<long> attentionMask,
        int batchIdx, int seqLen, int hiddenDim)
    {
        var embedding = new float[hiddenDim];
        Array.Fill(embedding, float.MinValue);
        bool hasTokens = false;

        for (int s = 0; s < seqLen; s++)
        {
            if (attentionMask[batchIdx * seqLen + s] > 0)
            {
                hasTokens = true;
                int offset = (batchIdx * seqLen + s) * hiddenDim;
                ReadOnlySpan<float> tokenEmbed = hiddenStates.Slice(offset, hiddenDim);
                TensorPrimitives.Max(embedding, tokenEmbed, embedding);
            }
        }

        if (!hasTokens)
            Array.Clear(embedding);

        return embedding;
    }

    private static void L2Normalize(Span<float> embedding)
    {
        float norm = TensorPrimitives.Norm(embedding);
        if (norm > 0)
            TensorPrimitives.Divide(embedding, norm, embedding);
    }
}
