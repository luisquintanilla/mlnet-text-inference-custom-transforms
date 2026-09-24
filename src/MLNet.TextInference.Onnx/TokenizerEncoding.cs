using Microsoft.ML.Tokenizers;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Bounded Microsoft tokenizer calls shared by direct and cursor adapters.
/// Layout, special-token placement, and model-specific budgets remain owned by
/// the consuming task.
/// </summary>
internal static class TokenizerEncoding
{
    public static IReadOnlyList<int> EncodeIds(
        Tokenizer tokenizer,
        string text,
        int maxTokenCount)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(text);
        if (maxTokenCount < 0)
            throw new ArgumentOutOfRangeException(nameof(maxTokenCount));
        if (maxTokenCount == 0)
            return [];

        return tokenizer.EncodeToIds(
            text,
            maxTokenCount,
            out _,
            out _,
            considerPreTokenization: true,
            considerNormalization: true);
    }

    public static IReadOnlyList<EncodedToken> EncodeTokens(
        Tokenizer tokenizer,
        string text)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(text);
        return tokenizer.EncodeToTokens(text, out _);
    }
}
