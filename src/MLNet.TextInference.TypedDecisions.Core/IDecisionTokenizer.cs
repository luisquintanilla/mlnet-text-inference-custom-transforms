namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// Tokenization contract required by Laya decision input preparation.
/// </summary>
/// <remarks>
/// This small abstraction keeps preparation independent from a tokenizer loader while
/// preserving the profile-specific special-token contract. <see cref="LayaTokenizer"/>
/// is the Microsoft.ML.Tokenizers-backed implementation used by the bundle facade.
/// </remarks>
public interface IDecisionTokenizer
{
    int ClsTokenId { get; }
    int SepTokenId { get; }
    int MaskTokenId { get; }
    int PadTokenId { get; }
    string MaskToken { get; }

    IReadOnlyList<int> Encode(string text);

    IReadOnlyList<int> Encode(string text, int maxTokenCount);
}
