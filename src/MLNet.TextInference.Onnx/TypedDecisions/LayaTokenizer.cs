using System.Text.Json;
using MLNet.TextInference.Onnx;
using Microsoft.ML.Tokenizers;

namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// Loads the selected profile's Hugging Face BPE tokenizer through Microsoft.ML.Tokenizers.
/// The loader intentionally does not fall back to another tokenizer runtime.
/// </summary>
/// <remarks>
/// The configured <see cref="Tokenizer"/> is exposed directly to preparation. This
/// loader only adapts the profile's Hugging Face asset files and keeps the separate
/// special-token metadata required to build the Laya sequence.
/// </remarks>
internal sealed class LayaTokenizer
{
    private LayaTokenizer(
        Tokenizer tokenizer,
        IReadOnlyDictionary<string, int> specialTokens,
        LayaTokenizerMetadata metadata)
    {
        Tokenizer = tokenizer;
        SpecialTokens = specialTokens;
        Metadata = metadata;
    }

    public Tokenizer Tokenizer { get; }
    public LayaTokenizerMetadata Metadata { get; }
    public IReadOnlyDictionary<string, int> SpecialTokens { get; }
    public int ClsTokenId => Metadata.ClsTokenId;
    public int SepTokenId => Metadata.SepTokenId;
    public int MaskTokenId => Metadata.MaskTokenId;
    public int PadTokenId => Metadata.PadTokenId;
    public string MaskToken => Metadata.MaskToken;

    public static LayaTokenizer Load(string tokenizerDirectory)
    {
        using var document = OpenTokenizerJson(tokenizerDirectory);
        var root = document.RootElement;
        ValidateModel(root);
        var specialTokens = HuggingFaceBpeTokenizerLoader.ReadSpecialTokens(root, tokenizerDirectory);
        var metadata = CreateMetadata(specialTokens);
        var tokenizer = HuggingFaceBpeTokenizerLoader.Create(
            root,
            tokenizerDirectory,
            specialTokens,
            forceByteLevel: true);
        return new LayaTokenizer(tokenizer, specialTokens, metadata);
    }

    public static LayaTokenizerMetadata LoadMetadata(string tokenizerDirectory)
    {
        using var document = OpenTokenizerJson(tokenizerDirectory);
        var root = document.RootElement;
        ValidateModel(root);
        var specialTokens = HuggingFaceBpeTokenizerLoader.ReadSpecialTokens(root, tokenizerDirectory);
        return CreateMetadata(specialTokens);
    }

    private static JsonDocument OpenTokenizerJson(string tokenizerDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerDirectory);
        var tokenizerJsonPath = Path.Combine(tokenizerDirectory, "tokenizer.json");
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException(
                "The tokenizer directory must contain tokenizer.json.", tokenizerJsonPath);
        return JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
    }

    private static void ValidateModel(JsonElement root)
    {
        if (!root.TryGetProperty("model", out var model) ||
            !model.TryGetProperty("type", out var modelType) ||
            !string.Equals(modelType.GetString(), "BPE", StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                "Typed decisions require a Hugging Face BPE tokenizer.json for the selected Laya profile.");
        }
    }

    private static LayaTokenizerMetadata CreateMetadata(
        IReadOnlyDictionary<string, int> specialTokens)
    {
        return new LayaTokenizerMetadata(
            RequiredSpecial(specialTokens, "[CLS]"),
            RequiredSpecial(specialTokens, "[SEP]"),
            RequiredSpecial(specialTokens, "[MASK]"),
            RequiredSpecial(specialTokens, "[PAD]"),
            "[MASK]");
    }

    private static int RequiredSpecial(
        IReadOnlyDictionary<string, int> specialTokens,
        string token)
    {
        if (!specialTokens.TryGetValue(token, out var id))
            throw new InvalidDataException($"Tokenizer is missing required special token '{token}'.");
        return id;
    }

}

/// <summary>
/// Profile-specific IDs that accompany the Microsoft tokenizer engine.
/// </summary>
internal sealed record LayaTokenizerMetadata(
    int ClsTokenId,
    int SepTokenId,
    int MaskTokenId,
    int PadTokenId,
    string MaskToken);
