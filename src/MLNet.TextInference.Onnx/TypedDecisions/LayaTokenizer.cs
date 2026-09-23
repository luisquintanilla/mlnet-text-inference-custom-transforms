using System.Text;
using System.Text.Json;
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
    private LayaTokenizer(Tokenizer tokenizer, IReadOnlyDictionary<string, int> specialTokens)
    {
        Tokenizer = tokenizer;
        SpecialTokens = specialTokens;
        Metadata = new LayaTokenizerMetadata(
            RequiredSpecial(specialTokens, "[CLS]"),
            RequiredSpecial(specialTokens, "[SEP]"),
            RequiredSpecial(specialTokens, "[MASK]"),
            RequiredSpecial(specialTokens, "[PAD]"),
            "[MASK]");
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
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerDirectory);
        var tokenizerJsonPath = Path.Combine(tokenizerDirectory, "tokenizer.json");
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException("The tokenizer directory must contain tokenizer.json.", tokenizerJsonPath);

        using var document = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var root = document.RootElement;
        if (!root.TryGetProperty("model", out var model) ||
            !model.TryGetProperty("type", out var modelType) ||
            !string.Equals(modelType.GetString(), "BPE", StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                "Typed decisions require a Hugging Face BPE tokenizer.json for the selected Laya profile.");
        }

        var vocabulary = new List<KeyValuePair<string, int>>();
        foreach (var entry in model.GetProperty("vocab").EnumerateObject())
            vocabulary.Add(new KeyValuePair<string, int>(entry.Name, entry.Value.GetInt32()));

        var merges = model.TryGetProperty("merges", out var mergeElement)
            ? mergeElement.EnumerateArray().Select(ReadMerge).ToArray()
            : [];

        var specialTokens = ReadSpecialTokens(root, tokenizerDirectory);
        var options = new BpeOptions(vocabulary)
        {
            Merges = merges,
            SpecialTokens = specialTokens,
            ByteLevel = model.TryGetProperty("byte_fallback", out var byteFallback)
                ? byteFallback.GetBoolean()
                : true,
            PreTokenizer = ResolvePreTokenizer(root, specialTokens),
            Normalizer = ResolveNormalizer(root),
            UnknownToken = model.TryGetProperty("unk_token", out var unknown)
                ? unknown.GetString()
                : null,
            FuseUnknownTokens = model.TryGetProperty("fuse_unk", out var fuseUnknown) &&
                fuseUnknown.GetBoolean()
        };

        // Laya's exported English tokenizer is byte-level BPE even when byte_fallback is
        // absent from tokenizer.json. The profile uses the explicit byte-level contract.
        options.ByteLevel = true;
        var tokenizer = BpeTokenizer.Create(options);
        return new LayaTokenizer(tokenizer, specialTokens);
    }

    private static string ReadMerge(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.String)
            return value.GetString()
                ?? throw new InvalidDataException("Tokenizer merges cannot contain null values.");

        if (value.ValueKind == JsonValueKind.Array)
        {
            var parts = value.EnumerateArray().ToArray();
            if (parts.Length == 2 &&
                parts[0].ValueKind == JsonValueKind.String &&
                parts[1].ValueKind == JsonValueKind.String)
            {
                return $"{parts[0].GetString()} {parts[1].GetString()}";
            }
        }

        throw new InvalidDataException(
            "Tokenizer merges must be strings or two-item string arrays.");
    }

    private static int RequiredSpecial(
        IReadOnlyDictionary<string, int> specialTokens,
        string token)
    {
        if (!specialTokens.TryGetValue(token, out var id))
            throw new InvalidDataException($"Tokenizer is missing required special token '{token}'.");
        return id;
    }

    private static IReadOnlyDictionary<string, int> ReadSpecialTokens(
        JsonElement root,
        string tokenizerDirectory)
    {
        var result = new Dictionary<string, int>(StringComparer.Ordinal);
        if (root.TryGetProperty("added_tokens", out var addedTokens) &&
            addedTokens.ValueKind == JsonValueKind.Array)
        {
            foreach (var entry in addedTokens.EnumerateArray())
            {
                if (entry.TryGetProperty("content", out var content) &&
                    entry.TryGetProperty("id", out var id) &&
                    content.ValueKind == JsonValueKind.String &&
                    id.TryGetInt32(out var tokenId))
                {
                    result[content.GetString()!] = tokenId;
                }
            }
        }

        var configPath = Path.Combine(tokenizerDirectory, "tokenizer_config.json");
        if (File.Exists(configPath))
        {
            using var config = JsonDocument.Parse(File.ReadAllText(configPath));
            if (config.RootElement.TryGetProperty("added_tokens_decoder", out var decoder) &&
                decoder.ValueKind == JsonValueKind.Object)
            {
                foreach (var entry in decoder.EnumerateObject())
                {
                    if (!int.TryParse(entry.Name, out var id) ||
                        !entry.Value.TryGetProperty("content", out var content) ||
                        content.ValueKind != JsonValueKind.String)
                        continue;
                    result[content.GetString()!] = id;
                }
            }
        }

        return result;
    }

    private static PreTokenizer? ResolvePreTokenizer(
        JsonElement root,
        IReadOnlyDictionary<string, int> specialTokens)
    {
        if (!root.TryGetProperty("pre_tokenizer", out var preTokenizer) ||
            preTokenizer.ValueKind != JsonValueKind.Object)
            return RobertaPreTokenizer.Instance;

        var type = preTokenizer.TryGetProperty("type", out var typeElement)
            ? typeElement.GetString()
            : null;

        return type switch
        {
            "ByteLevel" or "Sequence" => RobertaPreTokenizer.Instance,
            "Whitespace" or "WhitespaceSplit" =>
                PreTokenizer.CreateWhiteSpace(specialTokens),
            _ => RobertaPreTokenizer.Instance
        };
    }

    private static Normalizer? ResolveNormalizer(JsonElement root)
    {
        if (!root.TryGetProperty("normalizer", out var normalizer) ||
            normalizer.ValueKind == JsonValueKind.Null)
            return null;

        if (ContainsNormalizer(normalizer, "NFC"))
            return new UnicodeNfcNormalizer();
        if (ContainsNormalizer(normalizer, "Lowercase"))
            return LowerCaseNormalizer.Instance;
        return null;
    }

    private static bool ContainsNormalizer(JsonElement element, string type)
    {
        if (element.ValueKind != JsonValueKind.Object)
            return false;
        if (element.TryGetProperty("type", out var typeElement) &&
            string.Equals(typeElement.GetString(), type, StringComparison.Ordinal))
            return true;
        if (element.TryGetProperty("normalizers", out var sequence) &&
            sequence.ValueKind == JsonValueKind.Array)
            return sequence.EnumerateArray().Any(value => ContainsNormalizer(value, type));
        return false;
    }

    private sealed class UnicodeNfcNormalizer : Normalizer
    {
        public override string Normalize(string original)
            => original.Normalize(NormalizationForm.FormC);

        public override string Normalize(ReadOnlySpan<char> original)
            => original.ToString().Normalize(NormalizationForm.FormC);
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
