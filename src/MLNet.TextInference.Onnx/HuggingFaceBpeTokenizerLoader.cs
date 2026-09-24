using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Shared, explicit adapter for the supported Hugging Face BPE tokenizer.json
/// surface. It deliberately rejects tokenizer graph components that cannot be
/// represented by Microsoft.ML.Tokenizers rather than silently approximating them.
/// </summary>
internal static class HuggingFaceBpeTokenizerLoader
{
    public static Tokenizer Load(
        string tokenizerJsonPath,
        IReadOnlyDictionary<string, int>? specialTokens = null,
        bool forceByteLevel = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerJsonPath);
        using var document = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        return Create(
            document.RootElement,
            Path.GetDirectoryName(tokenizerJsonPath) ?? string.Empty,
            specialTokens,
            forceByteLevel);
    }

    public static Tokenizer Create(
        JsonElement root,
        string tokenizerDirectory,
        IReadOnlyDictionary<string, int>? specialTokens = null,
        bool forceByteLevel = false)
    {
        if (!root.TryGetProperty("model", out var model) ||
            !model.TryGetProperty("type", out var modelType) ||
            !string.Equals(modelType.GetString(), "BPE", StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                "The shared loader only supports a Hugging Face BPE tokenizer.json.");
        }

        if (!model.TryGetProperty("vocab", out var vocabElement))
            throw new InvalidDataException("BPE tokenizer.json is missing model.vocab.");

        var vocabulary = vocabElement.EnumerateObject()
            .Select(static entry => new KeyValuePair<string, int>(
                entry.Name,
                entry.Value.GetInt32()))
            .ToArray();
        ValidateBpeSettings(model);
        var effectiveSpecialTokens = specialTokens ??
            ReadSpecialTokens(root, tokenizerDirectory);
        var merges = model.TryGetProperty("merges", out var mergeElement)
            ? mergeElement.EnumerateArray().Select(ReadMerge).ToArray()
            : [];

        var options = new BpeOptions(vocabulary)
        {
            Merges = merges,
            SpecialTokens = effectiveSpecialTokens.Count == 0
                ? null
                : effectiveSpecialTokens,
            ByteLevel = forceByteLevel || IsByteLevelPreTokenizer(root),
            PreTokenizer = ResolvePreTokenizer(
                root,
                effectiveSpecialTokens,
                allowByteLevelPrefixSpace: forceByteLevel),
            Normalizer = ResolveNormalizer(root),
            UnknownToken = model.TryGetProperty("unk_token", out var unknown)
                ? unknown.GetString()
                : null,
            FuseUnknownTokens = model.TryGetProperty("fuse_unk", out var fuseUnknown) &&
                fuseUnknown.GetBoolean(),
            ContinuingSubwordPrefix = ReadOptionalString(model, "continuing_subword_prefix"),
            EndOfWordSuffix = ReadOptionalString(model, "end_of_word_suffix")
        };

        return BpeTokenizer.Create(options);
    }

    private static string ReadMerge(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.String)
            return value.GetString()
                ?? throw new InvalidDataException("BPE merges cannot contain null values.");
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

        throw new NotSupportedException(
            "BPE merges must be strings or two-item string arrays.");
    }

    private static bool IsByteLevelPreTokenizer(JsonElement root)
    {
        if (!root.TryGetProperty("pre_tokenizer", out var value) ||
            value.ValueKind != JsonValueKind.Object ||
            !value.TryGetProperty("type", out var type))
            return false;

        if (string.Equals(type.GetString(), "ByteLevel", StringComparison.Ordinal))
            return true;

        if (!string.Equals(type.GetString(), "Sequence", StringComparison.Ordinal) ||
            !value.TryGetProperty("pretokenizers", out var sequence))
            return false;

        var items = sequence.EnumerateArray().ToArray();
        return items.Length == 1 && IsByteLevel(items[0]);
    }

    private static bool IsByteLevel(JsonElement value)
        => value.ValueKind == JsonValueKind.Object &&
            value.TryGetProperty("type", out var type) &&
            string.Equals(type.GetString(), "ByteLevel", StringComparison.Ordinal);

    private static void ValidateBpeSettings(JsonElement model)
    {
        if (model.TryGetProperty("byte_fallback", out var byteFallback) &&
            byteFallback.ValueKind == JsonValueKind.True)
            throw new NotSupportedException(
                "BPE byte_fallback=true is not representable by Microsoft.ML.Tokenizers.");
        if (model.TryGetProperty("ignore_merges", out var ignoreMerges) &&
            ignoreMerges.ValueKind == JsonValueKind.True)
            throw new NotSupportedException(
                "BPE ignore_merges=true is not representable by Microsoft.ML.Tokenizers.");
        if (model.TryGetProperty("dropout", out var dropout) &&
            dropout.ValueKind is JsonValueKind.Number &&
            dropout.TryGetDouble(out var dropoutValue) &&
            dropoutValue != 0)
            throw new NotSupportedException(
                "BPE dropout must be zero for deterministic Microsoft.ML.Tokenizers encoding.");
    }

    private static void ValidateByteLevelSettings(
        JsonElement value,
        bool allowAddPrefixSpace)
    {
        if (!allowAddPrefixSpace &&
            value.TryGetProperty("add_prefix_space", out var addPrefixSpace) &&
            addPrefixSpace.ValueKind == JsonValueKind.True)
            throw new NotSupportedException(
                "ByteLevel add_prefix_space=true is not representable by Microsoft.ML.Tokenizers.");
        if (value.TryGetProperty("use_regex", out var useRegex) &&
            useRegex.ValueKind == JsonValueKind.False)
            throw new NotSupportedException(
                "ByteLevel use_regex=false is not representable by Microsoft.ML.Tokenizers.");
        if (value.TryGetProperty("trim_offsets", out var trimOffsets) &&
            trimOffsets.ValueKind == JsonValueKind.False)
            throw new NotSupportedException(
                "ByteLevel trim_offsets=false is not representable by Microsoft.ML.Tokenizers.");
    }

    private static string? ReadOptionalString(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var value) ||
            value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.String)
            throw new InvalidDataException(
                $"BPE property '{propertyName}' must be a string or null.");
        return value.GetString();
    }

    private static PreTokenizer? ResolvePreTokenizer(
        JsonElement root,
        IReadOnlyDictionary<string, int>? specialTokens,
        bool allowByteLevelPrefixSpace)
    {
        if (!root.TryGetProperty("pre_tokenizer", out var value) ||
            value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.Object ||
            !value.TryGetProperty("type", out var type))
            throw new NotSupportedException("Unsupported BPE pre_tokenizer configuration.");

        var typeName = type.GetString();
        if (string.Equals(typeName, "Sequence", StringComparison.Ordinal))
        {
            if (!value.TryGetProperty("pretokenizers", out var sequence))
                throw new NotSupportedException(
                    "BPE Sequence pre_tokenizer is missing pretokenizers.");

            var items = sequence.EnumerateArray().ToArray();
            if (items.Length != 1)
                throw new NotSupportedException(
                    "Only a single-item BPE Sequence pre_tokenizer is supported; " +
                    "ordered combinations are not equivalent to Microsoft.ML.Tokenizers.");

            value = items[0];
            if (!value.TryGetProperty("type", out type))
                throw new NotSupportedException(
                    "Unsupported BPE Sequence pre_tokenizer item.");
            typeName = type.GetString();
        }

        if (string.Equals(typeName, "ByteLevel", StringComparison.Ordinal))
        {
            ValidateByteLevelSettings(value, allowByteLevelPrefixSpace);
            return RobertaPreTokenizer.Instance;
        }

        return typeName switch
        {
            "WhitespaceSplit" =>
                PreTokenizer.CreateWhiteSpace(specialTokens ?? new Dictionary<string, int>()),
            "Whitespace" => throw new NotSupportedException(
                "The Hugging Face Whitespace pre_tokenizer is not equivalent to the " +
                "available Microsoft.ML.Tokenizers whitespace adapter."),
            _ => throw new NotSupportedException(
                $"Unsupported BPE pre_tokenizer type '{typeName}'.")
        };
    }

    private static Normalizer? ResolveNormalizer(JsonElement root)
    {
        if (!root.TryGetProperty("normalizer", out var value) ||
            value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.Object)
            throw new NotSupportedException("Unsupported BPE normalizer configuration.");

        var types = EnumerateNormalizerTypes(value).ToArray();
        if (types.Any(static type => type is not ("NFC" or "Lowercase")))
            throw new NotSupportedException(
                $"Unsupported BPE normalizer type '{types.First(type => type is not ("NFC" or "Lowercase"))}'.");
        if (types.Length > 1)
            throw new NotSupportedException(
                "Ordered BPE normalizer sequences are not representable by the shared tokenizer adapter.");
        if (types.Contains("NFC", StringComparer.Ordinal))
            return new UnicodeNfcNormalizer();
        if (types.Contains("Lowercase", StringComparer.Ordinal))
            return LowerCaseNormalizer.Instance;
        return null;
    }

    internal static IReadOnlyDictionary<string, int> ReadSpecialTokens(
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

    private static IEnumerable<string> EnumerateNormalizerTypes(JsonElement value)
    {
        if (value.TryGetProperty("type", out var type) && type.ValueKind == JsonValueKind.String)
            yield return type.GetString()!;
        if (value.TryGetProperty("normalizers", out var sequence) &&
            sequence.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in sequence.EnumerateArray())
            foreach (var nested in EnumerateNormalizerTypes(item))
                yield return nested;
        }
    }

    private sealed class UnicodeNfcNormalizer : Normalizer
    {
        public override string Normalize(string original)
            => original.Normalize(NormalizationForm.FormC);

        public override string Normalize(ReadOnlySpan<char> original)
            => original.ToString().Normalize(NormalizationForm.FormC);
    }
}
