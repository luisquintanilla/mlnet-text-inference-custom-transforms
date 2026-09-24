using System.Text.Json;
using System.Text.Json.Serialization;
using MLNet.TextInference.Onnx;

namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// Versioned profile metadata for the English FP32 Laya export.
/// </summary>
internal sealed class LayaDecisionProfile
{
    public const string EnglishFp32Revision = "68f27dfe5a27a54fb2b1fefc432f43f972e90868";
    public const string EnglishFp32ModelRepository = "receptron/laya-onnx";

    public required string Name { get; init; }
    public required string Revision { get; init; }
    public required int MaxLength { get; init; }
    public required int HeadMaxLength { get; init; }
    public required DecisionTemperaturePolicy TemperaturePolicy { get; init; }
    public required string ModelFile { get; init; }
    public required string TokenizerDirectory { get; init; }
    internal IReadOnlyList<TypedDecisionDiagnostic> Diagnostics { get; init; } = [];

    public static LayaDecisionProfile EnglishFp32 => new()
    {
        Name = "english-fp32-laya",
        Revision = EnglishFp32Revision,
        MaxLength = 512,
        HeadMaxLength = 128,
        ModelFile = "laya.onnx",
        TokenizerDirectory = "tokenizer",
        TemperaturePolicy = DecisionTemperaturePolicy.Default
    };

    internal static LayaDecisionProfile Load(string bundleDirectory, TypedDecisionBundleManifest manifest)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(bundleDirectory);
        ArgumentNullException.ThrowIfNull(manifest);

        var configPath = Path.Combine(bundleDirectory, "laya_config.json");
        if (!File.Exists(configPath))
            throw new FileNotFoundException(
                "The bundle is missing laya_config.json, which is required for calibration.", configPath);

        using var document = JsonDocument.Parse(File.ReadAllText(configPath));
        var root = document.RootElement;
        var maxLength = ReadPositiveInt(root, "max_len", "laya_config.json");
        var headMaxLength = ReadPositiveInt(root, "head_max_len", "laya_config.json");
        var temperatures = new Dictionary<string, float>(StringComparer.Ordinal);

        if (root.TryGetProperty("temperature_by_options", out var perCardinality) &&
            perCardinality.ValueKind == JsonValueKind.Object)
        {
            foreach (var entry in perCardinality.EnumerateObject())
                temperatures[entry.Name] = entry.Value.GetSingle();
        }

        if (root.TryGetProperty("temperature", out var byType) &&
            byType.ValueKind == JsonValueKind.Array)
        {
            var names = new[] { "choice", "score", "noul" };
            var values = byType.EnumerateArray().ToArray();
            for (int i = 0; i < Math.Min(names.Length, values.Length); i++)
                temperatures[names[i]] = values[i].GetSingle();
        }

        var policy = DecisionTemperaturePolicy.FromProfile(
            temperatures,
            manifest.Decoder?.TemperatureMinimum ?? 0.5f,
            manifest.Decoder?.TemperatureMaximum ?? 5f);

        return new LayaDecisionProfile
        {
            Name = manifest.Profile.Name ?? "english-fp32-laya",
            Revision = manifest.Profile.Revision ?? EnglishFp32Revision,
            MaxLength = maxLength,
            HeadMaxLength = headMaxLength,
            ModelFile = manifest.ModelFile,
            TokenizerDirectory = manifest.TokenizerDirectory,
            TemperaturePolicy = policy,
            Diagnostics = policy.Diagnostics
        };
    }

    private static int ReadPositiveInt(JsonElement root, string name, string file)
    {
        if (!root.TryGetProperty(name, out var value) ||
            !value.TryGetInt32(out var result) || result <= 0)
        {
            throw new InvalidDataException($"{file} must contain a positive integer '{name}'.");
        }

        return result;
    }
}

internal sealed class DecisionTemperaturePolicy
{
    public const float DefaultMinimum = 0.5f;
    public const float DefaultMaximum = 5f;

    public static DecisionTemperaturePolicy Default => FromProfile(
        new Dictionary<string, float>(StringComparer.Ordinal)
        {
            ["choice"] = 1,
            ["score"] = 1,
            ["noul"] = 1
        },
        DefaultMinimum,
        DefaultMaximum);

    public required float Minimum { get; init; }
    public required float Maximum { get; init; }
    public required IReadOnlyDictionary<string, float> Temperatures { get; init; }
    internal IReadOnlyList<TypedDecisionDiagnostic> Diagnostics { get; init; } = [];

    internal static DecisionTemperaturePolicy FromProfile(
        IReadOnlyDictionary<string, float> source,
        float minimum,
        float maximum)
    {
        if (!float.IsFinite(minimum) || !float.IsFinite(maximum) || minimum <= 0 || maximum < minimum)
            throw new ArgumentOutOfRangeException(nameof(minimum), "Temperature bounds must be finite and positive.");

        var diagnostics = new List<TypedDecisionDiagnostic>();
        var requestedMinimum = minimum;
        var requestedMaximum = maximum;
        minimum = Math.Clamp(minimum, DefaultMinimum, DefaultMaximum);
        maximum = Math.Clamp(maximum, minimum, DefaultMaximum);
        if (requestedMinimum != minimum || requestedMaximum != maximum)
        {
            diagnostics.Add(new TypedDecisionDiagnostic(
                "temperature-policy-clamped",
                $"Temperature bounds were normalized from [{requestedMinimum}, {requestedMaximum}] " +
                $"to [{minimum}, {maximum}] under the supported [0.5, 5.0] policy."));
        }

        var normalized = new Dictionary<string, float>(StringComparer.Ordinal);
        foreach (var pair in source)
        {
            var value = pair.Value;
            var clamped = float.IsFinite(value) ? Math.Clamp(value, minimum, maximum) : 1f;
            if (!float.IsFinite(value) || value != clamped)
            {
                diagnostics.Add(new TypedDecisionDiagnostic(
                    "temperature-clamped",
                    $"Temperature '{pair.Key}' was normalized from '{value}' to '{clamped}' " +
                    $"under the [{minimum}, {maximum}] policy."));
            }

            normalized[pair.Key] = clamped;
        }

        foreach (var type in new[] { "choice", "score", "noul" })
            normalized.TryAdd(type, 1f);

        return new DecisionTemperaturePolicy
        {
            Minimum = minimum,
            Maximum = maximum,
            Temperatures = normalized,
            Diagnostics = diagnostics
        };
    }

    public float For(DecisionQuestionType type, int optionCount)
    {
        var name = type switch
        {
            DecisionQuestionType.Choice => "choice",
            DecisionQuestionType.Score => "score",
            DecisionQuestionType.Noul => "noul",
            _ => throw new ArgumentOutOfRangeException(nameof(type))
        };

        var bucket = optionCount <= 2 ? "2" :
            optionCount <= 5 ? "3-5" :
            optionCount <= 10 ? "6-10" : "11+";

        var value = Temperatures.TryGetValue($"{name}:{bucket}", out var perCardinality)
            ? perCardinality
            : Temperatures.TryGetValue(name, out var byType)
                ? byType
                : 1f;
        return float.IsFinite(value)
            ? Math.Clamp(value, Minimum, Maximum)
            : 1f;
    }
}

internal sealed record TypedDecisionBundleProfile(
    string? Name = null,
    string? Revision = null);

internal sealed record TypedDecisionDecoderMetadata(
    float TemperatureMinimum = DecisionTemperaturePolicy.DefaultMinimum,
    float TemperatureMaximum = DecisionTemperaturePolicy.DefaultMaximum,
    int ActionProbabilityIndex = 1,
    bool RoundResultsToFourDecimals = false);

/// <summary>Manifest stored at the root of a typed-decision bundle.</summary>
internal sealed class TypedDecisionBundleManifest
{
    public int FormatVersion { get; init; } = 1;
    public string ModelFile { get; init; } = "laya.onnx";
    public string[] ExternalDataFiles { get; init; } = ["laya.onnx.data"];
    public string TokenizerDirectory { get; init; } = "tokenizer";
    public TypedDecisionBundleProfile Profile { get; init; } = new();
    public TypedDecisionDecoderMetadata Decoder { get; init; } = new();
    public Dictionary<string, string> FileSha256 { get; init; } = new(StringComparer.Ordinal);

    internal void Validate()
    {
        if (FormatVersion != 1)
            throw new InvalidDataException($"Unsupported typed-decision bundle format version {FormatVersion}.");
        ValidateRelativePath(ModelFile, nameof(ModelFile));
        ValidateRelativePath(TokenizerDirectory, nameof(TokenizerDirectory));
        foreach (var file in ExternalDataFiles)
            ValidateRelativePath(file, nameof(ExternalDataFiles));
        if (Decoder.ActionProbabilityIndex is < 0 or > 1)
            throw new InvalidDataException("ActionProbabilityIndex must be 0 or 1.");
    }

    internal static void ValidateRelativePath(string path, string property)
        => AssetArchive.NormalizeRelativePath(path, property);
}
