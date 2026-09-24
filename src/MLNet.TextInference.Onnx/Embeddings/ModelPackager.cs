using System.IO.Compression;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Tokenizers;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Handles saving and loading OnnxTextEmbeddingTransformer to/from a self-contained zip file.
/// The zip preserves the original model basename and external-data paths under model/,
/// tokenizer assets under tokenizer/, and versioned configuration/manifest metadata.
/// </summary>
internal static class ModelPackager
{
    private const string OnnxModelEntry = "model.onnx";
    private const string OnnxModelDirectory = "model";
    private const string ConfigEntry = "config.json";
    private const string ManifestEntry = "manifest.json";

    public static void Save(OnnxTextEmbeddingTransformer transformer, string path)
    {
        var options = transformer.Options;
        var modelPath = Path.GetFullPath(options.ModelPath);
        var tokenizerPath = Path.GetFullPath(options.TokenizerPath);
        var tokenizerIsDirectory = Directory.Exists(tokenizerPath);
        var tokenizerFileName = tokenizerIsDirectory
            ? "tokenizer"
            : $"tokenizer/{Path.GetFileName(tokenizerPath)}";
        var tokenizerFiles = GetTokenizerFiles(tokenizerPath, tokenizerIsDirectory);
        var modelFileName = $"{OnnxModelDirectory}/{Path.GetFileName(modelPath)}";

        using var zipStream = File.Create(path);
        using var archive = new ZipArchive(zipStream, ZipArchiveMode.Create);

        // Bundle the ONNX model
        archive.CreateEntryFromFile(
            modelPath,
            modelFileName,
            CompressionLevel.SmallestSize);

        // Bundle the tokenizer with its original filename
        if (tokenizerIsDirectory)
        {
            foreach (var file in tokenizerFiles)
            {
                var relative = Path.GetRelativePath(tokenizerPath, file)
                    .Replace('\\', '/');
                archive.CreateEntryFromFile(
                    file,
                    $"tokenizer/{relative}",
                    CompressionLevel.SmallestSize);
            }
        }
        else
        {
            foreach (var file in tokenizerFiles)
            {
                archive.CreateEntryFromFile(
                    file,
                    $"tokenizer/{Path.GetFileName(file)}",
                    CompressionLevel.SmallestSize);
            }
        }

        var externalData = AssetArchive.DiscoverOnnxExternalDataFiles(modelPath);
        foreach (var relative in externalData)
        {
            var externalPath = AssetArchive.ResolveWithinRoot(
                Path.GetDirectoryName(modelPath)!,
                relative,
                "ONNX external-data location");
            archive.CreateEntryFromFile(
                externalPath,
                $"{OnnxModelDirectory}/{relative}",
                CompressionLevel.NoCompression);
        }

        // Save config (serializable subset of options)
        var config = new SavedConfig
        {
            InputColumnName = options.InputColumnName,
            OutputColumnName = options.OutputColumnName,
            MaxTokenLength = options.MaxTokenLength,
            Pooling = options.Pooling,
            Normalize = options.Normalize,
            BatchSize = options.BatchSize,
            InputIdsName = options.InputIdsName,
            AttentionMaskName = options.AttentionMaskName,
            TokenTypeIdsName = options.TokenTypeIdsName,
            OutputTensorName = options.OutputTensorName,
            TokenizerFileName = tokenizerFileName,
            ModelFileName = modelFileName,
            ExternalDataFileNames = externalData.ToArray(),
        };

        var configEntry = archive.CreateEntry(ConfigEntry);
        using (var writer = new StreamWriter(configEntry.Open()))
        {
            writer.Write(JsonSerializer.Serialize(config, JsonContext.Default.SavedConfig));
        }

        // Save manifest
        var manifest = new Manifest
        {
            Version = "1.0",
            Framework = "MLNet.TextInference.Onnx",
            EmbeddingDimension = transformer.EmbeddingDimension,
            CreatedAt = DateTime.UtcNow.ToString("o")
        };

        var manifestEntry = archive.CreateEntry(ManifestEntry);
        using (var writer = new StreamWriter(manifestEntry.Open()))
        {
            writer.Write(JsonSerializer.Serialize(manifest, JsonContext.Default.Manifest));
        }
    }

    public static OnnxTextEmbeddingTransformer Load(MLContext mlContext, string path)
    {
        // Extract to a temp directory
        var extractDir = Path.Combine(Path.GetTempPath(), "mlnet-onnx-embed-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(extractDir);

        try
        {
            AssetArchive.ExtractZipSafely(path, extractDir);

            var configPath = Path.Combine(extractDir, ConfigEntry);

            // Read config
            var configJson = File.ReadAllText(configPath);
            var config = JsonSerializer.Deserialize(configJson, JsonContext.Default.SavedConfig)
                ?? throw new InvalidOperationException("Failed to deserialize config from model package.");

            var modelPath = AssetArchive.ResolveWithinRoot(
                extractDir,
                config.ModelFileName,
                "model path");
            var modelDirectory = Path.GetDirectoryName(modelPath)!;
            foreach (var external in config.GetExternalDataFileNames())
            {
                var externalPath = AssetArchive.ResolveWithinRoot(
                    modelDirectory,
                    external,
                    "ONNX external-data path");
                if (!File.Exists(externalPath))
                    throw new FileNotFoundException(
                        $"Model package is missing external-data file '{external}'.",
                        externalPath);
            }
            var tokenizerPath = AssetArchive.ResolveWithinRoot(
                extractDir,
                config.TokenizerFileName,
                "tokenizer path");

            var options = new OnnxTextEmbeddingOptions
            {
                ModelPath = modelPath,
                TokenizerPath = tokenizerPath,
                InputColumnName = config.InputColumnName,
                OutputColumnName = config.OutputColumnName,
                MaxTokenLength = config.MaxTokenLength,
                Pooling = config.Pooling,
                Normalize = config.Normalize,
                BatchSize = config.BatchSize,
                InputIdsName = config.InputIdsName,
                AttentionMaskName = config.AttentionMaskName,
                TokenTypeIdsName = config.TokenTypeIdsName,
                OutputTensorName = config.OutputTensorName
            };

            // Use the estimator's discovery logic to create the transformer
            var estimator = new OnnxTextEmbeddingEstimator(mlContext, options);

            // Create a dummy IDataView just for schema validation in Fit
            var dummyData = mlContext.Data.LoadFromEnumerable(
                new[] { new DummyTextRow { Text = "" } });

            // If the input column name isn't "Text", we need to rename
            IDataView fitData;
            if (options.InputColumnName != "Text")
            {
                fitData = mlContext.Transforms.CopyColumns(options.InputColumnName, "Text")
                    .Fit(dummyData).Transform(dummyData);
            }
            else
            {
                fitData = dummyData;
            }

            var transformer = estimator.Fit(fitData);
            transformer.AttachOwnedAssetDirectory(extractDir);
            extractDir = string.Empty;
            return transformer;
        }
        catch
        {
            // Clean up on failure
            try { Directory.Delete(extractDir, true); } catch { }
            throw;
        }
    }

    private sealed class DummyTextRow
    {
        public string Text { get; set; } = "";
    }

    private static IReadOnlyList<string> GetTokenizerFiles(
        string tokenizerPath,
        bool tokenizerIsDirectory)
    {
        // Reuse the production loader as the required-asset resolver so a
        // tokenizer_config.json-only package cannot succeed and fail on load.
        _ = TextTokenizerEstimator.LoadTokenizer(tokenizerPath);

        var directory = tokenizerIsDirectory
            ? tokenizerPath
            : Path.GetDirectoryName(tokenizerPath)
                ?? throw new InvalidOperationException(
                    $"Cannot determine tokenizer directory for '{tokenizerPath}'.");
        var files = EnumerateTokenizerFiles(directory, throwIfEmpty: tokenizerIsDirectory).ToList();
        if (!tokenizerIsDirectory &&
            !files.Contains(tokenizerPath, StringComparer.OrdinalIgnoreCase))
        {
            files.Add(tokenizerPath);
        }

        return files;
    }

    private static IEnumerable<string> EnumerateTokenizerFiles(string tokenizerDirectory, bool throwIfEmpty = true)
    {
        var supportedNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "sentencepiece.bpe.model",
            "spiece.model",
            "spm.model"
        };

        var files = Directory.EnumerateFiles(
                tokenizerDirectory,
                "*",
                SearchOption.TopDirectoryOnly)
            .Where(file => supportedNames.Contains(Path.GetFileName(file)))
            .ToArray();
        if (files.Length == 0 && throwIfEmpty)
            throw new FileNotFoundException(
                $"Tokenizer directory '{tokenizerDirectory}' contains no supported tokenizer assets.");
        return files;
    }

    internal sealed class SavedConfig
    {
        public string InputColumnName { get; set; } = "Text";
        public string OutputColumnName { get; set; } = "Embedding";
        public int MaxTokenLength { get; set; } = 128;
        public PoolingStrategy Pooling { get; set; } = PoolingStrategy.MeanPooling;
        public bool Normalize { get; set; } = true;
        public int BatchSize { get; set; } = 32;
        public string? InputIdsName { get; set; }
        public string? AttentionMaskName { get; set; }
        public string? TokenTypeIdsName { get; set; }
        public string? OutputTensorName { get; set; }
        public string TokenizerFileName { get; set; } = "vocab.txt";
        public string ModelFileName { get; set; } = OnnxModelEntry;
        public string[]? ExternalDataFileNames { get; set; }
        public string? ExternalDataFileName { get; set; }

        internal IEnumerable<string> GetExternalDataFileNames()
            => ExternalDataFileNames is { Length: > 0 }
                ? ExternalDataFileNames
                : ExternalDataFileName is null
                    ? []
                    : [ExternalDataFileName];
    }

    internal sealed class Manifest
    {
        public string Version { get; set; } = "1.0";
        public string Framework { get; set; } = "MLNet.TextInference.Onnx";
        public int EmbeddingDimension { get; set; }
        public string CreatedAt { get; set; } = "";
    }
}

[System.Text.Json.Serialization.JsonSerializable(typeof(ModelPackager.SavedConfig))]
[System.Text.Json.Serialization.JsonSerializable(typeof(ModelPackager.Manifest))]
internal partial class JsonContext : System.Text.Json.Serialization.JsonSerializerContext
{
}
