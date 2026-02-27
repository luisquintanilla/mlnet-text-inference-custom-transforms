using System.IO.Compression;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Tokenizers;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Handles saving and loading OnnxTextEmbeddingTransformer to/from a self-contained zip file.
/// The zip contains: model.onnx, tokenizer.json, config.json, manifest.json.
/// </summary>
internal static class ModelPackager
{
    private const string OnnxModelEntry = "model.onnx";
    private const string ConfigEntry = "config.json";
    private const string ManifestEntry = "manifest.json";

    public static void Save(OnnxTextEmbeddingTransformer transformer, string path)
    {
        var options = transformer.Options;
        var tokenizerFileName = Path.GetFileName(options.TokenizerPath);

        using var zipStream = File.Create(path);
        using var archive = new ZipArchive(zipStream, ZipArchiveMode.Create);

        // Bundle the ONNX model
        archive.CreateEntryFromFile(options.ModelPath, OnnxModelEntry, CompressionLevel.SmallestSize);

        // Bundle the tokenizer with its original filename
        archive.CreateEntryFromFile(options.TokenizerPath, tokenizerFileName, CompressionLevel.SmallestSize);

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
            TokenizerFileName = tokenizerFileName
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
            ZipFile.ExtractToDirectory(path, extractDir);

            var configPath = Path.Combine(extractDir, ConfigEntry);

            // Read config
            var configJson = File.ReadAllText(configPath);
            var config = JsonSerializer.Deserialize(configJson, JsonContext.Default.SavedConfig)
                ?? throw new InvalidOperationException("Failed to deserialize config from model package.");

            var modelPath = Path.Combine(extractDir, OnnxModelEntry);
            var tokenizerPath = Path.Combine(extractDir, config.TokenizerFileName);

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

            return estimator.Fit(fitData);
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
