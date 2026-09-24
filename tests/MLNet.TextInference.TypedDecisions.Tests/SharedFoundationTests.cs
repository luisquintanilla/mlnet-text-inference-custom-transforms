using System.IO.Compression;
using System.Text;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLNet.TextInference.Onnx;
using MLNet.TextInference.TypedDecisions;

namespace MLNet.TextInference.TypedDecisions.Tests;

[TestClass]
public sealed class SharedFoundationTests
{
    [TestMethod]
    public void TypedDecisionExtensionPreservesOriginatingMLContext()
    {
        var ml = new MLContext(seed: 17);
        ml.GpuDeviceId = 3;
        ml.FallbackToCpu = true;
        var estimator = ml.Transforms.OnnxTypedDecisions(
            new OnnxTypedDecisionsOptions
            {
                ModelAssetsPath = Directory.GetCurrentDirectory(),
                Questions = [DecisionQuestion.Choice("priority", "How urgent?", ["low", "high"])]
            });

        var contextField = estimator.GetType().GetField(
            "_mlContext",
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic);

        Assert.IsNotNull(contextField);
        var recovered = (MLContext)contextField.GetValue(estimator)!;
        Assert.AreEqual(ml.GpuDeviceId, recovered.GpuDeviceId);
        Assert.AreEqual(ml.FallbackToCpu, recovered.FallbackToCpu);
    }

    [TestMethod]
    public void BpeLoaderRejectsOrderedPretokenizerSequences()
    {
        using var document = JsonDocument.Parse(
            """
            {
              "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1},
                "merges": []
              },
              "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                  {"type": "ByteLevel", "add_prefix_space": true},
                  {"type": "WhitespaceSplit"}
                ]
              }
            }
            """);

        Assert.ThrowsExactly<NotSupportedException>(() =>
            HuggingFaceBpeTokenizerLoader.Create(
                document.RootElement,
                string.Empty));
    }

    [TestMethod]
    public void SharedBpeLoaderUsesAddedTokensAndNfcForGoldenIds()
    {
        var root = Directory.CreateTempSubdirectory("bpe-loader-golden-");
        try
        {
            var tokenizerPath = Path.Combine(root.FullName, "tokenizer.json");
            File.WriteAllText(
                tokenizerPath,
                """
                {
                  "normalizer": {"type": "NFC"},
                  "pre_tokenizer": {"type": "WhitespaceSplit"},
                  "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "vocab": {
                      "<unk>": 0,
                      "h": 1,
                      "e": 2,
                      "l": 3,
                      "o": 4,
                      "w": 5,
                      "r": 6,
                      "d": 7,
                      "c": 8,
                      "a": 9,
                      "f": 10,
                      "é": 11,
                      "he": 16,
                      "hel": 17,
                      "hell": 18,
                      "hello": 12,
                      "wo": 19,
                      "wor": 20,
                      "worl": 21,
                      "world": 13,
                      "ca": 22,
                      "caf": 23,
                      "café": 14,
                      "[SPECIAL]": 15
                    },
                    "merges": [
                      "h e",
                      "he l",
                      "hel l",
                      "hell o",
                      "w o",
                      "wo r",
                      "wor l",
                      "worl d",
                      "c a",
                      "ca f",
                      "caf é"
                    ]
                  },
                  "added_tokens": []
                }
                """);
            File.WriteAllText(
                Path.Combine(root.FullName, "tokenizer_config.json"),
                """{"added_tokens_decoder":{"15":{"content":"[SPECIAL]"}}}""");

            var tokenizer = HuggingFaceBpeTokenizerLoader.Load(tokenizerPath);
            var ids = tokenizer.EncodeToIds(
                "hello [SPECIAL] world cafe\u0301",
                considerPreTokenization: true,
                considerNormalization: true);

            CollectionAssert.AreEqual(new[] { 12, 15, 13, 14 }, ids.ToArray());
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    [TestMethod]
    public void SharedBpeLoaderRejectsUnsupportedNonDefaultSettings()
    {
        var unsupported = new[]
        {
            """
            {
              "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": true},
              "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
            }
            """,
            """
            {
              "pre_tokenizer": {"type": "ByteLevel", "use_regex": false},
              "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
            }
            """,
            """
            {
              "normalizer": {
                "type": "Sequence",
                "normalizers": [{"type": "NFC"}, {"type": "Lowercase"}]
              },
              "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
            }
            """,
            """
            {
              "model": {
                "type": "BPE",
                "byte_fallback": true,
                "vocab": {"a": 0},
                "merges": []
              }
            }
            """
        };

        foreach (var json in unsupported)
        {
            using var document = JsonDocument.Parse(json);
            Assert.ThrowsExactly<NotSupportedException>(() =>
                HuggingFaceBpeTokenizerLoader.Create(
                    document.RootElement,
                    string.Empty));
        }
    }

    [TestMethod]
    public void OnnxExternalDataDiscoveryPreservesRelativeLocations()
    {
        var root = Directory.CreateTempSubdirectory("onnx-external-data-");
        try
        {
            var modelPath = Path.Combine(root.FullName, "model.onnx");
            var externalPath = Path.Combine(root.FullName, "weights", "part.bin");
            Directory.CreateDirectory(Path.GetDirectoryName(externalPath)!);
            File.WriteAllBytes(externalPath, [1, 2, 3]);
            File.WriteAllBytes(modelPath, CreateExternalDataModel("weights/part.bin"));

            CollectionAssert.AreEqual(
                new[] { "weights/part.bin" },
                AssetArchive.DiscoverOnnxExternalDataFiles(modelPath).ToArray());
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    [TestMethod]
    public void OnnxExternalDataDiscoveryRejectsTraversal()
    {
        var root = Directory.CreateTempSubdirectory("onnx-external-data-");
        try
        {
            var modelPath = Path.Combine(root.FullName, "model.onnx");
            File.WriteAllBytes(modelPath, CreateExternalDataModel("../part.bin"));

            Assert.ThrowsExactly<InvalidDataException>(() =>
                AssetArchive.DiscoverOnnxExternalDataFiles(modelPath));
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    [TestMethod]
    public void OnnxExternalDataDiscoveryTraversesAttributesSparseGraphsTrainingAndFunctions()
    {
        var root = Directory.CreateTempSubdirectory("onnx-external-data-nested-");
        try
        {
            var locations = new[]
            {
                "main-initializer",
                "main-sparse-indices",
                "main-sparse-values",
                "node-attr-graph-initializer",
                "node-attr-sparse-indices",
                "node-attr-sparse-values",
                "node-attr-tensor",
                "node-attr-tensors-a",
                "node-attr-tensors-b",
                "node-tensor",
                "function-default",
                "function-node-tensor",
                "training-algorithm-initializer",
                "training-algorithm-sparse-indices",
                "training-algorithm-sparse-values",
                "training-initialization-initializer",
                "training-initialization-sparse-indices",
                "training-initialization-sparse-values"
            };
            foreach (var location in locations)
            {
                var path = Path.Combine(root.FullName, location);
                File.WriteAllBytes(path, [1]);
            }

            var modelPath = Path.Combine(root.FullName, "model.onnx");
            File.WriteAllBytes(modelPath, CreateNestedExternalDataModel());

            CollectionAssert.AreEquivalent(
                locations,
                AssetArchive.DiscoverOnnxExternalDataFiles(modelPath).ToArray());
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    [TestMethod]
    public void ZipExtractionAcceptsExplicitDirectoryEntries()
    {
        var root = Directory.CreateTempSubdirectory("zip-directory-entry-");
        var archivePath = Path.Combine(root.Parent!.FullName, root.Name + ".zip");
        var extractPath = Path.Combine(root.FullName, "extract") + Path.DirectorySeparatorChar;
        try
        {
            using (var stream = File.Create(archivePath))
            using (var archive = new ZipArchive(stream, ZipArchiveMode.Create))
            {
                archive.CreateEntry("tokenizer/");
                var file = archive.CreateEntry("tokenizer/vocab.txt");
                using var writer = new StreamWriter(file.Open());
                writer.Write("[UNK]");
            }

            AssetArchive.ExtractZipSafely(archivePath, extractPath);
            Assert.AreEqual(
                "[UNK]",
                File.ReadAllText(Path.Combine(extractPath, "tokenizer", "vocab.txt")));
        }
        finally
        {
            if (File.Exists(archivePath))
                File.Delete(archivePath);
            root.Delete(recursive: true);
        }
    }

    [TestMethod]
    public void ModelPackagerKeepsOriginalModelBasenameAndLoadsPackage()
    {
        const string modelBase64 =
            "CA0SBXRlc3RzOooCCicKCWlucHV0X2lkcxIJaWRzX2Zsb2F0IgRDYXN0KgkKAnRvGAGgAQIKHgoJaWRzX2Zsb2F0CgRiaWFzEgZvdXRwdXQiA0FkZBIQZXh0ZXJuYWxfZml4dHVyZSpFCAQQAUIEYmlhc2ocCghsb2NhdGlvbhIQd2VpZ2h0cy9iaWFzLmJpbmoLCgZvZmZzZXQSATBqDAoGbGVuZ3RoEgIxNnABWiAKCWlucHV0X2lkcxITChEIBxINCgcSBWJhdGNoCgIIBFolCg5hdHRlbnRpb25fbWFzaxITChEIBxINCgcSBWJhdGNoCgIIBGIdCgZvdXRwdXQSEwoRCAESDQoHEgViYXRjaAoCCARCAhAN";
        var root = Directory.CreateTempSubdirectory("onnx-model-package-");
        try
        {
            var modelPath = Path.Combine(root.FullName, "config.onnx");
            File.WriteAllBytes(modelPath, Convert.FromBase64String(modelBase64));
            var externalPath = Path.Combine(root.FullName, "weights", "bias.bin");
            Directory.CreateDirectory(Path.GetDirectoryName(externalPath)!);
            File.WriteAllBytes(
                externalPath,
                Convert.FromHexString("0000803e0000003f0000403f0000803f"));
            var tokenizerPath = Path.Combine(root.FullName, "custom-vocab.txt");
            File.WriteAllText(
                tokenizerPath,
                "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n");

            var ml = new MLContext(seed: 1);
            var source = ml.Data.LoadFromEnumerable([new TextRow { Text = "hello world" }]);
            using var transformer = new OnnxTextEmbeddingEstimator(
                ml,
                new OnnxTextEmbeddingOptions
                {
                    ModelPath = modelPath,
                    TokenizerPath = tokenizerPath,
                    MaxTokenLength = 4,
                    Pooling = PoolingStrategy.ClsToken,
                    Normalize = false,
                    OutputTensorName = "output"
                }).Fit(source);
            var beforeSave = transformer.GenerateEmbeddings(["hello world"])[0];
            transformer.Options.ModelPath = Path.GetRelativePath(
                Environment.CurrentDirectory,
                modelPath);
            transformer.Options.TokenizerPath = Path.GetRelativePath(
                Environment.CurrentDirectory,
                tokenizerPath);
            var packagePath = Path.Combine(root.FullName, "embedding.zip");
            ModelPackager.Save(transformer, packagePath);

            using (var archive = ZipFile.OpenRead(packagePath))
            {
                Assert.IsNotNull(archive.GetEntry("tokenizer/custom-vocab.txt"));
                Assert.IsNotNull(archive.GetEntry("model/config.onnx"));
                Assert.IsNotNull(archive.GetEntry("model/weights/bias.bin"));
            }
            using var loaded = ModelPackager.Load(ml, packagePath);
            Assert.AreEqual(transformer.EmbeddingDimension, loaded.EmbeddingDimension);
            var afterLoad = loaded.GenerateEmbeddings(["hello world"])[0];
            CollectionAssert.AreEqual(beforeSave, afterLoad);
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    [TestMethod]
    public void ClsPoolingDoesNotRequestAttentionMask()
    {
        var ml = new MLContext(seed: 1);
        var source = ml.Data.LoadFromEnumerable(
            new[]
            {
                new PoolingRow
                {
                    RawOutput = new VBuffer<float>(6, [1, 2, 3, 4, 5, 6]),
                    AttentionMask = new VBuffer<long>(3, [0, 0, 0])
                }
            });
        var transformer = new EmbeddingPoolingTransformer(
            ml,
            new EmbeddingPoolingOptions
            {
                InputColumnName = nameof(PoolingRow.RawOutput),
                AttentionMaskColumnName = nameof(PoolingRow.AttentionMask),
                OutputColumnName = "Embedding",
                HiddenDim = 2,
                SequenceLength = 3,
                Pooling = PoolingStrategy.ClsToken,
                Normalize = false
            });

        var mapper = transformer.GetRowToRowMapper(source.Schema);
        var output = mapper.OutputSchema["Embedding"];
        CollectionAssert.DoesNotContain(
            mapper.GetDependencies([output]).Select(static column => column.Name).ToArray(),
            nameof(PoolingRow.AttentionMask));

        using var cursor = source.GetRowCursor(mapper.GetDependencies([output]));
        using var row = mapper.GetRow(cursor, [output]);
        Assert.IsTrue(cursor.MoveNext());
        var getter = row.GetGetter<VBuffer<float>>(output);
        VBuffer<float> value = default;
        getter(ref value);
        CollectionAssert.AreEqual(new[] { 1f, 2f }, value.DenseValues().ToArray());
    }

    private sealed class PoolingRow
    {
        public VBuffer<float> RawOutput { get; set; }
        public VBuffer<long> AttentionMask { get; set; }
    }

    private sealed class TextRow
    {
        public string Text { get; set; } = string.Empty;
    }

    private static byte[] CreateExternalDataModel(string location)
    {
        var entry = Concat(
            Field(1, Encoding.UTF8.GetBytes("location")),
            Field(2, Encoding.UTF8.GetBytes(location)));
        var tensor = Concat(
            Field(13, entry),
            VarintField(14, 1));
        var graph = Field(5, tensor);
        return Field(7, graph);
    }

    private static byte[] CreateNestedExternalDataModel()
    {
        static byte[] ExternalTensor(string location)
            => Concat(
                Field(13, Concat(
                    Field(1, Encoding.UTF8.GetBytes("location")),
                    Field(2, Encoding.UTF8.GetBytes(location)))),
                VarintField(14, 1));

        static byte[] SparseTensor(string prefix)
            => Concat(
                Field(1, ExternalTensor($"{prefix}-values")),
                Field(2, ExternalTensor($"{prefix}-indices")));

        static byte[] TensorAttribute(string name, string location)
            => Concat(
                Field(1, Encoding.UTF8.GetBytes(name)),
                Field(5, ExternalTensor(location)));

        static byte[] GraphAttribute(string name, string location)
            => Concat(
                Field(1, Encoding.UTF8.GetBytes(name)),
                Field(6, Concat(
                    Field(2, Encoding.UTF8.GetBytes(name)),
                    Field(5, ExternalTensor(location)))));

        static byte[] SparseAttribute(string name, string prefix)
            => Concat(
                Field(1, Encoding.UTF8.GetBytes(name)),
                Field(22, SparseTensor(prefix)));

        static byte[] TensorListAttribute(string name, string first, string second)
            => Concat(
                Field(1, Encoding.UTF8.GetBytes(name)),
                Field(10, ExternalTensor(first)),
                Field(10, ExternalTensor(second)));

        static byte[] Node(string tensorAttributeName, string tensorLocation)
            => Concat(
                Field(1, Encoding.UTF8.GetBytes("x")),
                Field(2, Encoding.UTF8.GetBytes("y")),
                Field(3, Encoding.UTF8.GetBytes("node")),
                Field(4, Encoding.UTF8.GetBytes("Identity")),
                Field(5, TensorAttribute(tensorAttributeName, tensorLocation)));

        static byte[] Graph(
            string initializer,
            string sparsePrefix,
            byte[]? node = null)
            => Concat(
                node is null ? [] : Field(1, node),
                Field(2, Encoding.UTF8.GetBytes(initializer)),
                Field(5, ExternalTensor(initializer)),
                Field(15, SparseTensor(sparsePrefix)));

        var mainNode = Concat(
            Node("node-attr-tensor", "node-attr-tensor"),
            Field(5, TensorAttribute("node-tensor", "node-tensor")),
            Field(5, GraphAttribute("node-attr-graph", "node-attr-graph-initializer")),
            Field(5, SparseAttribute("node-attr-sparse", "node-attr-sparse")),
            Field(5, TensorListAttribute("node-attr-tensors", "node-attr-tensors-a", "node-attr-tensors-b")));

        var mainGraph = Concat(
            Field(1, mainNode),
            Field(2, Encoding.UTF8.GetBytes("main")),
            Field(5, ExternalTensor("main-initializer")),
            Field(15, SparseTensor("main-sparse")));

        var initialization = Graph(
            "training-initialization-initializer",
            "training-initialization-sparse");
        var algorithm = Graph(
            "training-algorithm-initializer",
            "training-algorithm-sparse");
        var trainingInfo = Concat(
            Field(1, initialization),
            Field(2, algorithm));

        var functionNode = Node("function-node-tensor", "function-node-tensor");
        var function = Concat(
            Field(1, Encoding.UTF8.GetBytes("ScannerFunction")),
            Field(4, Encoding.UTF8.GetBytes("input")),
            Field(5, Encoding.UTF8.GetBytes("output")),
            Field(7, functionNode),
            Field(11, TensorAttribute("function-default", "function-default")));

        return Concat(
            VarintField(1, 13),
            Field(2, Encoding.UTF8.GetBytes("scanner-fixture")),
            Field(7, mainGraph),
            Field(20, trainingInfo),
            Field(25, function));
    }

    private static byte[] Field(int number, byte[] value)
        => Concat(Varint(number << 3 | 2), Varint(value.Length), value);

    private static byte[] VarintField(int number, ulong value)
        => Concat(Varint(number << 3), Varint(value));

    private static byte[] Varint(int value)
        => Varint((ulong)value);

    private static byte[] Varint(ulong value)
    {
        var bytes = new List<byte>();
        do
        {
            var current = (byte)(value & 0x7f);
            value >>= 7;
            bytes.Add(value == 0 ? current : (byte)(current | 0x80));
        } while (value != 0);
        return [.. bytes];
    }

    private static byte[] Concat(params byte[][] parts)
    {
        var result = new List<byte>();
        foreach (var part in parts)
            result.AddRange(part);
        return [.. result];
    }
}
