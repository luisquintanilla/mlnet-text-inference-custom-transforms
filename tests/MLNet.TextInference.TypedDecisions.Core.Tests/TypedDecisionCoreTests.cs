using System.IO.Compression;
using System.Text.Json;
using Microsoft.ML.Tokenizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLNet.TextInference.TypedDecisions;

namespace MLNet.TextInference.TypedDecisions.Core.Tests;

[TestClass]
public sealed class TypedDecisionCoreTests
{
    [TestMethod]
    public void PrepareDecisionInputs_ProducesFiveDenseShapesAndMarkers()
    {
        using var fixture = TinyTokenizerFixture.Create();
        var profile = new LayaDecisionProfile
        {
            Name = "fixture",
            Revision = "fixture",
            MaxLength = 64,
            HeadMaxLength = 16,
            ModelFile = "model.onnx",
            TokenizerDirectory = "tokenizer",
            TemperaturePolicy = DecisionTemperaturePolicy.Default
        };
        var prepare = new PrepareDecisionInputs(
            profile,
            fixture.Tokenizer.Tokenizer,
            fixture.Tokenizer.Metadata);
        var input = prepare.Prepare(
            [
                DecisionRequest.Create(
                    "{\"subject\":\"Refund not received\"}",
                    [DecisionQuestion.Choice("team", "Which team?", ["billing", "support"])]),
                DecisionRequest.Create(
                    new string('a', 96),
                    [
                        DecisionQuestion.Score(
                            "urgency",
                            "How urgent?",
                            ["low", "high"])
                    ]),
                DecisionRequest.Create(
                    "short",
                    [DecisionQuestion.Noul("risk", "Will the customer churn?")])
            ]);

        Assert.AreEqual(3, input.BatchSize);
        Assert.AreEqual(input.BatchSize * input.SequenceLength, input.InputIds.Length);
        Assert.AreEqual(input.BatchSize * input.SequenceLength, input.AttentionMask.Length);
        Assert.AreEqual(input.BatchSize * input.MarkerWidth, input.MarkerPositions.Length);
        Assert.AreEqual(input.BatchSize * input.MarkerWidth, input.MarkerMask.Length);
        Assert.AreEqual(input.BatchSize, input.QuestionTypes.Length);
        Assert.IsTrue(input.Items.All(static item =>
            item.MarkerPositions.Length == item.OptionLabels.Length));
        Assert.IsTrue(input.MarkerMask.Any(static value => value));
        var paddingCount = 0;
        for (var index = 0; index < input.InputIds.Length; index++)
        {
            if (input.AttentionMask[index] == 0)
            {
                paddingCount++;
                Assert.AreEqual(fixture.Tokenizer.PadTokenId, input.InputIds[index]);
            }
        }
        Assert.IsTrue(paddingCount > 0, "The fixture should exercise nonzero padding.");
        for (var index = 0; index < input.MarkerMask.Length; index++)
        {
            if (input.MarkerMask[index])
            {
                var row = index / input.MarkerWidth;
                var token = (int)input.MarkerPositions[index];
                Assert.AreEqual(1L, input.AttentionMask[row * input.SequenceLength + token]);
            }
        }
    }

    [TestMethod]
    public void PrepareDecisionInputs_UsesMicrosoftTokenizerAndProfileMetadata()
    {
        using var fixture = TinyTokenizerFixture.Create();
        var profile = new LayaDecisionProfile
        {
            Name = "fixture",
            Revision = "fixture",
            MaxLength = 64,
            HeadMaxLength = 16,
            ModelFile = "model.onnx",
            TokenizerDirectory = "tokenizer",
            TemperaturePolicy = DecisionTemperaturePolicy.Default
        };
        var input = new PrepareDecisionInputs(
            profile,
            fixture.Tokenizer.Tokenizer,
            fixture.Tokenizer.Metadata).Prepare(
            "state",
            [DecisionQuestion.Choice("team", "Which team?", ["billing", "support"])]);

        Assert.AreEqual(1, input.BatchSize);
        Assert.AreEqual(fixture.Tokenizer.ClsTokenId, input.InputIds[0]);
        Assert.IsTrue(input.MarkerMask[0]);
        Assert.AreEqual(
            fixture.Tokenizer.MaskTokenId,
            input.InputIds[(int)input.MarkerPositions[0]]);
    }

    [TestMethod]
    public void PrepareDecisionInputs_HandlesZeroStateRoomAndMicrosoftBoundedEncoding()
    {
        using var fixture = TinyTokenizerFixture.Create();
        var profile = new LayaDecisionProfile
        {
            Name = "fixture",
            Revision = "fixture",
            MaxLength = 64,
            HeadMaxLength = 16,
            ModelFile = "model.onnx",
            TokenizerDirectory = "tokenizer",
            TemperaturePolicy = DecisionTemperaturePolicy.Default
        };
        var full = new PrepareDecisionInputs(
            profile,
            fixture.Tokenizer.Tokenizer,
            fixture.Tokenizer.Metadata).Prepare(
            string.Empty,
            [DecisionQuestion.Choice("team", "Which team?", ["billing", "support"])]);
        Assert.ThrowsException<ArgumentOutOfRangeException>(() =>
            fixture.Tokenizer.Tokenizer.EncodeToIds(
                "state", 0, out _, out _,
                considerPreTokenization: true,
                considerNormalization: true));

        var exactProfile = new LayaDecisionProfile
        {
            Name = profile.Name,
            Revision = profile.Revision,
            MaxLength = full.SequenceLength,
            HeadMaxLength = profile.HeadMaxLength,
            ModelFile = profile.ModelFile,
            TokenizerDirectory = profile.TokenizerDirectory,
            TemperaturePolicy = profile.TemperaturePolicy
        };
        var input = new PrepareDecisionInputs(
            exactProfile,
            fixture.Tokenizer.Tokenizer,
            fixture.Tokenizer.Metadata).Prepare(
            "a long state that has no room",
            [DecisionQuestion.Choice("team", "Which team?", ["billing", "support"])]);

        Assert.AreEqual(full.SequenceLength, input.SequenceLength);
        Assert.AreEqual(fixture.Tokenizer.SepTokenId, input.InputIds[^1]);
        Assert.IsTrue(input.MarkerMask[0]);
    }

    [TestMethod]
    public void LayaTokenizer_LoadsCurrentArrayFormBpeMerges()
    {
        using var fixture = ArrayMergeTokenizerFixture.Create();

        var encoded = fixture.Tokenizer.Tokenizer.EncodeToIds(
            "ab", considerPreTokenization: true, considerNormalization: true);

        CollectionAssert.AreEqual(new[] { 3 }, encoded.ToArray());
        Assert.AreEqual(7, fixture.Tokenizer.MaskTokenId);
        Assert.AreEqual(7, fixture.Tokenizer.SpecialTokens["[MASK]"]);
    }

    [TestMethod]
    public void DecodeDecisions_MapsChoiceScoreAndNoulWithConfidenceAndActionProbability()
    {
        var inputs = new DecisionInputBatch
        {
            BatchSize = 3,
            SequenceLength = 2,
            MarkerWidth = 3,
            InputIds = [1, 2, 3, 4, 5, 6],
            AttentionMask = [1, 1, 1, 1, 1, 1],
            MarkerPositions = [0, 1, 0, 1, 0, 1, 0, 0, 0],
            MarkerMask = [true, true, true, true, true, true, false, false, false],
            QuestionTypes = [0, 1, 2],
            Items =
            [
                new(0, DecisionQuestion.Choice("team", "Which?", ["billing", "support"]), [0, 1], ["billing", "support"], 0),
                new(0, DecisionQuestion.Score("urgency", "How urgent?", ["low", "high"]), [0, 1], ["0", "1"], 1),
                new(0, DecisionQuestion.Noul("risk", "Will it happen?"), [0, 1], ["false", "true"], 2)
            ]
        };
        var outputs = new DecisionModelOutputs
        {
            BatchSize = 3,
            MarkerWidth = 3,
            Logits = [0, 4, 0, 0, 10, 0, 0, 2, 0],
            ActionProbabilities = [0.8f, 0.2f, 0.3f, 0.7f, 0.1f, 0.9f]
        };

        var results = new DecodeDecisions(DecisionTemperaturePolicy.Default)
            .Decode(inputs, outputs).Results;

        Assert.IsInstanceOfType<ChoiceDecisionResult>(results[0]);
        Assert.AreEqual("support", ((ChoiceDecisionResult)results[0]).Choice);
        Assert.IsInstanceOfType<ScoreDecisionResult>(results[1]);
        Assert.AreEqual(1, ((ScoreDecisionResult)results[1]).Score, 0.001f);
        Assert.IsInstanceOfType<NoulDecisionResult>(results[2]);
        Assert.IsTrue(((NoulDecisionResult)results[2]).Value);
        Assert.AreEqual(0.9f, results[2].ActionProbability, 0.001f);
        Assert.IsTrue(results.All(static result => result.Confidence >= 0 && result.Confidence <= 1));
    }

    [TestMethod]
    public void DecisionTemperaturePolicy_ClampsAndReportsInvalidTemperatures()
    {
        var policy = new DecisionTemperaturePolicy
        {
            Minimum = 0.5f,
            Maximum = 5f,
            Temperatures = new Dictionary<string, float>
            {
                ["choice"] = 0.5f,
                ["choice:2"] = 20f,
                ["score"] = 1f,
                ["noul"] = float.NaN
            },
            Diagnostics =
            [
                new TypedDecisionDiagnostic("temperature-clamped", "choice:2 was clamped")
            ]
        };

        Assert.AreEqual(5f, policy.For(DecisionQuestionType.Choice, 2));
        Assert.AreEqual(1f, policy.For(DecisionQuestionType.Score, 2));
        Assert.AreEqual(1f, policy.For(DecisionQuestionType.Noul, 2));
        Assert.IsTrue(policy.Diagnostics.Any());
    }

    [TestMethod]
    public void DecisionJsonCodec_RoundTripsPreparedInputsAndScoredOutputs()
    {
        var question = DecisionQuestion.Choice("team", "Which?", ["billing", "support"]);
        var inputs = new DecisionInputBatch
        {
            BatchSize = 1,
            SequenceLength = 3,
            MarkerWidth = 2,
            InputIds = [1, 2, 3],
            AttentionMask = [1, 1, 0],
            MarkerPositions = [1, 2],
            MarkerMask = [true, true],
            QuestionTypes = [0],
            Items = [new(0, question, [1, 2], ["billing", "support"], 0)]
        };
        var outputs = new DecisionModelOutputs
        {
            BatchSize = 1,
            MarkerWidth = 2,
            Logits = [1, 2],
            ActionProbabilities = [0.6f, 0.4f]
        };

        var json = DecisionJsonCodec.SerializeScored(inputs, outputs);
        using (var document = JsonDocument.Parse(json))
        {
            var inputEnvelope = document.RootElement.GetProperty("inputs");
            Assert.IsTrue(inputEnvelope.TryGetProperty("inputIds", out _));
            Assert.IsTrue(inputEnvelope.TryGetProperty("attentionMask", out _));
            Assert.AreEqual(
                "Choice",
                inputEnvelope.GetProperty("items")[0]
                    .GetProperty("question")
                    .GetProperty("type")
                    .GetString());
            Assert.IsTrue(document.RootElement.GetProperty("outputs")
                .TryGetProperty("actionProbabilities", out _));
        }
        var roundTrip = DecisionJsonCodec.DeserializeScored(json);

        CollectionAssert.AreEqual(inputs.InputIds, roundTrip.Inputs.InputIds);
        CollectionAssert.AreEqual(outputs.Logits, roundTrip.Outputs.Logits);
        Assert.AreEqual("team", roundTrip.Inputs.Items[0].Question.Id);
    }

    [TestMethod]
    public void TypedDecisionBundle_WritesHashesAndRoundTripsThroughZip()
    {
        using var fixture = TinyTokenizerFixture.Create();
        var bundleDirectory = Path.Combine(fixture.Root, "bundle");
        var tokenizerDirectory = Path.Combine(bundleDirectory, "tokenizer");
        Directory.CreateDirectory(tokenizerDirectory);
        File.Copy(
            Path.Combine(fixture.Root, "tokenizer", "tokenizer.json"),
            Path.Combine(tokenizerDirectory, "tokenizer.json"));
        File.WriteAllBytes(Path.Combine(bundleDirectory, "laya.onnx"), [1, 2, 3]);
        File.WriteAllBytes(Path.Combine(bundleDirectory, "laya.onnx.data"), [4, 5, 6]);
        File.WriteAllText(
            Path.Combine(bundleDirectory, "laya_config.json"),
            """{"max_len":64,"head_max_len":16,"temperature":[1,1,1]}""");

        TypedDecisionBundle.WriteManifest(bundleDirectory);
        using (var directoryBundle = TypedDecisionBundle.Open(bundleDirectory))
        {
            Assert.AreEqual(1, directoryBundle.Manifest.FormatVersion);
            Assert.IsTrue(directoryBundle.Manifest.FileSha256.ContainsKey("laya.onnx"));
            Assert.AreEqual(64, directoryBundle.Profile.MaxLength);
        }

        var zipPath = Path.Combine(fixture.Root, "bundle.zip");
        TypedDecisionBundle.CreateZip(bundleDirectory, zipPath);
        using var zipBundle = TypedDecisionBundle.Open(zipPath);
        Assert.AreEqual("tokenizer", zipBundle.Manifest.TokenizerDirectory);
        Assert.IsTrue(File.Exists(zipBundle.ModelPath));
    }

    [TestMethod]
    public void TypedDecisionBundle_RejectsZipPathTraversal()
    {
        using var fixture = TinyTokenizerFixture.Create();
        var zipPath = Path.Combine(fixture.Root, "unsafe.zip");
        using (var archive = ZipFile.Open(zipPath, ZipArchiveMode.Create))
        using (var writer = new StreamWriter(archive.CreateEntry("../escape.txt").Open()))
            writer.Write("unsafe");

        Assert.ThrowsException<InvalidDataException>(() => TypedDecisionBundle.Open(zipPath));
    }

    private sealed class TinyTokenizerFixture : IDisposable
    {
        private TinyTokenizerFixture(string root, LayaTokenizer tokenizer)
        {
            Root = root;
            Tokenizer = tokenizer;
        }

        public string Root { get; }
        public LayaTokenizer Tokenizer { get; }

        public static TinyTokenizerFixture Create()
        {
            var root = Path.Combine(Path.GetTempPath(), "typed-decisions-tests", Guid.NewGuid().ToString("N"));
            var tokenizerDirectory = Path.Combine(root, "tokenizer");
            Directory.CreateDirectory(tokenizerDirectory);
            var tokenizerJson = new
            {
                version = "1.0",
                normalizer = new { type = "NFC" },
                pre_tokenizer = new { type = "ByteLevel", add_prefix_space = true },
                model = new
                {
                    type = "BPE",
                    vocab = new Dictionary<string, int>
                    {
                        ["<unk>"] = 0,
                        ["[PAD]"] = 1,
                        ["[CLS]"] = 2,
                        ["[SEP]"] = 3,
                        ["[MASK]"] = 4,
                        ["Ġ"] = 5,
                        ["a"] = 6,
                        ["b"] = 7,
                        ["c"] = 8,
                        ["d"] = 9,
                        ["e"] = 10,
                        ["f"] = 11,
                        ["g"] = 12,
                        ["h"] = 13,
                        ["i"] = 14,
                        ["j"] = 15,
                        ["k"] = 16,
                        ["l"] = 17,
                        ["m"] = 18,
                        ["n"] = 19,
                        ["o"] = 20,
                        ["p"] = 21,
                        ["q"] = 22,
                        ["r"] = 23,
                        ["s"] = 24,
                        ["t"] = 25,
                        ["u"] = 26,
                        ["v"] = 27,
                        ["w"] = 28,
                        ["x"] = 29,
                        ["y"] = 30,
                        ["z"] = 31
                    },
                    merges = Array.Empty<string>(),
                    unk_token = "<unk>"
                },
                added_tokens = new[]
                {
                    new { id = 1, content = "[PAD]", special = true },
                    new { id = 2, content = "[CLS]", special = true },
                    new { id = 3, content = "[SEP]", special = true },
                    new { id = 4, content = "[MASK]", special = true }
                }
            };
            File.WriteAllText(
                Path.Combine(tokenizerDirectory, "tokenizer.json"),
                JsonSerializer.Serialize(tokenizerJson));
            return new TinyTokenizerFixture(root, LayaTokenizer.Load(tokenizerDirectory));
        }

        public void Dispose()
        {
            if (Directory.Exists(Root))
                Directory.Delete(Root, recursive: true);
        }
    }

        private sealed class ArrayMergeTokenizerFixture : IDisposable
        {
            private ArrayMergeTokenizerFixture(string root, LayaTokenizer tokenizer)
            {
                Root = root;
                Tokenizer = tokenizer;
            }

            public string Root { get; }
            public LayaTokenizer Tokenizer { get; }

            public static ArrayMergeTokenizerFixture Create()
            {
                var root = Path.Combine(Path.GetTempPath(), "typed-decisions-array-merge-tests", Guid.NewGuid().ToString("N"));
                var tokenizerDirectory = Path.Combine(root, "tokenizer");
                Directory.CreateDirectory(tokenizerDirectory);
                var tokenizerJson = """
                    {
                      "version": "1.0",
                      "normalizer": {"type": "NFC"},
                      "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
                      "model": {
                        "type": "BPE",
                        "unk_token": "<unk>",
                        "vocab": {
                          "<unk>": 0,
                          "a": 1,
                          "b": 2,
                          "ab": 3,
                          "[PAD]": 4,
                          "[CLS]": 5,
                          "[SEP]": 6,
                          "[MASK]": 7
                        },
                        "merges": [["a", "b"]]
                      },
                      "added_tokens": [
                        {"id": 4, "content": "[PAD]", "special": true},
                        {"id": 5, "content": "[CLS]", "special": true},
                        {"id": 6, "content": "[SEP]", "special": true},
                        {"id": 7, "content": "[MASK]", "special": true},
                        {"id": 0, "content": "<unk>", "special": true}
                      ]
                    }
                    """;
                File.WriteAllText(Path.Combine(tokenizerDirectory, "tokenizer.json"), tokenizerJson);

                return new ArrayMergeTokenizerFixture(
                    root,
                    LayaTokenizer.Load(tokenizerDirectory));
            }

            public void Dispose()
            {
                if (Directory.Exists(Root))
                    Directory.Delete(Root, recursive: true);
            }
        }
}
