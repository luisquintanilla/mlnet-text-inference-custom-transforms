using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLNet.TextInference.Onnx;
using MLNet.TextInference.TypedDecisions;
using static MLNet.TextInference.TypedDecisions.DecisionQuestion;

namespace MLNet.TextInference.TypedDecisions.Tests;

[TestClass]
public sealed class TypedDecisionNativeDataViewTests
{
    [TestMethod]
    public void Facade_DirectAndDataViewPathsUseTheSameDecodedResults()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var questions = Questions();
        var states = new[]
        {
            "The first state has enough evidence.",
            "The second state needs more evidence."
        };
        var data = ml.Data.LoadFromEnumerable(
            states.Select(static state => new StateRow { State = state }));
        var options = fixture.CreateOptions(questions);

        using var transformer = ml.Transforms.OnnxTypedDecisions(options).Fit(data);
        var direct = transformer.Infer(states);
        var rows = ml.Data.CreateEnumerable<DecisionRow>(
            transformer.Transform(data),
            reuseRowObject: false).ToArray();

        Assert.AreEqual(states.Length, rows.Length);
        for (var index = 0; index < rows.Length; index++)
        {
            using var json = JsonDocument.Parse(rows[index].DecisionResults);
            Assert.AreEqual(
                direct[index].Results.Count,
                json.RootElement.GetProperty("results").GetArrayLength());
            Assert.IsTrue(float.IsFinite(rows[index].Decision_quality_Score));
            Assert.IsTrue(rows[index].Decision_actionable_Probability is >= 0 and <= 1);
            Assert.AreEqual(
                direct[index].Results.OfType<ScoreDecisionResult>().Single().Score,
                rows[index].Decision_quality_Score,
                0.000001f);
            Assert.AreEqual(
                direct[index].Results.OfType<NoulDecisionResult>().Single().Value,
                rows[index].Decision_actionable_PredictedLabel);
        }
    }

    [TestMethod]
    public void Stages_UseNativeTensorColumnsAndPreserveAllQuestions()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var questions = Questions();
        var data = ml.Data.LoadFromEnumerable(
            new[]
            {
                new StateRow { State = "A state with enough evidence." },
                new StateRow { State = "Another state." }
            });

        using var preparation = ml.Transforms.PrepareDecisionInputs(
            new DecisionInputPreparationOptions
            {
                ModelAssetsPath = fixture.ModelAssetsPath,
                Questions = questions
            }).Fit(data);
        using var scoring = ml.Transforms.ScoreOnnxDecisionModel(
            new OnnxDecisionModelScorerOptions { ModelAssetsPath = fixture.ModelAssetsPath })
            .Fit(preparation.Transform(data));
        using var decoding = ml.Transforms.DecodeDecisions(
            new DecisionDecodingOptions
            {
                ModelAssetsPath = fixture.ModelAssetsPath,
                Questions = questions
            }).Fit(scoring.Transform(preparation.Transform(data)));

        var output = decoding.Transform(scoring.Transform(preparation.Transform(data)));
        var schema = output.Schema;
        Assert.IsInstanceOfType(schema["DecisionInputIds"].Type, typeof(VectorDataViewType));
        Assert.AreEqual(NumberDataViewType.Int64,
            ((VectorDataViewType)schema["DecisionInputIds"].Type).ItemType);
        Assert.AreEqual(BooleanDataViewType.Instance,
            ((VectorDataViewType)schema["DecisionMarkerMask"].Type).ItemType);
        Assert.AreEqual(NumberDataViewType.Single,
            ((VectorDataViewType)schema["DecisionLogits"].Type).ItemType);

        var rows = ml.Data.CreateEnumerable<StageRow>(output, reuseRowObject: false).ToArray();
        Assert.AreEqual(2, rows.Length);
        Assert.IsTrue(rows.All(row =>
            row.DecisionBatchSize == questions.Count &&
            row.DecisionInputIds.Length == row.DecisionBatchSize * row.DecisionSequenceLength &&
            row.DecisionLogits.Length == row.DecisionBatchSize * row.DecisionMarkerWidth));
        Assert.IsTrue(rows.All(row => row.DecisionResults.Contains("\"results\"", StringComparison.Ordinal)));
    }

    [TestMethod]
    public void Facade_BatchesRowsAndPreservesUpstreamVectorAndLabelColumns()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var questions = Questions();
        var data = ml.Data.LoadFromEnumerable(
            Enumerable.Range(0, 5).Select(index => new FeatureRow
            {
                State = $"State {index}",
                Label = index % 2 == 0,
                Features = new VBuffer<float>(3, [index, index + 1, index + 2])
            }));
        var options = fixture.CreateOptions(questions);
        options = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = options.ModelAssetsPath,
            Questions = options.Questions,
            BatchSize = 2
        };

        using var transformer = ml.Transforms.OnnxTypedDecisions(options).Fit(data);
        var output = transformer.Transform(data);
        var rows = ml.Data.CreateEnumerable<FeatureOutputRow>(
            output,
            reuseRowObject: false).ToArray();

        Assert.AreEqual(5, rows.Length);
        for (var index = 0; index < rows.Length; index++)
        {
            Assert.AreEqual(index % 2 == 0, rows[index].Label);
            CollectionAssert.AreEqual(
                new[] { (float)index, index + 1f, index + 2f },
                rows[index].Features.DenseValues().ToArray());
            Assert.IsFalse(string.IsNullOrWhiteSpace(rows[index].DecisionResults));
        }
    }

    [TestMethod]
    public void ModelDirectoryWithoutManifest_IsAccepted()
    {
        using var fixture = NativeBundleFixture.Create();
        using var bundle = TypedDecisionBundle.Open(fixture.ModelAssetsPath);

        Assert.AreEqual(LayaDecisionProfile.EnglishFp32.Name, bundle.Manifest.Profile.Name);
        Assert.AreEqual("laya.onnx", bundle.Manifest.ModelFile);
        Assert.IsTrue(File.Exists(bundle.ModelPath));
    }

    [TestMethod]
    public void Facade_ReportsUnsupportedRowMapperCapabilityConsistently()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(
            new[] { new StateRow { State = "state" } });
        using var transformer = ml.Transforms.OnnxTypedDecisions(
            fixture.CreateOptions([Noul("risk", "Will the customer churn?")]))
            .Fit(data);

        Assert.IsFalse(transformer.IsRowToRowMapper);
        Assert.ThrowsException<NotSupportedException>(() =>
            transformer.GetRowToRowMapper(data.Schema));
    }

    private static IReadOnlyList<DecisionQuestion> Questions() =>
    [
        Choice("priority", "How urgent?", ["low", "high"]),
        Score("quality", "How strong?", ["weak", "moderate", "strong"]),
        Noul("actionable", "Can it be acted on?")
    ];

    public sealed class StateRow
    {
        public string State { get; set; } = string.Empty;
    }

    public class DecisionRow
    {
        public string DecisionResults { get; set; } = string.Empty;
        public float Decision_quality_Score { get; set; }
        public bool Decision_actionable_PredictedLabel { get; set; }
        public float Decision_actionable_Probability { get; set; }
    }

    public sealed class StageRow : DecisionRow
    {
        public VBuffer<long> DecisionInputIds { get; set; }
        public VBuffer<bool> DecisionMarkerMask { get; set; }
        public int DecisionBatchSize { get; set; }
        public int DecisionSequenceLength { get; set; }
        public int DecisionMarkerWidth { get; set; }
        public VBuffer<float> DecisionLogits { get; set; }
    }

    public sealed class FeatureRow
    {
        public string State { get; set; } = string.Empty;
        public bool Label { get; set; }
        public VBuffer<float> Features { get; set; }
    }

    public sealed class FeatureOutputRow
    {
        public bool Label { get; set; }
        public VBuffer<float> Features { get; set; }
        public string DecisionResults { get; set; } = string.Empty;
    }

    private sealed class NativeBundleFixture : IDisposable
    {
        private const string ModelBase64 =
            "CAgSB2ZpeHR1cmU6sgMKFgoFcXR5cGUSBnFzaGFwZSIFU2hhcGUKLQoGcXNoYXBlCgN0d28SCWFjdF9zaGFwZSIGQ29uY2F0KgsKBGF4aXMYAKABAgpFCglhY3Rfc2hhcGUSCWFjdF9wcm9icyIPQ29uc3RhbnRPZlNoYXBlKhwKBXZhbHVlKhAIARABIgQAAAAAQgR6ZXJvoAEECiUKCm1hcmtlcl9wb3MSBmxvZ2l0cyIEQ2FzdCoJCgJ0bxgBoAECEgdmaXh0dXJlKgwIARAHOgECQgN0d28qEAgBEAEiBAAAAABCBHplcm9aHQoJaW5wdXRfaWRzEhAKDggHEgoKAxIBQgoDEgFMWiIKDmF0dGVudGlvbl9tYXNrEhAKDggHEgoKAxIBQgoDEgFMWh4KCm1hcmtlcl9wb3MSEAoOCAcSCgoDEgFCCgMSAUtaHwoLbWFya2VyX21hc2sSEAoOCAkSCgoDEgFCCgMSAUtaFAoFcXR5cGUSCwoJCAcSBQoDEgFCYhoKBmxvZ2l0cxIQCg4IARIKCgMSAUIKAxIBS2IcCglhY3RfcHJvYnMSDwoNCAESCQoDEgFCCgIIAkIECgAQDQ==";

        private NativeBundleFixture(string root)
        {
            Root = root;
            ModelAssetsPath = Path.Combine(root, "bundle");
        }

        public string Root { get; }
        public string ModelAssetsPath { get; }

        public static NativeBundleFixture Create()
        {
            var fixture = new NativeBundleFixture(Path.Combine(
                Path.GetTempPath(),
                "typed-decisions-native-tests",
                Guid.NewGuid().ToString("N")));
            var tokenizerDirectory = Path.Combine(fixture.ModelAssetsPath, "tokenizer");
            Directory.CreateDirectory(tokenizerDirectory);
            File.WriteAllBytes(
                Path.Combine(fixture.ModelAssetsPath, "laya.onnx"),
                Convert.FromBase64String(ModelBase64));
            File.WriteAllText(
                Path.Combine(fixture.ModelAssetsPath, "laya_config.json"),
                """{"max_len":64,"head_max_len":16,"temperature":[1,1,1]}""");
            File.WriteAllText(
                Path.Combine(tokenizerDirectory, "tokenizer.json"),
                """
                {
                  "version": "1.0",
                  "normalizer": {"type": "NFC"},
                  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": true},
                  "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "vocab": {
                      "<unk>": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
                      "Ġ": 5, "a": 6, "b": 7, "c": 8, "d": 9, "e": 10, "f": 11,
                      "g": 12, "h": 13, "i": 14, "j": 15, "k": 16, "l": 17, "m": 18,
                      "n": 19, "o": 20, "p": 21, "q": 22, "r": 23, "s": 24, "t": 25,
                      "u": 26, "v": 27, "w": 28, "x": 29, "y": 30, "z": 31
                    },
                    "merges": []
                  },
                  "added_tokens": [
                    {"id": 1, "content": "[PAD]", "special": true},
                    {"id": 2, "content": "[CLS]", "special": true},
                    {"id": 3, "content": "[SEP]", "special": true},
                    {"id": 4, "content": "[MASK]", "special": true}
                  ]
                }
                """);
            return fixture;
        }

        public OnnxTypedDecisionsOptions CreateOptions(IReadOnlyList<DecisionQuestion> questions) =>
            new()
            {
                ModelAssetsPath = ModelAssetsPath,
                Questions = questions
            };

        public void Dispose()
        {
            if (Directory.Exists(Root))
                Directory.Delete(Root, recursive: true);
        }
    }
}
