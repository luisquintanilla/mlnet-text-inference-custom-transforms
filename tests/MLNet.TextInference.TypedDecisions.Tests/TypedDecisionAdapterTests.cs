using System.Text.Json;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLNet.TextInference.Onnx;
using MLNet.TextInference.TypedDecisions;

namespace MLNet.TextInference.TypedDecisions.Tests;

[TestClass]
public sealed class TypedDecisionAdapterTests
{
    [TestMethod]
    public void FacadeEstimator_GetOutputSchema_AdvertisesTypedDecisionColumns()
    {
        var options = new OnnxTypedDecisionsOptions
        {
            BundlePath = "not-opened-by-schema-validation",
            Questions =
            [
                DecisionQuestion.Choice("team", "Which team?", ["billing", "support"])
            ]
        };
        var estimator = new OnnxTypedDecisionsEstimator(new MLContext(), options);
        var input = new SchemaShape(
        [
            CreateColumn("State", TextDataViewType.Instance)
        ]);

        var output = estimator.GetOutputSchema(input);

        CollectionAssert.IsSubsetOf(
            new[]
            {
                "State",
                options.ResultsColumnName,
                options.ChoiceColumnName,
                options.ScoreColumnName,
                options.ProbabilityTrueColumnName,
                options.ConfidenceColumnName,
                options.ActionProbabilityColumnName
            },
            output.Select(column => column.Name).ToArray());
    }

    [TestMethod]
    public void FacadeEstimator_Fit_ValidatesInputSchemaBeforeOpeningBundle()
    {
        var options = new OnnxTypedDecisionsOptions
        {
            BundlePath = "missing-bundle",
            Questions =
            [
                DecisionQuestion.Noul("risk", "Will the customer churn?")
            ]
        };
        var estimator = new OnnxTypedDecisionsEstimator(new MLContext(), options);
        var data = new MLContext().Data.LoadFromEnumerable(
            new[] { new MissingStateRow { Value = "state" } });

        Assert.ThrowsException<ArgumentException>(() => estimator.Fit(data));
    }

    [TestMethod]
    public void StagedEstimators_PropagateIntermediateTextColumns()
    {
        var ml = new MLContext();
        var prepare = new DecisionInputPreparationEstimator(
            ml,
            new DecisionInputPreparationOptions
            {
                BundlePath = "not-opened-by-schema-validation",
                Questions = [DecisionQuestion.Noul("risk", "Will the customer churn?")]
            });
        var score = new OnnxDecisionModelScorerEstimator(
            ml,
            new OnnxDecisionModelScorerOptions
            {
                BundlePath = "not-opened-by-schema-validation"
            });
        var input = new SchemaShape(
        [
            CreateColumn("State", TextDataViewType.Instance)
        ]);

        var prepared = prepare.GetOutputSchema(input);
        var scored = score.GetOutputSchema(prepared);

        Assert.IsTrue(prepared.Any(column => column.Name == "PreparedDecisionInputs"));
        Assert.IsTrue(scored.Any(column => column.Name == "ScoredDecisionOutputs"));
    }

    [TestMethod]
    public void DecodeDecisions_ProjectsMatchingScoreAndNoulScalars()
    {
        using var fixture = DecodeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var scoredJson = CreateScoredJson();
        var data = ml.Data.LoadFromEnumerable(
            new[] { new ScoredRow { ScoredDecisionOutputs = scoredJson } });

        var transformer = new DecisionDecodingEstimator(
            ml,
            new DecisionDecodingOptions { BundlePath = fixture.BundlePath })
            .Fit(data);
        var row = ml.Data.CreateEnumerable<DecodedRow>(
            transformer.Transform(data),
            reuseRowObject: false).Single();

        using var response = JsonDocument.Parse(row.DecisionResults);
        var results = response.RootElement.GetProperty("results");
        Assert.AreEqual("support", row.DecisionChoice);
        Assert.AreEqual(1f, row.DecisionScore, 0.001f);
        Assert.AreEqual(0.880797f, row.DecisionProbabilityTrue, 0.001f);
        Assert.IsTrue(float.IsFinite(row.DecisionScore));
        Assert.IsTrue(float.IsFinite(row.DecisionProbabilityTrue));
        Assert.AreEqual(
            row.DecisionScore,
            results.EnumerateArray()
                .Single(result => result.GetProperty("type").GetString() == "score")
                .GetProperty("score").GetSingle(),
            0.001f);
        Assert.AreEqual(
            row.DecisionProbabilityTrue,
            results.EnumerateArray()
                .Single(result => result.GetProperty("type").GetString() == "noul")
                .GetProperty("probability_true").GetSingle(),
            0.001f);
    }

    private static string CreateScoredJson()
    {
        var inputs = new DecisionInputBatch
        {
            BatchSize = 3,
            SequenceLength = 2,
            MarkerWidth = 2,
            InputIds = [1, 2, 3, 4, 5, 6],
            AttentionMask = [1, 1, 1, 1, 1, 1],
            MarkerPositions = [0, 1, 0, 1, 0, 1],
            MarkerMask = [true, true, true, true, true, true],
            QuestionTypes = [0, 1, 2],
            Items =
            [
                new(
                    0,
                    DecisionQuestion.Choice("team", "Which team?", ["billing", "support"]),
                    [0, 1],
                    ["billing", "support"],
                    0),
                new(
                    0,
                    DecisionQuestion.Score("urgency", "How urgent?", ["low", "high"]),
                    [0, 1],
                    ["0", "1"],
                    1),
                new(
                    0,
                    DecisionQuestion.Noul("risk", "Will the customer churn?"),
                    [0, 1],
                    ["false", "true"],
                    2)
            ]
        };
        var outputs = new DecisionModelOutputs
        {
            BatchSize = 3,
            MarkerWidth = 2,
            Logits = [0, 4, 0, 10, 0, 2],
            ActionProbabilities = [0.2f, 0.8f, 0.3f, 0.7f, 0.1f, 0.9f]
        };

        return DecisionJsonCodec.SerializeScored(inputs, outputs);
    }

    private static SchemaShape.Column CreateColumn(string name, DataViewType type)
    {
        var constructor = typeof(SchemaShape.Column).GetConstructors(
            BindingFlags.NonPublic | BindingFlags.Instance)[0];
        return (SchemaShape.Column)constructor.Invoke([
            name,
            SchemaShape.Column.VectorKind.Scalar,
            type,
            false,
            (SchemaShape?)null
        ]);
    }

    private sealed class MissingStateRow
    {
        public string Value { get; init; } = string.Empty;
    }

    private sealed class ScoredRow
    {
        public string ScoredDecisionOutputs { get; init; } = string.Empty;
    }

    private sealed class DecodedRow
    {
        public string DecisionResults { get; init; } = string.Empty;
        public string DecisionChoice { get; init; } = string.Empty;
        public float DecisionScore { get; init; }
        public float DecisionProbabilityTrue { get; init; }
    }

    private sealed class DecodeBundleFixture : IDisposable
    {
        private DecodeBundleFixture(string root)
        {
            Root = root;
            BundlePath = Path.Combine(root, "bundle");
        }

        public string Root { get; }
        public string BundlePath { get; }

        public static DecodeBundleFixture Create()
        {
            var fixture = new DecodeBundleFixture(Path.Combine(
                Path.GetTempPath(),
                "typed-decisions-adapter-tests",
                Guid.NewGuid().ToString("N")));
            var tokenizerDirectory = Path.Combine(fixture.BundlePath, "tokenizer");
            Directory.CreateDirectory(tokenizerDirectory);
            File.WriteAllBytes(Path.Combine(fixture.BundlePath, "laya.onnx"), [1]);
            File.WriteAllText(
                Path.Combine(fixture.BundlePath, "laya_config.json"),
                """{"max_len":64,"head_max_len":16,"temperature":[1,1,1]}""");
            File.WriteAllText(
                Path.Combine(tokenizerDirectory, "tokenizer.json"),
                """
                {
                  "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "vocab": {
                      "<unk>": 0,
                      "[PAD]": 1,
                      "[CLS]": 2,
                      "[SEP]": 3,
                      "[MASK]": 4,
                      "a": 5,
                      "b": 6
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
            TypedDecisionBundle.WriteManifest(
                fixture.BundlePath,
                new TypedDecisionBundleManifest
                {
                    ExternalDataFiles = [],
                    Profile = new TypedDecisionBundleProfile("fixture", "fixture")
                });
            return fixture;
        }

        public void Dispose()
        {
            if (Directory.Exists(Root))
                Directory.Delete(Root, recursive: true);
        }
    }
}
