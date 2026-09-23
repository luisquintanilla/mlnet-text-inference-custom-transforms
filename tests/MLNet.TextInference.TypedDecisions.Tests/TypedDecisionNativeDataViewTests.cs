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
    public void CopyValue_DeepCopiesDenseAndTextVectors()
    {
        var floatValues = new[] { 1f, 2f };
        var floatSnapshot = DecisionDataViewUtils.CopyValue(new VBuffer<float>(2, floatValues));
        floatValues[0] = 99f;

        var doubleValues = new[] { 3d, 4d };
        var doubleSnapshot = DecisionDataViewUtils.CopyValue(new VBuffer<double>(2, doubleValues));
        doubleValues[0] = 98d;

        var textBuffer = "first-second".ToCharArray();
        ReadOnlyMemory<char>[] textValues =
        {
            textBuffer.AsMemory(0, 5),
            textBuffer.AsMemory(6, 6)
        };
        var textSnapshot = DecisionDataViewUtils.CopyValue(
            new VBuffer<ReadOnlyMemory<char>>(2, textValues));
        textBuffer[0] = 'x';

        CollectionAssert.AreEqual(new[] { 1f, 2f }, floatSnapshot.DenseValues().ToArray());
        CollectionAssert.AreEqual(new[] { 3d, 4d }, doubleSnapshot.DenseValues().ToArray());
        CollectionAssert.AreEqual(
            new[] { "first", "second" },
            textSnapshot.GetValues().ToArray().Select(static value => value.ToString()).ToArray());
    }

    [TestMethod]
    public void Facade_CopiesReusedDenseSparseDoubleAndTextVectorsAcrossLookaheadBatches()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var data = new ReusedBufferDataView(rowCount: 5);
        var options = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = fixture.ModelAssetsPath,
            Questions = [Noul("risk", "Can it be acted on?")],
            BatchSize = 2
        };

        using var transformer = ml.Transforms.OnnxTypedDecisions(options).Fit(data);
        var rows = ml.Data.CreateEnumerable<ReusedOutputRow>(
            transformer.Transform(data),
            reuseRowObject: false).ToArray();

        Assert.AreEqual(5, rows.Length);
        for (var index = 0; index < rows.Length; index++)
        {
            Assert.AreEqual($"state-{index}", rows[index].State);
            CollectionAssert.AreEqual(
                new[] { (float)index, index + 0.5f, index + 1f },
                rows[index].Features.DenseValues().ToArray());
            CollectionAssert.AreEqual(
                new[] { (double)index + 0.25d, index + 0.75d },
                rows[index].DoubleFeatures.DenseValues().ToArray());
            Assert.IsFalse(rows[index].SparseFeatures.IsDense);
            CollectionAssert.AreEqual(
                new[] { 1, 4 },
                rows[index].SparseFeatures.GetIndices().ToArray());
            CollectionAssert.AreEqual(
                new[] { (float)index + 10f, index + 20f },
                rows[index].SparseFeatures.GetValues().ToArray());
            CollectionAssert.AreEqual(
                new[] { $"row-{index}-a", $"row-{index}-b" },
                rows[index].TextFeatures.GetValues()
                    .ToArray()
                    .Select(static value => value.ToString())
                    .ToArray());
        }
    }

    [TestMethod]
    public void Facade_CachesActivePassThroughGetterRequestedAfterMoveNext()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var data = new ReusedBufferDataView(rowCount: 3);
        var options = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = fixture.ModelAssetsPath,
            Questions = [Noul("risk", "Can it be acted on?")],
            BatchSize = 2
        };

        using var transformer = ml.Transforms.OnnxTypedDecisions(options).Fit(data);
        var transformed = transformer.Transform(data);
        using var cursor = transformed.GetRowCursor([transformed.Schema["Features"]]);

        Assert.IsTrue(cursor.MoveNext());
        var getter = cursor.GetGetter<VBuffer<float>>(transformed.Schema["Features"]);
        VBuffer<float> features = default;
        getter(ref features);

        CollectionAssert.AreEqual(
            new[] { 0f, 0.5f, 1f },
            features.DenseValues().ToArray());
    }

    [TestMethod]
    public void TypedCursors_RejectGettersForInactiveColumns()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var data = new ReusedBufferDataView(rowCount: 1);
        var question = Noul("risk", "Can it be acted on?");

        using var facade = ml.Transforms.OnnxTypedDecisions(
            fixture.CreateOptions([question])).Fit(data);
        var facadeOutput = facade.Transform(data);
        using (var cursor = facadeOutput.GetRowCursor([facadeOutput.Schema["Features"]]))
        {
            Assert.ThrowsException<InvalidOperationException>(() =>
                cursor.GetGetter<ReadOnlyMemory<char>>(
                    facadeOutput.Schema["DecisionResults"]));
        }

        using var preparation = ml.Transforms.PrepareDecisionInputs(
                new DecisionInputPreparationOptions
                {
                    ModelAssetsPath = fixture.ModelAssetsPath,
                    Questions = [question]
                })
            .Fit(data);
        var prepared = preparation.Transform(data);
        using (var cursor = prepared.GetRowCursor([prepared.Schema["Features"]]))
        {
            Assert.ThrowsException<InvalidOperationException>(() =>
                cursor.GetGetter<VBuffer<long>>(prepared.Schema["DecisionInputIds"]));
        }

        using var scorer = ml.Transforms.ScoreOnnxDecisionModel(
                new OnnxDecisionModelScorerOptions
                {
                    ModelAssetsPath = fixture.ModelAssetsPath,
                    BatchSize = 2
                })
            .Fit(prepared);
        var scored = scorer.Transform(prepared);
        using (var cursor = scored.GetRowCursor([scored.Schema["Features"]]))
        {
            Assert.ThrowsException<InvalidOperationException>(() =>
                cursor.GetGetter<VBuffer<float>>(scored.Schema["DecisionLogits"]));
        }
    }

    [TestMethod]
    public void StagedScoring_CopiesReusedVectorsAcrossLookaheadBatches()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var data = new ReusedBufferDataView(rowCount: 5);
        var question = Noul("risk", "Can it be acted on?");
        var preparedOptions = new DecisionInputPreparationOptions
        {
            ModelAssetsPath = fixture.ModelAssetsPath,
            Questions = [question]
        };

        using var preparation = ml.Transforms.PrepareDecisionInputs(preparedOptions).Fit(data);
        var prepared = preparation.Transform(data);
        using var scorer = ml.Transforms.ScoreOnnxDecisionModel(
                new OnnxDecisionModelScorerOptions
                {
                    ModelAssetsPath = fixture.ModelAssetsPath,
                    BatchSize = 2
                })
            .Fit(prepared);
        var scored = scorer.Transform(prepared);
        var rows = ml.Data.CreateEnumerable<ReusedStageOutputRow>(
            scored,
            reuseRowObject: false).ToArray();

        Assert.AreEqual(5, rows.Length);
        for (var index = 0; index < rows.Length; index++)
        {
            Assert.AreEqual($"state-{index}", rows[index].State);
            CollectionAssert.AreEqual(
                new[] { (float)index, index + 0.5f, index + 1f },
                rows[index].Features.DenseValues().ToArray());
            CollectionAssert.AreEqual(
                new[] { (double)index + 0.25d, index + 0.75d },
                rows[index].DoubleFeatures.DenseValues().ToArray());
            Assert.IsFalse(rows[index].SparseFeatures.IsDense);
            CollectionAssert.AreEqual(
                new[] { 1, 4 },
                rows[index].SparseFeatures.GetIndices().ToArray());
            CollectionAssert.AreEqual(
                new[] { (float)index + 10f, index + 20f },
                rows[index].SparseFeatures.GetValues().ToArray());
            CollectionAssert.AreEqual(
                new[] { $"row-{index}-a", $"row-{index}-b" },
                rows[index].TextFeatures.GetValues()
                    .ToArray()
                    .Select(static value => value.ToString())
                    .ToArray());
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

    public sealed class ReusedOutputRow
    {
        public string State { get; set; } = string.Empty;
        public VBuffer<float> Features { get; set; }
        public VBuffer<double> DoubleFeatures { get; set; }
        public VBuffer<float> SparseFeatures { get; set; }
        public VBuffer<ReadOnlyMemory<char>> TextFeatures { get; set; }
        public string DecisionResults { get; set; } = string.Empty;
    }

    public sealed class ReusedStageOutputRow
    {
        public string State { get; set; } = string.Empty;
        public VBuffer<float> Features { get; set; }
        public VBuffer<double> DoubleFeatures { get; set; }
        public VBuffer<float> SparseFeatures { get; set; }
        public VBuffer<ReadOnlyMemory<char>> TextFeatures { get; set; }
        public VBuffer<float> DecisionLogits { get; set; }
    }

    private sealed class ReusedBufferDataView : IDataView
    {
        private readonly int _rowCount;

        public ReusedBufferDataView(int rowCount)
        {
            _rowCount = rowCount;
            var builder = new DataViewSchema.Builder();
            builder.AddColumn("State", TextDataViewType.Instance);
            builder.AddColumn("Features", new VectorDataViewType(NumberDataViewType.Single));
            builder.AddColumn("DoubleFeatures", new VectorDataViewType(NumberDataViewType.Double));
            builder.AddColumn("SparseFeatures", new VectorDataViewType(NumberDataViewType.Single));
            builder.AddColumn("TextFeatures", new VectorDataViewType(TextDataViewType.Instance));
            Schema = builder.ToSchema();
        }

        public DataViewSchema Schema { get; }
        public bool CanShuffle => false;
        public long? GetRowCount() => _rowCount;

        public DataViewRowCursor GetRowCursor(
            IEnumerable<DataViewSchema.Column> columnsNeeded,
            Random? rand = null)
            => new Cursor(this, columnsNeeded);

        public DataViewRowCursor[] GetRowCursorSet(
            IEnumerable<DataViewSchema.Column> columnsNeeded,
            int n,
            Random? rand = null)
            => [GetRowCursor(columnsNeeded, rand)];

        private sealed class Cursor : DataViewRowCursor
        {
            private readonly ReusedBufferDataView _parent;
            private readonly HashSet<string> _columnsNeeded;
            private readonly char[] _stateBuffer = new char[32];
            private readonly float[] _floatValues = new float[3];
            private readonly double[] _doubleValues = new double[2];
            private readonly char[] _textBuffer = new char[64];
            private readonly ReadOnlyMemory<char>[] _textValues = new ReadOnlyMemory<char>[2];
            private VBuffer<float> _features;
            private VBuffer<double> _doubleFeatures;
            private VBuffer<float> _sparseFeatures;
            private VBuffer<ReadOnlyMemory<char>> _textFeatures;
            private ReadOnlyMemory<char> _state;
            private int _index = -1;

            public Cursor(
                ReusedBufferDataView parent,
                IEnumerable<DataViewSchema.Column> columnsNeeded)
            {
                _parent = parent;
                _columnsNeeded = columnsNeeded
                    .Select(static column => column.Name)
                    .ToHashSet(StringComparer.Ordinal);
            }

            public override DataViewSchema Schema => _parent.Schema;
            public override long Position => _index;
            public override long Batch => 0;

            public override bool MoveNext()
            {
                _index++;
                if (_index >= _parent._rowCount)
                    return false;

                var state = $"state-{_index}";
                state.AsSpan().CopyTo(_stateBuffer);
                _state = _stateBuffer.AsMemory(0, state.Length);

                _floatValues[0] = _index;
                _floatValues[1] = _index + 0.5f;
                _floatValues[2] = _index + 1f;
                SetDense(ref _features, _floatValues);

                _doubleValues[0] = _index + 0.25d;
                _doubleValues[1] = _index + 0.75d;
                SetDense(ref _doubleFeatures, _doubleValues);

                var sparseEditor = VBufferEditor.Create(ref _sparseFeatures, 6, 2);
                sparseEditor.Values[0] = _index + 10f;
                sparseEditor.Values[1] = _index + 20f;
                sparseEditor.Indices[0] = 1;
                sparseEditor.Indices[1] = 4;
                _sparseFeatures = sparseEditor.Commit();

                var first = $"row-{_index}-a";
                var second = $"row-{_index}-b";
                first.AsSpan().CopyTo(_textBuffer);
                second.AsSpan().CopyTo(_textBuffer.AsSpan(first.Length));
                _textValues[0] = _textBuffer.AsMemory(0, first.Length);
                _textValues[1] = _textBuffer.AsMemory(first.Length, second.Length);
                var textEditor = VBufferEditor.Create(ref _textFeatures, 2);
                _textValues.AsSpan().CopyTo(textEditor.Values);
                _textFeatures = textEditor.Commit();
                return true;
            }

            public override ValueGetter<TValue> GetGetter<TValue>(
                DataViewSchema.Column column)
            {
                return column.Name switch
                {
                    "State" => Cast<ReadOnlyMemory<char>, TValue>(
                        (ref ReadOnlyMemory<char> value) => value = _state),
                    "Features" => Cast<VBuffer<float>, TValue>(
                        (ref VBuffer<float> value) => value = _features),
                    "DoubleFeatures" => Cast<VBuffer<double>, TValue>(
                        (ref VBuffer<double> value) => value = _doubleFeatures),
                    "SparseFeatures" => Cast<VBuffer<float>, TValue>(
                        (ref VBuffer<float> value) => value = _sparseFeatures),
                    "TextFeatures" => Cast<VBuffer<ReadOnlyMemory<char>>, TValue>(
                        (ref VBuffer<ReadOnlyMemory<char>> value) => value = _textFeatures),
                    _ => throw new InvalidOperationException($"Unknown column '{column.Name}'.")
                };
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
                => (ref DataViewRowId value) => value = new DataViewRowId((ulong)_index, 0);

            public override bool IsColumnActive(DataViewSchema.Column column)
                => _columnsNeeded.Contains(column.Name);

            protected override void Dispose(bool disposing)
                => base.Dispose(disposing);

            private static ValueGetter<TValue> Cast<TSource, TValue>(
                ValueGetter<TSource> getter)
                => (ValueGetter<TValue>)(object)getter;

            private static void SetDense<T>(ref VBuffer<T> target, T[] values)
            {
                var editor = VBufferEditor.Create(ref target, values.Length);
                values.AsSpan().CopyTo(editor.Values);
                target = editor.Commit();
            }
        }
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
