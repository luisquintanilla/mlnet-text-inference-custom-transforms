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
    public void Facade_PassthroughOnlyProjectionDoesNotReadStateOrInferenceInputs()
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
        var output = transformer.Transform(data);
        var features = output.Schema["Features"];
        using var cursor = output.GetRowCursor([features]);
        var getter = cursor.GetGetter<VBuffer<float>>(features);

        while (cursor.MoveNext())
        {
            VBuffer<float> value = default;
            getter(ref value);
        }

        Assert.AreEqual(0, data.StateReadCount);
    }

    [TestMethod]
    public void Facade_PreservesUpstreamRowIdsAcrossLookaheadBatches()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var data = new ReusedBufferDataView(rowCount: 5);
        using var transformer = ml.Transforms.OnnxTypedDecisions(
            new OnnxTypedDecisionsOptions
            {
                ModelAssetsPath = fixture.ModelAssetsPath,
                Questions = [Noul("risk", "Can it be acted on?")],
                BatchSize = 2
            }).Fit(data);

        var output = transformer.Transform(data);
        var features = output.Schema["Features"];
        using var cursor = output.GetRowCursor([features]);
        var idGetter = cursor.GetIdGetter();
        for (var index = 0; index < 5; index++)
        {
            Assert.IsTrue(cursor.MoveNext());
            DataViewRowId id = default;
            idGetter(ref id);
            Assert.AreEqual((ulong)index, id.Low);
            Assert.AreEqual(index / 2, cursor.Batch);
        }
        Assert.IsFalse(cursor.MoveNext());
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
    public void StagedScoring_PassthroughProjectionDoesNotReadInferenceInputsAndPreservesBatches()
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
        var state = scored.Schema["State"];
        var features = scored.Schema["Features"];
        var logits = scored.Schema["DecisionLogits"];
        using var cursor = scored.GetRowCursor([state, features]);
        Assert.ThrowsExactly<InvalidOperationException>(
            () => cursor.GetGetter<VBuffer<float>>(logits));
        var stateGetter = cursor.GetGetter<ReadOnlyMemory<char>>(state);
        var featureGetter = cursor.GetGetter<VBuffer<float>>(features);
        var batches = new List<long>();
        for (var index = 0; index < 5; index++)
        {
            Assert.IsTrue(cursor.MoveNext());
            batches.Add(cursor.Batch);
            ReadOnlyMemory<char> stateValue = default;
            VBuffer<float> featureValue = default;
            stateGetter(ref stateValue);
            featureGetter(ref featureValue);
            Assert.AreEqual($"state-{index}", stateValue.ToString());
            CollectionAssert.AreEqual(
                new[] { (float)index, index + 0.5f, index + 1f },
                featureValue.DenseValues().ToArray());
        }

        Assert.IsFalse(cursor.MoveNext());
        CollectionAssert.AreEqual(new long[] { 0, 0, 1, 1, 2 }, batches);
        Assert.AreEqual(5, data.StateReadCount);
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
    public void Facade_ReportsRowMapperCapabilityConsistently()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(
            new[] { new StateRow { State = "state" } });
        using var transformer = ml.Transforms.OnnxTypedDecisions(
            fixture.CreateOptions([Noul("risk", "Will the customer churn?")]))
            .Fit(data);

        Assert.IsTrue(transformer.IsRowToRowMapper);
        var mapper = transformer.GetRowToRowMapper(data.Schema);
        Assert.AreSame(data.Schema, mapper.InputSchema);
    }

    [TestMethod]
    public void Facade_PredictionEngine_MapsAllTypedOutputsAcrossRepeatedStates()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var questions = Questions();
        var states = new[]
        {
            "a",
            "The customer supplied reproducible steps and requested an urgent fix with complete logs."
        };
        var data = ml.Data.LoadFromEnumerable(
            states.Select(static state => new StateRow { State = state }));
        var options = fixture.CreateOptions(questions);

        using var transformer = ml.Transforms.OnnxTypedDecisions(options).Fit(data);
        Assert.IsTrue(transformer.IsRowToRowMapper);
        var direct = transformer.Infer(states).ToArray();
        Assert.AreNotEqual(
            direct[0].InputTokenCount,
            direct[1].InputTokenCount,
            "The repeated PredictionEngine case must observe distinct prepared rows.");

        using (var engine = ml.Model.CreatePredictionEngine<StateRow, FullDecisionRow>(
            transformer,
            new PredictionEngineOptions { OwnsTransformer = false }))
        {
            for (var repeat = 0; repeat < 3; repeat++)
            {
                for (var index = 0; index < states.Length; index++)
                {
                    var prediction = engine.Predict(new StateRow { State = states[index] });
                    AssertDecisionRowMatches(prediction, direct[index]);
                }
            }
        }
        Assert.AreEqual(states.Length, transformer.Infer(states).Count);
    }

    [TestMethod]
    public void Stages_PredictionEngine_MapsPreparationScoringAndDecoding()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var questions = Questions();
        var states = new[]
        {
            "short",
            "A materially longer state containing enough evidence to change token counts."
        };
        var data = ml.Data.LoadFromEnumerable(
            states.Select(static state => new StateRow { State = state }));
        var prepared = new DecisionInputPreparationOptions
        {
            ModelAssetsPath = fixture.ModelAssetsPath,
            Questions = questions
        };
        var scored = new OnnxDecisionModelScorerOptions
        {
            ModelAssetsPath = fixture.ModelAssetsPath
        };
        var decoded = new DecisionDecodingOptions
        {
            ModelAssetsPath = fixture.ModelAssetsPath,
            Questions = questions
        };
        var pipeline = ml.Transforms.PrepareDecisionInputs(prepared)
            .Append(ml.Transforms.ScoreOnnxDecisionModel(scored))
            .Append(ml.Transforms.DecodeDecisions(decoded));
        using var transformer = pipeline.Fit(data);

        using var directTransformer = ml.Transforms.OnnxTypedDecisions(
            fixture.CreateOptions(questions)).Fit(data);
        var direct = directTransformer.Infer(states).ToArray();
        using var engine = ml.Model.CreatePredictionEngine<StateRow, FullStageRow>(
            transformer,
            new PredictionEngineOptions { OwnsTransformer = false });
        for (var repeat = 0; repeat < 3; repeat++)
        {
            for (var index = 0; index < states.Length; index++)
            {
                var prediction = engine.Predict(new StateRow { State = states[index] });
                Assert.IsTrue(prediction.DecisionInputIds.Length > 0);
                Assert.IsTrue(prediction.DecisionLogits.Length > 0);
                AssertDecisionRowMatches(prediction, direct[index]);
            }
        }
    }

    [TestMethod]
    public void Composed_PredictionEngine_MapsOriginalAndAppendedOutputs()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var questions = Questions();
        var states = new[]
        {
            "composed state",
            "a much longer composed state with additional context"
        };
        var data = ml.Data.LoadFromEnumerable(
            states.Select(static state => new StateRow { State = state }));
        var original = fixture.CreateOptions(questions);
        var appended = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = fixture.ModelAssetsPath,
            Questions = questions,
            OutputPrefix = "AppendedDecision_",
            ResultsColumnName = "AppendedDecisionResults"
        };
        var transformer = ml.Transforms.OnnxTypedDecisions(original)
            .AppendOnnxTypedDecisions(ml, appended)
            .Fit(data);
        try
        {
            using var directTransformer = ml.Transforms.OnnxTypedDecisions(original).Fit(data);
            var direct = directTransformer.Infer(states).ToArray();
            using var engine = ml.Model.CreatePredictionEngine<StateRow, FullComposedRow>(
                transformer,
                new PredictionEngineOptions { OwnsTransformer = false });
            for (var repeat = 0; repeat < 3; repeat++)
            {
                for (var index = 0; index < states.Length; index++)
                {
                    var prediction = engine.Predict(new StateRow { State = states[index] });
                    AssertDecisionRowMatches(prediction, direct[index]);
                    AssertAppendedDecisionRowMatches(prediction, direct[index]);
                }
            }
        }
        finally
        {
            (transformer as IDisposable)?.Dispose();
        }
    }

    [TestMethod]
    public void RowMapper_RequiresExactSchemaAndRejectsInactiveGetters()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            new[] { new StateRow { State = "state" } });
        using var transformer = ml.Transforms.OnnxTypedDecisions(
            fixture.CreateOptions([Noul("risk", "Can it be acted on?")]))
            .Fit(input);
        var mapper = transformer.GetRowToRowMapper(input.Schema);
        Assert.AreSame(input.Schema, mapper.InputSchema);
        using (var identityCursor = input.GetRowCursor(input.Schema))
        using (var identityRow = mapper.GetRow(
            identityCursor,
            Array.Empty<DataViewSchema.Column>()))
        {
            Assert.AreSame(mapper.OutputSchema, identityRow.Schema);
        }

        using var cursor = input.GetRowCursor([input.Schema["State"]]);
        Assert.IsTrue(cursor.MoveNext());
        using var row = mapper.GetRow(
            cursor,
            [mapper.OutputSchema["State"]]);
        Assert.IsFalse(row.IsColumnActive(mapper.OutputSchema["DecisionResults"]));
        Assert.ThrowsException<InvalidOperationException>(() =>
            row.GetGetter<ReadOnlyMemory<char>>(mapper.OutputSchema["DecisionResults"]));
    }

    [TestMethod]
    public void PredictionEngine_CustomPrefixMapsMultipleSameTypeQuestionsAndSlotNames()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var questions = new[]
        {
            Choice("priority_a", "First priority?", ["low", "high"]),
            Choice("priority_b", "Second priority?", ["red", "amber", "green"]),
            Score("quality_a", "First quality?", ["weak", "strong"]),
            Score("quality_b", "Second quality?", ["poor", "fair", "good", "excellent"])
        };
        var options = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = fixture.ModelAssetsPath,
            Questions = questions,
            OutputPrefix = "Custom_"
        };
        var input = ml.Data.LoadFromEnumerable(
            new[] { new StateRow { State = "same-type questions" } });
        using var transformer = ml.Transforms.OnnxTypedDecisions(options).Fit(input);
        var mapper = transformer.GetRowToRowMapper(input.Schema);
        AssertSlotNames(
            mapper.OutputSchema["Custom_priority_a_Probabilities"],
            ["low", "high"]);
        AssertSlotNames(
            mapper.OutputSchema["Custom_priority_b_Probabilities"],
            ["red", "amber", "green"]);
        AssertSlotNames(
            mapper.OutputSchema["Custom_quality_a_Probabilities"],
            ["0", "1"]);
        AssertSlotNames(
            mapper.OutputSchema["Custom_quality_b_Probabilities"],
            ["0", "1", "2", "3"]);

        using var engine = ml.Model.CreatePredictionEngine<StateRow, MultiQuestionRow>(
            transformer,
            new PredictionEngineOptions { OwnsTransformer = false });
        var actual = engine.Predict(new StateRow { State = "same-type questions" });
        var expected = transformer.Infer("same-type questions");
        var choiceA = (ChoiceDecisionResult)expected.Results[0];
        var choiceB = (ChoiceDecisionResult)expected.Results[1];
        var scoreA = (ScoreDecisionResult)expected.Results[2];
        var scoreB = (ScoreDecisionResult)expected.Results[3];
        Assert.AreEqual(choiceA.Choice, actual.Custom_priority_a_PredictedLabel);
        Assert.AreEqual(choiceB.Choice, actual.Custom_priority_b_PredictedLabel);
        CollectionAssert.AreEqual(
            choiceA.Distribution.Probabilities.ToArray(),
            actual.Custom_priority_a_Probabilities.DenseValues().ToArray());
        CollectionAssert.AreEqual(
            choiceB.Distribution.Probabilities.ToArray(),
            actual.Custom_priority_b_Probabilities.DenseValues().ToArray());
        Assert.AreEqual(scoreA.Score, actual.Custom_quality_a_Score, 0.000001f);
        Assert.AreEqual(scoreB.Score, actual.Custom_quality_b_Score, 0.000001f);
        Assert.AreEqual(scoreA.Confidence, actual.Custom_quality_a_Confidence, 0.000001f);
        Assert.AreEqual(scoreB.ActionProbability, actual.Custom_quality_b_ActionProbability, 0.000001f);
    }

    [TestMethod]
    public void RowMapper_UsesMinimalDependenciesAndValidatesSchemaAndGetterTypes()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            new[]
            {
                new StateWithPassthroughRow
                {
                    State = "state",
                    Label = true,
                    Features = new VBuffer<float>(2, new[] { 1f, 2f })
                }
            });
        using var transformer = ml.Transforms.OnnxTypedDecisions(
            fixture.CreateOptions([Noul("risk", "Can it be acted on?")]))
            .Fit(input);
        var mapper = transformer.GetRowToRowMapper(input.Schema);
        CollectionAssert.AreEqual(
            new[] { input.Schema["Features"].Index },
            mapper.GetDependencies([mapper.OutputSchema["Features"]])
                .Select(static column => column.Index)
                .ToArray());
        CollectionAssert.AreEqual(
            new[] { input.Schema["State"].Index },
            mapper.GetDependencies([mapper.OutputSchema["DecisionResults"]])
                .Select(static column => column.Index)
                .ToArray());

        var invalidColumnInput = ml.Data.LoadFromEnumerable(
            new[] { new LabelRow { Label = true } });
        Assert.ThrowsException<ArgumentException>(() =>
            mapper.GetDependencies([invalidColumnInput.Schema["Label"]]));

        var otherInput = ml.Data.LoadFromEnumerable(
            new[] { new StateRow { State = "other" } });
        using (var otherCursor = otherInput.GetRowCursor(otherInput.Schema))
        {
            Assert.IsTrue(otherCursor.MoveNext());
            Assert.ThrowsException<ArgumentException>(() =>
                mapper.GetRow(otherCursor, [mapper.OutputSchema["State"]]));
        }

        using var cursor = input.GetRowCursor([input.Schema["Features"]]);
        Assert.IsTrue(cursor.MoveNext());
        using var row = mapper.GetRow(cursor, [mapper.OutputSchema["Features"]]);
        var featureGetter = row.GetGetter<VBuffer<float>>(mapper.OutputSchema["Features"]);
        VBuffer<float> features = default;
        featureGetter(ref features);
        CollectionAssert.AreEqual(new[] { 1f, 2f }, features.DenseValues().ToArray());
        Assert.IsFalse(row.IsColumnActive(mapper.OutputSchema["DecisionResults"]));

        using var inferenceCursor = input.GetRowCursor([input.Schema["Features"]]);
        Assert.IsTrue(inferenceCursor.MoveNext());
        using var inferenceRow = mapper.GetRow(
            inferenceCursor,
            [mapper.OutputSchema["Features"], mapper.OutputSchema["DecisionResults"]]);
        Assert.ThrowsException<InvalidCastException>(() =>
            inferenceRow.GetGetter<float>(mapper.OutputSchema["DecisionResults"]));
    }

    [TestMethod]
    public void RowMapper_CachesInferencePerRowAndDisposesInputWithoutOwningTransformer()
    {
        using var fixture = NativeBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            new[] { new StateRow { State = "state" } });
        using var transformer = ml.Transforms.OnnxTypedDecisions(
            fixture.CreateOptions([Noul("risk", "Can it be acted on?")]))
            .Fit(input);
        var mapper = transformer.GetRowToRowMapper(input.Schema);
        using var cursor = input.GetRowCursor([input.Schema["State"]]);
        Assert.IsTrue(cursor.MoveNext());
        var tracked = new TrackingDataViewRow(cursor);
        var mapped = mapper.GetRow(
            tracked,
            [
                mapper.OutputSchema["DecisionResults"],
                mapper.OutputSchema["Decision_risk_Confidence"],
                mapper.OutputSchema["Decision_risk_ActionProbability"]
            ]);

        var resultsGetter = mapped.GetGetter<ReadOnlyMemory<char>>(
            mapper.OutputSchema["DecisionResults"]);
        var confidenceGetter = mapped.GetGetter<float>(
            mapper.OutputSchema["Decision_risk_Confidence"]);
        var actionGetter = mapped.GetGetter<float>(
            mapper.OutputSchema["Decision_risk_ActionProbability"]);
        ReadOnlyMemory<char> results = default;
        float confidence = default;
        float actionProbability = default;
        resultsGetter(ref results);
        confidenceGetter(ref confidence);
        actionGetter(ref actionProbability);
        Assert.IsFalse(results.IsEmpty);
        Assert.IsTrue(float.IsFinite(confidence));
        Assert.IsTrue(float.IsFinite(actionProbability));
        Assert.AreEqual(1, tracked.StateReads);

        mapped.Dispose();
        mapped.Dispose();
        Assert.AreEqual(1, tracked.DisposeCount);
        Assert.ThrowsException<ObjectDisposedException>(() =>
            mapped.GetGetter<ReadOnlyMemory<char>>(
                mapper.OutputSchema["DecisionResults"]));
        var afterRowDispose = transformer.Infer("still usable");
        Assert.AreEqual(1, afterRowDispose.Results.Count);
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

    public sealed class StateWithPassthroughRow
    {
        public string State { get; set; } = string.Empty;
        public bool Label { get; set; }
        public VBuffer<float> Features { get; set; }
    }

    public sealed class LabelRow
    {
        public bool Label { get; set; }
    }

    private sealed class TrackingDataViewRow : DataViewRow
    {
        private readonly DataViewRow _inner;
        private bool _disposed;

        internal TrackingDataViewRow(DataViewRow inner)
        {
            _inner = inner;
        }

        internal int StateReads { get; private set; }
        internal int DisposeCount { get; private set; }

        public override DataViewSchema Schema => _inner.Schema;
        public override long Position => _inner.Position;
        public override long Batch => _inner.Batch;

        public override ValueGetter<DataViewRowId> GetIdGetter()
            => _inner.GetIdGetter();

        public override bool IsColumnActive(DataViewSchema.Column column)
            => _inner.IsColumnActive(column);

        public override ValueGetter<TValue> GetGetter<TValue>(
            DataViewSchema.Column column)
        {
            var getter = _inner.GetGetter<TValue>(column);
            if (column.Name != "State")
                return getter;
            return (ref TValue value) =>
            {
                StateReads++;
                getter(ref value);
            };
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && !_disposed)
            {
                _disposed = true;
                DisposeCount++;
                _inner.Dispose();
            }

            base.Dispose(disposing);
        }
    }

    public class DecisionRow
    {
        public string DecisionResults { get; set; } = string.Empty;
        public float Decision_quality_Score { get; set; }
        public bool Decision_actionable_PredictedLabel { get; set; }
        public float Decision_actionable_Probability { get; set; }
    }

    public class FullDecisionRow : DecisionRow
    {
        public string Decision_priority_PredictedLabel { get; set; } = string.Empty;
        public VBuffer<float> Decision_priority_Probabilities { get; set; }
        public float Decision_priority_Confidence { get; set; }
        public float Decision_priority_ActionProbability { get; set; }
        public VBuffer<float> Decision_quality_Probabilities { get; set; }
        public float Decision_quality_Confidence { get; set; }
        public float Decision_quality_ActionProbability { get; set; }
        public float Decision_actionable_Confidence { get; set; }
        public float Decision_actionable_ActionProbability { get; set; }
    }

    public sealed class FullStageRow : FullDecisionRow
    {
        public VBuffer<long> DecisionInputIds { get; set; }
        public VBuffer<long> DecisionAttentionMask { get; set; }
        public VBuffer<long> DecisionMarkerPositions { get; set; }
        public VBuffer<bool> DecisionMarkerMask { get; set; }
        public VBuffer<long> DecisionQuestionTypes { get; set; }
        public int DecisionBatchSize { get; set; }
        public int DecisionSequenceLength { get; set; }
        public int DecisionMarkerWidth { get; set; }
        public VBuffer<float> DecisionLogits { get; set; }
        public VBuffer<float> DecisionActionProbabilities { get; set; }
    }

    public sealed class FullComposedRow : FullDecisionRow
    {
        public string AppendedDecisionResults { get; set; } = string.Empty;
        public string AppendedDecision_priority_PredictedLabel { get; set; } = string.Empty;
        public VBuffer<float> AppendedDecision_priority_Probabilities { get; set; }
        public float AppendedDecision_priority_Confidence { get; set; }
        public float AppendedDecision_priority_ActionProbability { get; set; }
        public float AppendedDecision_quality_Score { get; set; }
        public VBuffer<float> AppendedDecision_quality_Probabilities { get; set; }
        public float AppendedDecision_quality_Confidence { get; set; }
        public float AppendedDecision_quality_ActionProbability { get; set; }
        public bool AppendedDecision_actionable_PredictedLabel { get; set; }
        public float AppendedDecision_actionable_Probability { get; set; }
        public float AppendedDecision_actionable_Confidence { get; set; }
        public float AppendedDecision_actionable_ActionProbability { get; set; }
    }

    public sealed class MultiQuestionRow
    {
        public string Custom_priority_a_PredictedLabel { get; set; } = string.Empty;
        public VBuffer<float> Custom_priority_a_Probabilities { get; set; }
        public string Custom_priority_b_PredictedLabel { get; set; } = string.Empty;
        public VBuffer<float> Custom_priority_b_Probabilities { get; set; }
        public float Custom_quality_a_Score { get; set; }
        public VBuffer<float> Custom_quality_a_Probabilities { get; set; }
        public float Custom_quality_a_Confidence { get; set; }
        public float Custom_quality_a_ActionProbability { get; set; }
        public float Custom_quality_b_Score { get; set; }
        public VBuffer<float> Custom_quality_b_Probabilities { get; set; }
        public float Custom_quality_b_Confidence { get; set; }
        public float Custom_quality_b_ActionProbability { get; set; }
    }

    private static void AssertDecisionRowMatches(
        FullDecisionRow row,
        DecisionResponse expected)
    {
        var choice = expected.Results.OfType<ChoiceDecisionResult>().Single();
        var score = expected.Results.OfType<ScoreDecisionResult>().Single();
        var noul = expected.Results.OfType<NoulDecisionResult>().Single();
        Assert.AreEqual(choice.Choice, row.Decision_priority_PredictedLabel);
        CollectionAssert.AreEqual(
            choice.Distribution.Probabilities.ToArray(),
            row.Decision_priority_Probabilities.DenseValues().ToArray());
        Assert.AreEqual(choice.Confidence, row.Decision_priority_Confidence, 0.000001f);
        Assert.AreEqual(choice.ActionProbability, row.Decision_priority_ActionProbability, 0.000001f);
        Assert.AreEqual(score.Score, row.Decision_quality_Score, 0.000001f);
        CollectionAssert.AreEqual(
            score.Distribution.Probabilities.ToArray(),
            row.Decision_quality_Probabilities.DenseValues().ToArray());
        Assert.AreEqual(score.Confidence, row.Decision_quality_Confidence, 0.000001f);
        Assert.AreEqual(score.ActionProbability, row.Decision_quality_ActionProbability, 0.000001f);
        Assert.AreEqual(noul.Value, row.Decision_actionable_PredictedLabel);
        Assert.AreEqual(noul.ProbabilityTrue, row.Decision_actionable_Probability, 0.000001f);
        Assert.AreEqual(noul.Confidence, row.Decision_actionable_Confidence, 0.000001f);
        Assert.AreEqual(noul.ActionProbability, row.Decision_actionable_ActionProbability, 0.000001f);
        using var json = JsonDocument.Parse(row.DecisionResults);
        Assert.AreEqual(
            expected.InputTokenCount,
            json.RootElement.GetProperty("input_tokens").GetInt32());
    }

    private static void AssertSlotNames(
        DataViewSchema.Column column,
        IReadOnlyList<string> expected)
    {
        var annotation = column.Annotations.Schema["SlotNames"];
        var getter = column.Annotations.GetGetter<VBuffer<ReadOnlyMemory<char>>>(annotation);
        VBuffer<ReadOnlyMemory<char>> actual = default;
        getter(ref actual);
        CollectionAssert.AreEqual(
            expected.ToArray(),
            actual.DenseValues().ToArray().Select(static value => value.ToString()).ToArray());
    }

    private static void AssertAppendedDecisionRowMatches(
        FullComposedRow row,
        DecisionResponse expected)
    {
        var choice = expected.Results.OfType<ChoiceDecisionResult>().Single();
        var score = expected.Results.OfType<ScoreDecisionResult>().Single();
        var noul = expected.Results.OfType<NoulDecisionResult>().Single();
        Assert.AreEqual(choice.Choice, row.AppendedDecision_priority_PredictedLabel);
        CollectionAssert.AreEqual(
            choice.Distribution.Probabilities.ToArray(),
            row.AppendedDecision_priority_Probabilities.DenseValues().ToArray());
        Assert.AreEqual(choice.Confidence, row.AppendedDecision_priority_Confidence, 0.000001f);
        Assert.AreEqual(choice.ActionProbability, row.AppendedDecision_priority_ActionProbability, 0.000001f);
        Assert.AreEqual(score.Score, row.AppendedDecision_quality_Score, 0.000001f);
        CollectionAssert.AreEqual(
            score.Distribution.Probabilities.ToArray(),
            row.AppendedDecision_quality_Probabilities.DenseValues().ToArray());
        Assert.AreEqual(score.Confidence, row.AppendedDecision_quality_Confidence, 0.000001f);
        Assert.AreEqual(score.ActionProbability, row.AppendedDecision_quality_ActionProbability, 0.000001f);
        Assert.AreEqual(noul.Value, row.AppendedDecision_actionable_PredictedLabel);
        Assert.AreEqual(noul.ProbabilityTrue, row.AppendedDecision_actionable_Probability, 0.000001f);
        Assert.AreEqual(noul.Confidence, row.AppendedDecision_actionable_Confidence, 0.000001f);
        Assert.AreEqual(noul.ActionProbability, row.AppendedDecision_actionable_ActionProbability, 0.000001f);
        using var json = JsonDocument.Parse(row.AppendedDecisionResults);
        Assert.AreEqual(
            expected.InputTokenCount,
            json.RootElement.GetProperty("input_tokens").GetInt32());
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
        public int StateReadCount { get; private set; }

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
            public override long Batch => Math.Max(0, _index / 2);

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
                        (ref ReadOnlyMemory<char> value) =>
                        {
                            _parent.StateReadCount++;
                            value = _state;
                        }),
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

    internal sealed class NativeBundleFixture : IDisposable
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
