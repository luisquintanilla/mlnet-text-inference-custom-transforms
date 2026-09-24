using System.IO.Compression;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLNet.TextInference.Onnx;
using MLNet.TextInference.TypedDecisions;

namespace MLNet.TextInference.TypedDecisions.Tests;

[TestClass]
public sealed class TypedDecisionPortableModelTests
{
    [TestMethod]
    public void FacadeRoundTripsDirectSingleAndBatch()
    {
        using var fixture = PortableBundleFixture.Create();
        var archive = Path.Combine(fixture.Root, "facade.zip");
        var secondArchive = Path.Combine(fixture.Root, "facade-again.zip");
        var questions = Questions();
        var ml = new MLContext(seed: 1);
        var states = new[]
        {
            "portable round-trip state",
            "a longer portable state with different token count"
        };
        var data = ml.Data.LoadFromEnumerable(
            states.Select(static state => new StateRow { Evidence = state }));
        var options = fixture.CreateFacadeOptions(questions);

        using (var transformer = ml.Transforms.OnnxTypedDecisions(options).Fit(data))
        {
            var expectedBatch = transformer.Infer(states).ToArray();
            transformer.Save(archive);

            Assert.IsTrue(File.Exists(archive));
            using (var zip = ZipFile.OpenRead(archive))
            using (var document = JsonDocument.Parse(
                       zip.GetEntry("typed-decision-portable.json")!.Open()))
            {
                Assert.AreEqual(
                    1,
                    document.RootElement
                        .GetProperty("Facade")
                        .GetProperty("BatchSize")
                        .GetInt32());
            }

            fixture.DeleteSourceAssets();

            var reloaded = OnnxTypedDecisionsTransformer.Load(new MLContext(seed: 2), archive);
            try
            {
                Assert.AreEqual(options.BatchSize, reloaded.Options.BatchSize);
                var actualBatch = reloaded.Infer(states);
                Assert.AreEqual(expectedBatch.Length, actualBatch.Count);
                for (var index = 0; index < expectedBatch.Length; index++)
                    AssertResponsesEqual(expectedBatch[index], actualBatch[index]);
                AssertResponsesEqual(
                    expectedBatch[0],
                    reloaded.Infer("portable round-trip state"));

                var outputSchema = reloaded.GetOutputSchema(data.Schema);
                Assert.IsNotNull(outputSchema["Portable_priority_PredictedLabel"]);
                Assert.IsNotNull(outputSchema["Portable_quality_Score"]);
                Assert.IsNotNull(outputSchema["Portable_actionable_PredictedLabel"]);
                Assert.IsNotNull(outputSchema["PortableResults"]);

                reloaded.Save(secondArchive);
            }
            finally
            {
                reloaded.Dispose();
            }
        }

        var replay = OnnxTypedDecisionsTransformer.Load(new MLContext(seed: 3), secondArchive);
        try
        {
            Assert.AreEqual(2, replay.Infer(states).Count);
        }
        finally
        {
            replay.Dispose();
        }
    }

    [TestMethod]
    public void FacadeRoundTripsLazyDataViewAndPredictionEngine()
    {
        using var fixture = PortableBundleFixture.Create();
        var archive = Path.Combine(fixture.Root, "lazy-facade.zip");
        var states = new[]
        {
            "lazy first state",
            "lazy second state with different token count"
        };
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(
            states.Select(static state => new StateRow { Evidence = state }));

        DecisionResponse[] expected;
        using (var original = ml.Transforms.OnnxTypedDecisions(
                   fixture.CreateFacadeOptions(Questions())).Fit(data))
        {
            expected = states.Select(original.Infer).ToArray();
            original.Save(archive);
        }

        fixture.DeleteSourceAssets();
        using var loaded = OnnxTypedDecisionsTransformer.Load(
            new MLContext(seed: 2),
            archive);
        var rows = ml.Data.CreateEnumerable<FacadeOutputRow>(
            loaded.Transform(data),
            reuseRowObject: false).ToArray();
        Assert.AreEqual(states.Length, rows.Length);
        for (var index = 0; index < rows.Length; index++)
            AssertFacadeRowMatches(rows[index], expected[index]);

        using var engine = ml.Model.CreatePredictionEngine<StateRow, FacadeOutputRow>(
            loaded,
            new PredictionEngineOptions { OwnsTransformer = false });
        for (var repeat = 0; repeat < 2; repeat++)
        {
            for (var index = 0; index < states.Length; index++)
                AssertFacadeRowMatches(
                    engine.Predict(new StateRow { Evidence = states[index] }),
                    expected[index]);
        }
    }

    [TestMethod]
    public void SupportedPipelinesRoundTripAndRejectUnsupportedComponents()
    {
        using var fixture = PortableBundleFixture.Create();
        var archive = Path.Combine(fixture.Root, "pipeline.zip");
        var replayArchive = Path.Combine(fixture.Root, "pipeline-again.zip");
        var ml = new MLContext(seed: 1);
        var questions = Questions();
        var input = ml.Data.LoadFromEnumerable(
            [new StateRow { Evidence = "pipeline state" }]);

        using var preparation = ml.Transforms.PrepareDecisionInputs(
            fixture.CreatePreparationOptions(questions)).Fit(input);
        using var scoring = ml.Transforms.ScoreOnnxDecisionModel(
            fixture.CreateScoringOptions()).Fit(preparation.Transform(input));
        using var decoding = ml.Transforms.DecodeDecisions(
            fixture.CreateDecodingOptions(questions)).Fit(
                scoring.Transform(preparation.Transform(input)));
        var pipeline = new TransformerChain<ITransformer>(
            [preparation, scoring, decoding]);

        TypedDecisionPortableModel.SavePipeline(pipeline, archive);
        using (var zip = ZipFile.OpenRead(archive))
        using (var document = JsonDocument.Parse(
                   zip.GetEntry("typed-decision-portable.json")!.Open()))
        {
            var descriptors = document.RootElement
                .GetProperty("Pipeline")
                .GetProperty("Transformers");
            Assert.AreEqual(
                2,
                descriptors[1].GetProperty("Scoring").GetProperty("BatchSize").GetInt32());
        }
        var expected = ml.Data.CreateEnumerable<PipelineRow>(
            pipeline.Transform(input),
            reuseRowObject: false).Single();

        fixture.DeleteSourceAssets();

        var loaded = TypedDecisionPortableModel.LoadPipeline(
            new MLContext(seed: 2),
            archive);
        try
        {
            var actual = new MLContext(seed: 3).Data.CreateEnumerable<PipelineRow>(
                loaded.Transform(
                    new MLContext(seed: 3).Data.LoadFromEnumerable(
                        [new StateRow { Evidence = "pipeline state" }])),
                reuseRowObject: false).Single();

            using (var expectedJson = JsonDocument.Parse(expected.PortableResults))
            using (var actualJson = JsonDocument.Parse(actual.PortableResults))
            {
                Assert.AreEqual(
                    expectedJson.RootElement.GetProperty("input_tokens").GetInt32(),
                    actualJson.RootElement.GetProperty("input_tokens").GetInt32());
            }
            CollectionAssert.AreEqual(
                expected.PortableInputIds.DenseValues().ToArray(),
                actual.PortableInputIds.DenseValues().ToArray());
            CollectionAssert.AreEqual(
                expected.PortableLogits.DenseValues().ToArray(),
                actual.PortableLogits.DenseValues().ToArray());
            Assert.AreEqual(
                expected.Portable_priority_PredictedLabel,
                actual.Portable_priority_PredictedLabel);
            Assert.AreEqual(expected.Portable_quality_Score, actual.Portable_quality_Score);
            Assert.AreEqual(expected.Portable_actionable_PredictedLabel,
                actual.Portable_actionable_PredictedLabel);
            TypedDecisionPortableModel.SavePipeline(loaded, replayArchive);
        }
        finally
        {
            loaded.Dispose();
        }

        using var replay = TypedDecisionPortableModel.LoadPipeline(
            new MLContext(seed: 4),
            replayArchive);
        Assert.IsNotNull(replay);

        var unsupported = ml.Transforms.CopyColumns(
            "CopiedEvidence",
            nameof(StateRow.Evidence)).Fit(input);
        Assert.ThrowsException<NotSupportedException>(() =>
            TypedDecisionPortableModel.Save(
                unsupported,
                Path.Combine(fixture.Root, "unsupported.zip")));

        var nested = new TransformerChain<ITransformer>([preparation]);
        var nestedPipeline = new TransformerChain<ITransformer>([nested]);
        Assert.ThrowsException<NotSupportedException>(() =>
            TypedDecisionPortableModel.SavePipeline(
                nestedPipeline,
                Path.Combine(fixture.Root, "nested.zip")));
    }

    [TestMethod]
    public void SavePipelineRejectsTrainingAndScoringOnlyAndUnsupportedScopeBits()
    {
        using var fixture = PortableBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            [new StateRow { Evidence = "scope state" }]);

        using var facade = ml.Transforms.OnnxTypedDecisions(
            fixture.CreateFacadeOptions(Questions())).Fit(input);
        var trainingAndScoringOnly = new TransformerChain<ITransformer>(
            [facade],
            [TransformerScope.Training | TransformerScope.Scoring]);
        Assert.ThrowsException<NotSupportedException>(() =>
            TypedDecisionPortableModel.SavePipeline(
                trainingAndScoringOnly,
                Path.Combine(fixture.Root, "training-and-scoring-only.zip")));

        var unsupportedBits = new TransformerChain<ITransformer>(
            [facade],
            [TransformerScope.Everything | (TransformerScope)8]);
        Assert.ThrowsException<NotSupportedException>(() =>
            TypedDecisionPortableModel.SavePipeline(
                unsupportedBits,
                Path.Combine(fixture.Root, "unsupported-scope-bits.zip")));
    }

    [TestMethod]
    public void NaturallyInferredAppendedFacadesRoundTripWithDynamicSecondWidth()
    {
        using var fixture = PortableBundleFixture.CreateExternal();
        var archive = Path.Combine(fixture.Root, "composed-facades.zip");
        var replayArchive = Path.Combine(fixture.Root, "composed-facades-again.zip");
        var ml = new MLContext(seed: 1);
        var states = new[]
        {
            "composed portable state",
            "composed portable state with more context"
        };
        var data = ml.Data.LoadFromEnumerable(
            states.Select(static state => new StateRow { Evidence = state }));
        var firstQuestions = Questions();
        var secondQuestions = new[]
        {
            DecisionQuestion.Choice(
                "binary",
                "Is it binary?",
                new Dictionary<string, string?>(StringComparer.Ordinal)
                {
                    ["no"] = "no",
                    ["yes"] = "yes"
                })
        };
        var first = fixture.CreateFacadeOptions(firstQuestions);
        var second = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = fixture.ModelAssetsPath,
            Questions = secondQuestions,
            StateColumnName = "Evidence",
            OutputPrefix = "Appended_",
            ResultsColumnName = "AppendedDecisionResults",
            BatchSize = 2
        };

        var composed = ml.Transforms.OnnxTypedDecisions(first)
            .AppendOnnxTypedDecisions(ml, second)
            .Fit(data);
        try
        {
            Assert.IsInstanceOfType(
                composed,
                typeof(TransformerChain<OnnxTypedDecisionsTransformer>));
            var inferredChain = (TransformerChain<OnnxTypedDecisionsTransformer>)composed;
            var expectedRows = ml.Data.CreateEnumerable<ComposedOutputRow>(
                composed.Transform(data),
                reuseRowObject: false).ToArray();
            Assert.AreEqual(states.Length, expectedRows.Length);
            Assert.AreEqual(3, expectedRows[0].Portable_priority_Probabilities.Length);
            Assert.AreEqual(2, expectedRows[0].Appended_binary_Probabilities.Length);

            var composedSchema = composed.GetOutputSchema(data.Schema);
            Assert.IsNotNull(composedSchema["PortableResults"]);
            Assert.IsNotNull(composedSchema["AppendedDecisionResults"]);
            AssertSlotNames(
                composedSchema["Appended_binary_Probabilities"],
                ["no", "yes"]);

            TypedDecisionPortableModel.SavePipeline(inferredChain, archive);
            fixture.DeleteSourceAssets();

            using var loaded = TypedDecisionPortableModel.LoadPipeline(
                new MLContext(seed: 2),
                archive);
            var actualRows = new MLContext(seed: 3).Data.CreateEnumerable<ComposedOutputRow>(
                loaded.Transform(
                    new MLContext(seed: 3).Data.LoadFromEnumerable(
                        states.Select(static state => new StateRow { Evidence = state }))),
                reuseRowObject: false).ToArray();
            AssertComposedRowsEqual(expectedRows, actualRows);

            using var engine = new MLContext(seed: 4).Model.CreatePredictionEngine<
                StateRow,
                ComposedOutputRow>(
                loaded,
                new PredictionEngineOptions { OwnsTransformer = false });
            for (var index = 0; index < states.Length; index++)
                AssertComposedRowMatches(
                    engine.Predict(new StateRow { Evidence = states[index] }),
                    expectedRows[index]);

            TypedDecisionPortableModel.SavePipeline(loaded, replayArchive);
        }
        finally
        {
            (composed as IDisposable)?.Dispose();
        }

        using var replay = TypedDecisionPortableModel.LoadPipeline(
            new MLContext(seed: 5),
            replayArchive);
        var replayRows = new MLContext(seed: 5).Data.CreateEnumerable<ComposedOutputRow>(
            replay.Transform(
                new MLContext(seed: 5).Data.LoadFromEnumerable(
                    [new StateRow { Evidence = states[0] }])),
            reuseRowObject: false).ToArray();
        Assert.AreEqual(3, replayRows[0].Portable_priority_Probabilities.Length);
        Assert.AreEqual(2, replayRows[0].Appended_binary_Probabilities.Length);
    }

    [TestMethod]
    public void SavePipelineRejectsMismatchedAssetPayloads()
    {
        using var first = PortableBundleFixture.CreateExternal();
        using var second = PortableBundleFixture.CreateExternal();
        second.SetTemperature(2.0);
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            [new StateRow { Evidence = "mismatched assets" }]);
        var questions = Questions();

        using var preparation = ml.Transforms.PrepareDecisionInputs(
            first.CreatePreparationOptions(questions)).Fit(input);
        using var scoring = ml.Transforms.ScoreOnnxDecisionModel(
            second.CreateScoringOptions()).Fit(preparation.Transform(input));
        var pipeline = new TransformerChain<ITransformer>(
            [preparation, scoring]);

        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.SavePipeline(
                pipeline,
                Path.Combine(first.Root, "mismatch.zip")));
    }

    [TestMethod]
    public void SeparatelyLoadedSelectiveStageArchivesCannotBeRecombined()
    {
        using var full = PortableBundleFixture.CreateExternal();
        using var profileOnly = PortableBundleFixture.CreateProfileOnly();
        var fullArchive = Path.Combine(full.Root, "scoring.zip");
        var decoderArchive = Path.Combine(profileOnly.Root, "decoder.zip");
        var ml = new MLContext(seed: 1);
        var questions = Questions();
        var input = ml.Data.LoadFromEnumerable(
            [new StateRow { Evidence = "selective archive state" }]);

        using (var preparation = ml.Transforms.PrepareDecisionInputs(
                   full.CreatePreparationOptions(questions)).Fit(input))
        using (var scoring = ml.Transforms.ScoreOnnxDecisionModel(
                   full.CreateScoringOptions()).Fit(preparation.Transform(input)))
        {
            scoring.Save(fullArchive);
            using var decoder = ml.Transforms.DecodeDecisions(
                    full.CreateDecodingOptions(questions, profileOnly.ModelAssetsPath))
                .Fit(scoring.Transform(preparation.Transform(input)));
            decoder.Save(decoderArchive);
        }

        full.DeleteSourceAssets();
        profileOnly.DeleteSourceAssets();
        using var loadedScoring = OnnxDecisionModelScorerTransformer.Load(
            new MLContext(seed: 2),
            fullArchive);
        using var loadedDecoder = DecisionDecodingTransformer.Load(
            new MLContext(seed: 3),
            decoderArchive);
        var selectivePipeline = new TransformerChain<ITransformer>(
            [loadedScoring, loadedDecoder]);

        Assert.ThrowsException<FileNotFoundException>(() =>
            TypedDecisionPortableModel.SavePipeline(
                selectivePipeline,
                Path.Combine(full.Root, "recombined.zip")));
    }

    [TestMethod]
    public void Decoder_ProfileOnlySaveLoadDoesNotRequireModelOrTokenizer()
    {
        using var fixture = PortableBundleFixture.CreateProfileOnly();
        var archive = Path.Combine(fixture.Root, "decoder.zip");
        var ml = new MLContext(seed: 1);
        var questions = Questions();
        var input = ml.Data.LoadFromEnumerable(
            [new DefaultScoredRow()]);

        using var decoder = ml.Transforms.DecodeDecisions(
            new DecisionDecodingOptions
            {
                ModelAssetsPath = fixture.ModelAssetsPath,
                Questions = questions
            }).Fit(input);
        var expected = ml.Data.CreateEnumerable<DecoderOutputRow>(
            decoder.Transform(input),
            reuseRowObject: false).Single();
        decoder.Save(archive);
        fixture.DeleteSourceAssets();

        using var loaded = DecisionDecodingTransformer.Load(
            new MLContext(seed: 2),
            archive);
        var actual = ml.Data.CreateEnumerable<DecoderOutputRow>(
            loaded.Transform(input),
            reuseRowObject: false).Single();
        var output = loaded.GetOutputSchema(input.Schema);
        Assert.IsNotNull(output["Decision_priority_PredictedLabel"]);
        Assert.IsNotNull(output["Decision_quality_Score"]);
        Assert.IsNotNull(output["Decision_actionable_PredictedLabel"]);
        Assert.AreEqual(
            expected.Decision_priority_PredictedLabel,
            actual.Decision_priority_PredictedLabel);
        CollectionAssert.AreEqual(
            expected.Decision_priority_Probabilities.DenseValues().ToArray(),
            actual.Decision_priority_Probabilities.DenseValues().ToArray());
        Assert.AreEqual(expected.Decision_quality_Score, actual.Decision_quality_Score);
        Assert.AreEqual(
            expected.Decision_actionable_PredictedLabel,
            actual.Decision_actionable_PredictedLabel);
    }

    [TestMethod]
    public void StagesRoundTripWithNonDefaultColumnsAndSlotNames()
    {
        using var fixture = PortableBundleFixture.Create();
        var ml = new MLContext(seed: 1);
        var questions = Questions();
        var data = ml.Data.LoadFromEnumerable(
            new[]
            {
                new StateRow { Evidence = "stage one" },
                new StateRow { Evidence = "stage two with more context" }
            });
        var preparationArchive = Path.Combine(fixture.Root, "preparation.zip");
        var scoringArchive = Path.Combine(fixture.Root, "scoring.zip");
        var decodingArchive = Path.Combine(fixture.Root, "decoding.zip");

        using var preparation = ml.Transforms.PrepareDecisionInputs(
                fixture.CreatePreparationOptions(questions))
            .Fit(data);
        var prepared = preparation.Transform(data);
        using var scoring = ml.Transforms.ScoreOnnxDecisionModel(
                fixture.CreateScoringOptions())
            .Fit(prepared);
        var scored = scoring.Transform(prepared);
        using var decoding = ml.Transforms.DecodeDecisions(
                fixture.CreateDecodingOptions(questions))
            .Fit(scored);
        var expectedRows = ml.Data.CreateEnumerable<StageOutputRow>(
            decoding.Transform(scored),
            reuseRowObject: false).ToArray();

        preparation.Save(preparationArchive);
        scoring.Save(scoringArchive);
        decoding.Save(decodingArchive);

        preparation.Dispose();
        scoring.Dispose();
        decoding.Dispose();
        Assert.ThrowsException<ObjectDisposedException>(() => preparation.Save(
            Path.Combine(fixture.Root, "preparation-after-dispose.zip")));
        Assert.ThrowsException<ObjectDisposedException>(() => scoring.Save(
            Path.Combine(fixture.Root, "scoring-after-dispose.zip")));
        Assert.ThrowsException<ObjectDisposedException>(() => decoding.Save(
            Path.Combine(fixture.Root, "decoding-after-dispose.zip")));

        using var loadedPreparation = DecisionInputPreparationTransformer.Load(
            new MLContext(seed: 2),
            preparationArchive);
        using var loadedScoring = OnnxDecisionModelScorerTransformer.Load(
            new MLContext(seed: 3),
            scoringArchive);
        using var loadedDecoding = DecisionDecodingTransformer.Load(
            new MLContext(seed: 4),
            decodingArchive);
        Assert.AreEqual(2, loadedScoring.Options.BatchSize);

        var preparedSchema = loadedPreparation.GetOutputSchema(data.Schema);
        Assert.IsNotNull(preparedSchema["PortableInputIds"]);
        Assert.IsNotNull(preparedSchema["PortableSequenceLength"]);

        var scoredSchema = loadedScoring.GetOutputSchema(preparedSchema);
        Assert.IsNotNull(scoredSchema["PortableLogits"]);
        Assert.IsNotNull(scoredSchema["PortableActionProbabilities"]);

        var decodedSchema = loadedDecoding.GetOutputSchema(scoredSchema);
        AssertSlotNames(decodedSchema["Portable_priority_Probabilities"], ["zebra", "!", "alpha"]);
        AssertSlotNames(decodedSchema["Portable_quality_Probabilities"], ["0", "1", "2"]);

        fixture.DeleteSourceAssets();
        var loadedOutput = loadedDecoding.Transform(
            loadedScoring.Transform(
                loadedPreparation.Transform(data)));
        var rows = ml.Data.CreateEnumerable<StageOutputRow>(
            loadedOutput,
            reuseRowObject: false).ToArray();

        Assert.AreEqual(2, rows.Length);
        Assert.IsTrue(rows.All(row => row.PortableInputIds.Length > 0));
        Assert.IsTrue(rows.All(row => row.PortableLogits.Length == 9));
        Assert.IsTrue(rows.All(row => !string.IsNullOrWhiteSpace(row.PortableResults)));
        Assert.IsTrue(rows.All(row => row.Portable_priority_Probabilities.Length == 3));
        Assert.IsTrue(rows.All(row => row.Portable_quality_Probabilities.Length == 3));
        Assert.AreEqual(expectedRows.Length, rows.Length);
        for (var index = 0; index < rows.Length; index++)
        {
            CollectionAssert.AreEqual(
                expectedRows[index].PortableInputIds.DenseValues().ToArray(),
                rows[index].PortableInputIds.DenseValues().ToArray());
            CollectionAssert.AreEqual(
                expectedRows[index].PortableLogits.DenseValues().ToArray(),
                rows[index].PortableLogits.DenseValues().ToArray());
            Assert.AreEqual(
                expectedRows[index].PortableResults,
                rows[index].PortableResults);
            CollectionAssert.AreEqual(
                expectedRows[index].Portable_priority_Probabilities.DenseValues().ToArray(),
                rows[index].Portable_priority_Probabilities.DenseValues().ToArray());
            CollectionAssert.AreEqual(
                expectedRows[index].Portable_quality_Probabilities.DenseValues().ToArray(),
                rows[index].Portable_quality_Probabilities.DenseValues().ToArray());
        }
    }

    [TestMethod]
    public void InvalidPortableArchivesFailExplicitlyAndCleanUp()
    {
        using var fixture = PortableBundleFixture.Create();
        var archive = Path.Combine(fixture.Root, "facade.zip");
        var corrupted = Path.Combine(fixture.Root, "corrupted.zip");
        var unsupportedVersion = Path.Combine(fixture.Root, "unsupported-version.zip");
        var missingAsset = Path.Combine(fixture.Root, "missing-asset.zip");
        var tamperedModel = Path.Combine(fixture.Root, "tampered-model.zip");
        var missingHash = Path.Combine(fixture.Root, "missing-hash.zip");
        var unsupportedPolicy = Path.Combine(fixture.Root, "unsupported-policy.zip");
        var wrongKind = Path.Combine(fixture.Root, "wrong-kind.zip");
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            [new StateRow { Evidence = "state" }]);

        using (var transformer = ml.Transforms.OnnxTypedDecisions(
                   fixture.CreateFacadeOptions(Questions())).Fit(input))
            transformer.Save(archive);

        CopyArchive(
            archive,
            corrupted,
            "typed-decision-bundle.json",
            static _ => "corrupt");
        CopyArchive(
            archive,
            unsupportedVersion,
            "typed-decision-portable.json",
            static json => json.Replace(
                "\"FormatVersion\":1",
                "\"FormatVersion\":99",
                StringComparison.Ordinal));
        CopyArchive(archive, missingAsset, "laya_config.json", null);
        CopyArchive(
            archive,
            tamperedModel,
            "laya.onnx",
            static json => json + "tampered");
        CopyArchive(
            archive,
            missingHash,
            "typed-decision-portable.json",
            static json =>
            {
                var root = JsonNode.Parse(json)!.AsObject();
                root["AssetHashes"]!.AsObject().Remove("laya_config.json");
                return root.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
            });
        CopyArchive(
            archive,
            unsupportedPolicy,
            "typed-decision-portable.json",
            static json => json.Replace(
                "\"ExecutionPolicy\":\"UseLoadContext\"",
                "\"ExecutionPolicy\":\"Unsupported\"",
                StringComparison.Ordinal));
        CopyArchive(
            archive,
            wrongKind,
            "typed-decision-portable.json",
            static json => json.Replace(
                "\"Kind\":\"facade\"",
                "\"Kind\":\"Unknown\"",
                StringComparison.Ordinal));

        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 2), corrupted));
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 3), unsupportedVersion));
        Assert.ThrowsException<FileNotFoundException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 4), missingAsset));
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 5), tamperedModel));
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 55), missingHash));
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 6), unsupportedPolicy));
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 7), wrongKind));
        Assert.ThrowsException<InvalidDataException>(() =>
            DecisionDecodingTransformer.Load(new MLContext(seed: 8), archive));

        var destination = Path.Combine(fixture.Root, "existing.zip");
        File.WriteAllText(destination, "sentinel");
        var unsupported = ml.Transforms.CopyColumns(
            "CopiedEvidence",
            nameof(StateRow.Evidence)).Fit(input);
        Assert.ThrowsException<NotSupportedException>(() =>
            TypedDecisionPortableModel.Save(unsupported, destination));
        Assert.AreEqual("sentinel", File.ReadAllText(destination));

        var validDestination = Path.Combine(fixture.Root, "valid-destination");
        Directory.CreateDirectory(validDestination);
        File.WriteAllText(Path.Combine(validDestination, "sentinel.txt"), "keep");
        using var validTransformer = ml.Transforms.OnnxTypedDecisions(
            fixture.CreateFacadeOptions(Questions())).Fit(input);
        Exception? destinationFailure = null;
        try
        {
            validTransformer.Save(validDestination);
        }
        catch (Exception exception)
        {
            destinationFailure = exception;
        }
        Assert.IsNotNull(destinationFailure);
        Assert.IsTrue(
            destinationFailure is UnauthorizedAccessException or IOException,
            $"Unexpected destination-save exception: {destinationFailure.GetType().FullName}");
        Assert.AreEqual(
            "keep",
            File.ReadAllText(Path.Combine(validDestination, "sentinel.txt")));
    }

    [TestMethod]
    public void NullManifestStructuresAndPartialPipelineFailureCleanUp()
    {
        using var fixture = PortableBundleFixture.CreateExternal();
        var facadeArchive = Path.Combine(fixture.Root, "facade.zip");
        var nullHashesArchive = Path.Combine(fixture.Root, "null-hashes.zip");
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            [new StateRow { Evidence = "null manifest state" }]);

        using (var facade = ml.Transforms.OnnxTypedDecisions(
                   fixture.CreateFacadeOptions(Questions())).Fit(input))
            facade.Save(facadeArchive);

        CopyArchive(
            facadeArchive,
            nullHashesArchive,
            "typed-decision-portable.json",
            static json =>
            {
                var root = JsonNode.Parse(json)!.AsObject();
                root["AssetHashes"] = null;
                return root.ToJsonString();
            });
        var rootCount = CountOwnedExtractionRoots();
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 2), nullHashesArchive));
        Assert.AreEqual(rootCount, CountOwnedExtractionRoots());

        var nullQuestionArchive = Path.Combine(fixture.Root, "null-question.zip");
        CopyArchive(
            facadeArchive,
            nullQuestionArchive,
            "typed-decision-portable.json",
            static json =>
            {
                var root = JsonNode.Parse(json)!.AsObject();
                root["Facade"]!.AsObject()["Questions"]!.AsArray()[0] = null;
                return root.ToJsonString();
            });
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 22), nullQuestionArchive));

        var secondQuestions = new[]
        {
            DecisionQuestion.Choice(
                "binary",
                "Is it binary?",
                new Dictionary<string, string?>(StringComparer.Ordinal)
                {
                    ["no"] = "no",
                    ["yes"] = "yes"
                })
        };
        var composed = ml.Transforms.OnnxTypedDecisions(
                fixture.CreateFacadeOptions(Questions()))
            .AppendOnnxTypedDecisions(
                ml,
                new OnnxTypedDecisionsOptions
                {
                    ModelAssetsPath = fixture.ModelAssetsPath,
                    Questions = secondQuestions,
                    StateColumnName = "Evidence",
                    OutputPrefix = "Appended_",
                    ResultsColumnName = "AppendedResults",
                    BatchSize = 2
                })
            .Fit(input);
        var inferred = (TransformerChain<OnnxTypedDecisionsTransformer>)composed;
        var pipelineArchive = Path.Combine(fixture.Root, "composed.zip");
        TypedDecisionPortableModel.SavePipeline(inferred, pipelineArchive);
        (composed as IDisposable)?.Dispose();

        CopyArchive(
            pipelineArchive,
            Path.Combine(fixture.Root, "null-transformers.zip"),
            "typed-decision-portable.json",
            static json =>
            {
                var root = JsonNode.Parse(json)!.AsObject();
                root["Pipeline"]!.AsObject()["Transformers"] = null;
                return root.ToJsonString();
            });
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.LoadPipeline(
                new MLContext(seed: 3),
                Path.Combine(fixture.Root, "null-transformers.zip")));

        CopyArchive(
            pipelineArchive,
            Path.Combine(fixture.Root, "null-descriptor.zip"),
            "typed-decision-portable.json",
            static json =>
            {
                var root = JsonNode.Parse(json)!.AsObject();
                root["Pipeline"]!.AsObject()["Transformers"]!.AsArray()[1] = null;
                return root.ToJsonString();
            });
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.LoadPipeline(
                new MLContext(seed: 4),
                Path.Combine(fixture.Root, "null-descriptor.zip")));

        CopyArchive(
            pipelineArchive,
            Path.Combine(fixture.Root, "partial-failure.zip"),
            "typed-decision-portable.json",
            static json =>
            {
                var root = JsonNode.Parse(json)!.AsObject();
                root["Pipeline"]!.AsObject()["Transformers"]!.AsArray()[1]!
                    .AsObject()["Facade"]!.AsObject()["Questions"]!.AsArray()[0]!
                    .AsObject()["Type"] = 99;
                return root.ToJsonString();
            });
        rootCount = CountOwnedExtractionRoots();
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.LoadPipeline(
                new MLContext(seed: 5),
                Path.Combine(fixture.Root, "partial-failure.zip")));
        Assert.AreEqual(rootCount, CountOwnedExtractionRoots());
    }

    [TestMethod]
    public void ExternalDataFacade_SaveLoadRetainsSidecarAndInference()
    {
        using var fixture = PortableBundleFixture.CreateExternal();
        var archive = Path.Combine(fixture.Root, "external-facade.zip");
        var missingSidecar = Path.Combine(fixture.Root, "missing-sidecar.zip");
        var missingSidecarHash = Path.Combine(fixture.Root, "missing-sidecar-hash.zip");
        var missingGraphSidecarHash = Path.Combine(
            fixture.Root,
            "missing-graph-sidecar-hash.zip");
        var manifestWithoutGraphSidecar = Path.Combine(
            fixture.Root,
            "manifest-without-graph-sidecar.zip");
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            [new StateRow { Evidence = "external-data state" }]);
        var states = new[]
        {
            "external-data state",
            "a different external-data state with more tokens"
        };

        IReadOnlyList<DecisionResponse> expected;
        using (var transformer = ml.Transforms.OnnxTypedDecisions(
                   fixture.CreateFacadeOptions(Questions())).Fit(input))
        {
            expected = transformer.Infer(states);
            var firstDistribution = ((ChoiceDecisionResult)expected[0].Results[0])
                .Distribution.Probabilities;
            var secondDistribution = ((ChoiceDecisionResult)expected[1].Results[0])
                .Distribution.Probabilities;
            Assert.IsTrue(firstDistribution.Any(
                probability => probability > 0.0001f &&
                               probability < 0.9999f));
            CollectionAssert.AreNotEqual(
                firstDistribution.ToArray(),
                secondDistribution.ToArray());
            transformer.Save(archive);
        }

        using (var zip = ZipFile.OpenRead(archive))
        {
            Assert.IsNotNull(zip.GetEntry("weights.bin"));
            Assert.IsNotNull(zip.GetEntry("laya.onnx"));
        }

        CopyArchive(archive, missingSidecar, "weights.bin", null);
        Assert.ThrowsException<FileNotFoundException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 20), missingSidecar));
        CopyArchive(
            archive,
            missingSidecarHash,
            "typed-decision-portable.json",
            static json =>
            {
                var root = JsonNode.Parse(json)!.AsObject();
                root["AssetHashes"]!.AsObject().Remove("weights.bin");
                return root.ToJsonString();
            });
        Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.Load(new MLContext(seed: 21), missingSidecarHash));

        string updatedBundleManifest;
        CopyArchive(
            archive,
            manifestWithoutGraphSidecar,
            TypedDecisionBundle.ManifestFileName,
            static json =>
            {
                var root = JsonNode.Parse(json)!.AsObject();
                root["ExternalDataFiles"] = new JsonArray();
                return root.ToJsonString();
            });
        using (var bundleArchive = ZipFile.OpenRead(manifestWithoutGraphSidecar))
        using (var bundleManifestReader = new StreamReader(
                   bundleArchive.GetEntry(TypedDecisionBundle.ManifestFileName)!.Open()))
            updatedBundleManifest = bundleManifestReader.ReadToEnd();

        CopyArchive(
            manifestWithoutGraphSidecar,
            missingGraphSidecarHash,
            "typed-decision-portable.json",
            json =>
            {
                var root = JsonNode.Parse(json)!.AsObject();
                var hashes = root["AssetHashes"]!.AsObject();
                hashes.Remove("weights.bin");
                hashes["typed-decision-bundle.json"] =
                    Convert.ToHexString(
                            SHA256.HashData(Encoding.UTF8.GetBytes(updatedBundleManifest)))
                        .ToLowerInvariant();
                return root.ToJsonString();
            });
        var missingGraphHashException = Assert.ThrowsException<InvalidDataException>(() =>
            TypedDecisionPortableModel.Load(
                new MLContext(seed: 211),
                missingGraphSidecarHash));
        StringAssert.Contains(missingGraphHashException.Message, "weights.bin");

        fixture.DeleteSourceAssets();
        using var loaded = OnnxTypedDecisionsTransformer.Load(
            new MLContext(seed: 2),
            archive);
        var actual = loaded.Infer(states);
        for (var index = 0; index < expected.Count; index++)
            AssertResponsesEqual(expected[index], actual[index]);
    }

    [TestMethod]
    public void ExternalDataFixture_NonDefaultTemperatureChangesDistribution()
    {
        using var defaultFixture = PortableBundleFixture.CreateExternal();
        using var hotterFixture = PortableBundleFixture.CreateExternal();
        hotterFixture.SetTemperature(2.0);
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            [new StateRow { Evidence = "temperature-sensitive state" }]);
        DecisionResponse defaultResponse;
        DecisionResponse hotterResponse;

        using (var transformer = ml.Transforms.OnnxTypedDecisions(
                   defaultFixture.CreateFacadeOptions(Questions())).Fit(input))
            defaultResponse = transformer.Infer("temperature-sensitive state");
        using (var transformer = ml.Transforms.OnnxTypedDecisions(
                   hotterFixture.CreateFacadeOptions(Questions())).Fit(input))
            hotterResponse = transformer.Infer("temperature-sensitive state");

        var defaultProbabilities = ((ChoiceDecisionResult)defaultResponse.Results[0])
            .Distribution.Probabilities;
        var hotterProbabilities = ((ChoiceDecisionResult)hotterResponse.Results[0])
            .Distribution.Probabilities;
        Assert.IsTrue(defaultProbabilities.All(float.IsFinite));
        Assert.IsTrue(hotterProbabilities.All(float.IsFinite));
        Assert.AreEqual(1f, defaultProbabilities.Sum(), 0.000001f);
        Assert.AreEqual(1f, hotterProbabilities.Sum(), 0.000001f);
        Assert.IsTrue(
            defaultProbabilities.Max() - defaultProbabilities.Min() >
            hotterProbabilities.Max() - hotterProbabilities.Min());
    }

    [TestMethod]
    public void ExternalDataFixture_SupportsDynamicTwoOptionMarkerWidth()
    {
        using var fixture = PortableBundleFixture.CreateExternal();
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            [new StateRow { Evidence = "two option state" }]);
        var questions = new[]
        {
            DecisionQuestion.Choice(
                "binary",
                "Is it binary?",
                new Dictionary<string, string?>(StringComparer.Ordinal)
                {
                    ["no"] = "no",
                    ["yes"] = "yes"
                })
        };

        using var transformer = ml.Transforms.OnnxTypedDecisions(
            fixture.CreateFacadeOptions(questions)).Fit(input);
        var response = transformer.Infer("two option state");
        var choice = (ChoiceDecisionResult)response.Results.Single();
        CollectionAssert.Contains(new[] { "no", "yes" }, choice.Choice);
        Assert.AreEqual(2, choice.Distribution.Probabilities.Count);
        Assert.AreNotEqual(
            choice.Distribution.Probabilities[0],
            choice.Distribution.Probabilities[1]);
        Assert.AreEqual(
            1f,
            choice.Distribution.Probabilities.Sum(),
            0.000001f);
    }

    [TestMethod]
    public void LoadedInstancesAreIndependentAndCanBeSavedAgain()
    {
        using var fixture = PortableBundleFixture.Create();
        var archive = Path.Combine(fixture.Root, "independent.zip");
        var secondArchive = Path.Combine(fixture.Root, "independent-again.zip");
        var ml = new MLContext(seed: 1);
        var input = ml.Data.LoadFromEnumerable(
            [new StateRow { Evidence = "independent load state" }]);

        DecisionResponse expected;
        using (var original = ml.Transforms.OnnxTypedDecisions(
                   fixture.CreateFacadeOptions(Questions())).Fit(input))
        {
            expected = original.Infer("independent load state");
            original.Save(archive);
        }

        fixture.DeleteSourceAssets();
        using var first = OnnxTypedDecisionsTransformer.Load(
            new MLContext(seed: 2),
            archive);
        using var second = OnnxTypedDecisionsTransformer.Load(
            new MLContext(seed: 3),
            archive);
        var firstRoot = first.AssetsRootPath;
        var secondRoot = second.AssetsRootPath;
        Assert.AreNotEqual(firstRoot, secondRoot);
        Assert.IsTrue(Directory.Exists(firstRoot));
        Assert.IsTrue(Directory.Exists(secondRoot));
        first.Dispose();
        Assert.IsFalse(Directory.Exists(firstRoot));
        Assert.IsTrue(Directory.Exists(secondRoot));

        var surviving = second.Infer("independent load state");
        AssertResponsesEqual(expected, surviving);
        second.Save(secondArchive);
        second.Dispose();
        Assert.IsFalse(Directory.Exists(secondRoot));

        using var replay = OnnxTypedDecisionsTransformer.Load(
            new MLContext(seed: 4),
            secondArchive);
        AssertResponsesEqual(expected, replay.Infer("independent load state"));
    }

    [TestMethod]
    public async Task TrackedFreshProcessHarnessRoundTripsStructuredFacadeStagesAndComposedArtifacts()
    {
        using var fixture = PortableBundleFixture.CreateExternal();
        var artifacts = new Dictionary<string, string>(StringComparer.Ordinal);
        var writerSnapshots = new Dictionary<string, string>(StringComparer.Ordinal);
        var repoRoot = FindRepositoryRoot();
        var trackedHarness = Path.Combine(
            repoRoot,
            "samples",
            "TypedDecisions",
            "PortableProcessHarness",
            "Program.cs");
        var coldHarnessRoot = Path.Combine(
            repoRoot,
            "samples",
            "TypedDecisions",
            $"PortableProcessHarness.Run.{Guid.NewGuid():N}");
        var coldHarness = Path.Combine(coldHarnessRoot, "Program.cs");
        Directory.CreateDirectory(coldHarnessRoot);
        File.Copy(trackedHarness, coldHarness);
        var writerRoot = Path.Combine(fixture.Root, "writer-archives");
        var movedRoot = Path.Combine(fixture.Root, "moved-archives");
        Directory.CreateDirectory(writerRoot);
        Directory.CreateDirectory(movedRoot);

        try
        {
            foreach (var kind in new[] { "facade", "stages", "composed" })
            {
                var writerArchive = Path.Combine(writerRoot, $"{kind}-process.zip");
                var movedArchive = Path.Combine(movedRoot, $"{kind}-process.zip");
                writerSnapshots[kind] = await RunPortableProcess(
                    "writer",
                    kind,
                    fixture.ModelAssetsPath,
                    writerArchive,
                    coldHarness,
                    noRestore: kind != "facade");
                Assert.IsTrue(File.Exists(writerArchive));
                File.Move(writerArchive, movedArchive);
                Assert.IsFalse(File.Exists(writerArchive));
                Assert.IsTrue(File.Exists(movedArchive));
                artifacts[kind] = movedArchive;
            }

            fixture.DeleteSourceAssets();
            Assert.IsFalse(Directory.Exists(fixture.ModelAssetsPath));
            foreach (var kind in artifacts.Keys)
            {
                Assert.IsFalse(Directory.Exists(fixture.ModelAssetsPath));
                var readerSnapshot = await RunPortableProcess(
                    "reader",
                    kind,
                    assets: null,
                    archive: artifacts[kind],
                    harness: coldHarness,
                    noRestore: true);
                using var expected = JsonDocument.Parse(writerSnapshots[kind]);
                using var actual = JsonDocument.Parse(readerSnapshot);
                AssertJsonEquivalent(expected.RootElement, actual.RootElement);
            }
        }
        finally
        {
            if (Directory.Exists(coldHarnessRoot))
                Directory.Delete(coldHarnessRoot, recursive: true);
        }
    }

    private static async Task<string> RunPortableProcess(
        string mode,
        string kind,
        string? assets,
        string archive,
        string harness,
        bool noRestore)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = "dotnet",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        startInfo.ArgumentList.Add("run");
        startInfo.ArgumentList.Add("--file");
        startInfo.ArgumentList.Add(harness);
        if (noRestore)
            startInfo.ArgumentList.Add("--no-restore");
        startInfo.ArgumentList.Add("--");
        startInfo.ArgumentList.Add("--mode");
        startInfo.ArgumentList.Add(mode);
        startInfo.ArgumentList.Add("--kind");
        startInfo.ArgumentList.Add(kind);
        startInfo.ArgumentList.Add("--portable-path");
        startInfo.ArgumentList.Add(archive);
        if (!string.IsNullOrWhiteSpace(assets))
        {
            startInfo.ArgumentList.Add("--model-assets");
            startInfo.ArgumentList.Add(assets);
        }

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Could not start portable process harness.");
        var standardOutputTask = process.StandardOutput.ReadToEndAsync();
        var standardErrorTask = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(2));
        try
        {
            await process.WaitForExitAsync(timeout.Token);
        }
        catch (OperationCanceledException) when (timeout.IsCancellationRequested)
        {
            try
            {
                if (!process.HasExited)
                    process.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException)
            {
            }

            await process.WaitForExitAsync();
            var timedOutOutput = await standardOutputTask;
            var timedOutError = await standardErrorTask;
            Assert.Fail(
                $"Portable process harness timed out for {mode}/{kind}. " +
                $"stderr: {timedOutError}\nstdout: {timedOutOutput}");
        }

        var standardOutput = await standardOutputTask;
        var standardError = await standardErrorTask;
        Assert.AreEqual(
            0,
            process.ExitCode,
            $"Portable process harness failed for {mode}/{kind}: {standardError}\n{standardOutput}");
        var json = standardOutput
            .Split(["\r\n", "\n"], StringSplitOptions.RemoveEmptyEntries)
            .LastOrDefault(static line => line.TrimStart().StartsWith('{'));
        Assert.IsFalse(
            string.IsNullOrWhiteSpace(json),
            $"Portable process harness did not emit structured JSON: {standardOutput}");
        return json!;
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null &&
               !File.Exists(Path.Combine(directory.FullName, "MLNet.TextInference.Onnx.slnx")))
            directory = directory.Parent;
        return directory?.FullName
            ?? throw new DirectoryNotFoundException("Could not locate the repository root.");
    }

    private static void AssertJsonEquivalent(
        JsonElement expected,
        JsonElement actual,
        string path = "$")
    {
        Assert.AreEqual(
            expected.ValueKind,
            actual.ValueKind,
            $"JSON kind mismatch at {path}.");
        switch (expected.ValueKind)
        {
            case JsonValueKind.Object:
                var expectedProperties = expected.EnumerateObject()
                    .ToDictionary(
                        static property => property.Name,
                        static property => property.Value,
                        StringComparer.Ordinal);
                var actualProperties = actual.EnumerateObject()
                    .ToDictionary(
                        static property => property.Name,
                        static property => property.Value,
                        StringComparer.Ordinal);
                CollectionAssert.AreEquivalent(
                    expectedProperties.Keys.ToArray(),
                    actualProperties.Keys.ToArray(),
                    $"JSON properties differ at {path}.");
                foreach (var property in expectedProperties)
                    AssertJsonEquivalent(
                        property.Value,
                        actualProperties[property.Key],
                        $"{path}.{property.Key}");
                break;
            case JsonValueKind.Array:
                var expectedValues = expected.EnumerateArray().ToArray();
                var actualValues = actual.EnumerateArray().ToArray();
                Assert.AreEqual(expectedValues.Length, actualValues.Length, path);
                for (var index = 0; index < expectedValues.Length; index++)
                    AssertJsonEquivalent(
                        expectedValues[index],
                        actualValues[index],
                        $"{path}[{index}]");
                break;
            case JsonValueKind.Number:
                Assert.AreEqual(
                    expected.GetDouble(),
                    actual.GetDouble(),
                    0.0001,
                    $"JSON number mismatch at {path}.");
                break;
            case JsonValueKind.String:
                Assert.AreEqual(expected.GetString(), actual.GetString(), path);
                break;
            case JsonValueKind.True:
            case JsonValueKind.False:
                Assert.AreEqual(expected.GetBoolean(), actual.GetBoolean(), path);
                break;
            case JsonValueKind.Null:
                break;
        }
    }

    private static int CountOwnedExtractionRoots()
    {
        var root = Path.Combine(Path.GetTempPath(), "mlnet-typed-decision-portable");
        return Directory.Exists(root)
            ? Directory.GetDirectories(root).Length
            : 0;
    }

    private static IReadOnlyList<DecisionQuestion> Questions() =>
    [
        DecisionQuestion.Choice(
            "priority",
            "How urgent?",
            new Dictionary<string, string?>(StringComparer.Ordinal)
            {
                ["zebra"] = null,
                ["!"] = string.Empty,
                ["alpha"] = "explicit"
            }),
        DecisionQuestion.Score("quality", "How strong?", ["weak", "moderate", "strong"]),
        DecisionQuestion.Noul(
            "actionable",
            "Can it be acted on?",
            new NoulCriteria(True: null, False: string.Empty))
    ];

    private static void AssertResponsesEqual(
        DecisionResponse expected,
        DecisionResponse actual)
    {
        Assert.AreEqual(expected.InputTokenCount, actual.InputTokenCount);
        Assert.AreEqual(expected.Results.Count, actual.Results.Count);
        for (var index = 0; index < expected.Results.Count; index++)
        {
            var left = expected.Results[index];
            var right = actual.Results[index];
            Assert.AreEqual(left.Id, right.Id);
            Assert.AreEqual(left.Type, right.Type);
            Assert.AreEqual(left.Confidence, right.Confidence, 0.000001f);
            Assert.AreEqual(left.ActionProbability, right.ActionProbability, 0.000001f);
            CollectionAssert.AreEqual(
                left.Distribution.Labels.ToArray(),
                right.Distribution.Labels.ToArray());
            CollectionAssert.AreEqual(
                left.Distribution.Probabilities.ToArray(),
                right.Distribution.Probabilities.ToArray());
            switch (left, right)
            {
                case (ChoiceDecisionResult expectedChoice, ChoiceDecisionResult actualChoice):
                    Assert.AreEqual(expectedChoice.Choice, actualChoice.Choice);
                    break;
                case (ScoreDecisionResult expectedScore, ScoreDecisionResult actualScore):
                    Assert.AreEqual(expectedScore.Score, actualScore.Score);
                    CollectionAssert.AreEqual(
                        expectedScore.Legend.ToArray(),
                        actualScore.Legend.ToArray());
                    break;
                case (NoulDecisionResult expectedNoul, NoulDecisionResult actualNoul):
                    Assert.AreEqual(expectedNoul.Value, actualNoul.Value);
                    Assert.AreEqual(
                        expectedNoul.ProbabilityTrue,
                        actualNoul.ProbabilityTrue,
                        0.000001f);
                    break;
                default:
                    Assert.Fail($"Unexpected decision result type '{left.GetType()}'.");
                    break;
            }
        }
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
            actual.DenseValues().Select(static value => value.ToString()).ToArray());
    }

    private static void AssertFacadeRowMatches(
        FacadeOutputRow row,
        DecisionResponse expected)
    {
        var choice = expected.Results.OfType<ChoiceDecisionResult>().Single();
        var score = expected.Results.OfType<ScoreDecisionResult>().Single();
        var noul = expected.Results.OfType<NoulDecisionResult>().Single();
        Assert.AreEqual(choice.Choice, row.Portable_priority_PredictedLabel);
        CollectionAssert.AreEqual(
            choice.Distribution.Probabilities.ToArray(),
            row.Portable_priority_Probabilities.DenseValues().ToArray());
        Assert.AreEqual(score.Score, row.Portable_quality_Score, 0.000001f);
        CollectionAssert.AreEqual(
            score.Distribution.Probabilities.ToArray(),
            row.Portable_quality_Probabilities.DenseValues().ToArray());
        Assert.AreEqual(noul.Value, row.Portable_actionable_PredictedLabel);
        using var json = JsonDocument.Parse(row.PortableResults);
        Assert.AreEqual(
            expected.InputTokenCount,
            json.RootElement.GetProperty("input_tokens").GetInt32());
    }

    private static void AssertComposedRowsEqual(
        IReadOnlyList<ComposedOutputRow> expected,
        IReadOnlyList<ComposedOutputRow> actual)
    {
        Assert.AreEqual(expected.Count, actual.Count);
        for (var index = 0; index < expected.Count; index++)
            AssertComposedRowMatches(actual[index], expected[index]);
    }

    private static void AssertComposedRowMatches(
        ComposedOutputRow actual,
        ComposedOutputRow expected)
    {
        Assert.AreEqual(expected.PortableResults, actual.PortableResults);
        Assert.AreEqual(
            expected.Portable_priority_PredictedLabel,
            actual.Portable_priority_PredictedLabel);
        AssertFloatVectorsEqual(
            expected.Portable_priority_Probabilities.DenseValues().ToArray(),
            actual.Portable_priority_Probabilities.DenseValues().ToArray());
        Assert.AreEqual(expected.Portable_quality_Score, actual.Portable_quality_Score);
        AssertFloatVectorsEqual(
            expected.Portable_quality_Probabilities.DenseValues().ToArray(),
            actual.Portable_quality_Probabilities.DenseValues().ToArray());
        Assert.AreEqual(
            expected.Portable_actionable_PredictedLabel,
            actual.Portable_actionable_PredictedLabel);
        using (var expectedJson = JsonDocument.Parse(expected.AppendedDecisionResults))
        using (var actualJson = JsonDocument.Parse(actual.AppendedDecisionResults))
        {
            Assert.AreEqual(
                expectedJson.RootElement.GetProperty("input_tokens").GetInt32(),
                actualJson.RootElement.GetProperty("input_tokens").GetInt32());
        }
        Assert.AreEqual(
            expected.Appended_binary_PredictedLabel,
            actual.Appended_binary_PredictedLabel);
        AssertFloatVectorsEqual(
            expected.Appended_binary_Probabilities.DenseValues().ToArray(),
            actual.Appended_binary_Probabilities.DenseValues().ToArray());
    }

    private static void AssertFloatVectorsEqual(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual)
    {
        Assert.AreEqual(expected.Count, actual.Count);
        for (var index = 0; index < expected.Count; index++)
            Assert.AreEqual(expected[index], actual[index], 0.00001f);
    }

    private static void CopyArchive(
        string sourcePath,
        string destinationPath,
        string? transformedEntry,
        Func<string, string>? transform)
    {
        using var source = ZipFile.OpenRead(sourcePath);
        using var destination = ZipFile.Open(destinationPath, ZipArchiveMode.Create);
        foreach (var entry in source.Entries)
        {
            if (string.Equals(entry.FullName, transformedEntry, StringComparison.Ordinal) &&
                transform is null)
                continue;

            using var input = entry.Open();
            using var buffer = new MemoryStream();
            input.CopyTo(buffer);
            var bytes = buffer.ToArray();
            if (string.Equals(entry.FullName, transformedEntry, StringComparison.Ordinal))
            {
                var text = Encoding.UTF8.GetString(bytes);
                bytes = Encoding.UTF8.GetBytes(transform!(text));
            }

            var copy = destination.CreateEntry(entry.FullName);
            using var output = copy.Open();
            output.Write(bytes);
        }
    }

    private sealed class StateRow
    {
        public string Evidence { get; set; } = string.Empty;
    }

    private sealed class PipelineRow
    {
        public string Evidence { get; set; } = string.Empty;
        public VBuffer<long> PortableInputIds { get; set; }
        public VBuffer<float> PortableLogits { get; set; }
        public string PortableResults { get; set; } = string.Empty;
        public string Portable_priority_PredictedLabel { get; set; } = string.Empty;
        public float Portable_quality_Score { get; set; }
        public bool Portable_actionable_PredictedLabel { get; set; }
    }

    private sealed class FacadeOutputRow
    {
        public string PortableResults { get; set; } = string.Empty;
        public string Portable_priority_PredictedLabel { get; set; } = string.Empty;
        public VBuffer<float> Portable_priority_Probabilities { get; set; }
        public float Portable_quality_Score { get; set; }
        public VBuffer<float> Portable_quality_Probabilities { get; set; }
        public bool Portable_actionable_PredictedLabel { get; set; }
    }

    private sealed class ComposedOutputRow
    {
        public string Evidence { get; set; } = string.Empty;
        public string PortableResults { get; set; } = string.Empty;
        public string Portable_priority_PredictedLabel { get; set; } = string.Empty;
        public VBuffer<float> Portable_priority_Probabilities { get; set; }
        public float Portable_quality_Score { get; set; }
        public VBuffer<float> Portable_quality_Probabilities { get; set; }
        public bool Portable_actionable_PredictedLabel { get; set; }
        public string AppendedDecisionResults { get; set; } = string.Empty;
        public string Appended_binary_PredictedLabel { get; set; } = string.Empty;
        public VBuffer<float> Appended_binary_Probabilities { get; set; }
    }

    private sealed class StageOutputRow
    {
        public VBuffer<long> PortableInputIds { get; set; }
        public VBuffer<float> PortableLogits { get; set; }
        public string PortableResults { get; set; } = string.Empty;
        public VBuffer<float> Portable_priority_Probabilities { get; set; }
        public VBuffer<float> Portable_quality_Probabilities { get; set; }
    }

    private sealed class DecoderOutputRow
    {
        public string Decision_priority_PredictedLabel { get; set; } = string.Empty;
        public VBuffer<float> Decision_priority_Probabilities { get; set; }
        public float Decision_quality_Score { get; set; }
        public bool Decision_actionable_PredictedLabel { get; set; }
    }

    private sealed class ScoredRow
    {
        public VBuffer<long> DecisionInputIds { get; set; } = new(0, []);
        public VBuffer<long> DecisionAttentionMask { get; set; } = new(0, []);
        public VBuffer<long> DecisionMarkerPositions { get; set; } = new(0, []);
        public VBuffer<bool> DecisionMarkerMask { get; set; } = new(0, []);
        public VBuffer<long> DecisionQuestionTypes { get; set; } = new(0, []);
        public int DecisionBatchSize { get; set; } = 3;
        public int DecisionSequenceLength { get; set; } = 64;
        public int DecisionMarkerWidth { get; set; } = 3;
        public VBuffer<float> DecisionLogits { get; set; } = new(6, [0, 1, 0, 1, 0, 1]);
        public VBuffer<float> DecisionActionProbabilities { get; set; } = new(2, [0.25f, 0.75f]);
    }

    private sealed class DefaultScoredRow
    {
        public VBuffer<long> DecisionInputIds { get; set; } =
            new(192, Enumerable.Repeat(1L, 192).ToArray());
        public VBuffer<long> DecisionAttentionMask { get; set; } =
            new(192, Enumerable.Repeat(1L, 192).ToArray());
        public VBuffer<long> DecisionMarkerPositions { get; set; } =
            new(9, [0, 1, 2, 0, 1, 2, 0, 1, 2]);
        public VBuffer<bool> DecisionMarkerMask { get; set; } =
            new(9, [true, true, true, true, true, true, true, true, true]);
        public VBuffer<long> DecisionQuestionTypes { get; set; } =
            new(3, [0, 1, 2]);
        public int DecisionBatchSize { get; set; } = 3;
        public int DecisionSequenceLength { get; set; } = 64;
        public int DecisionMarkerWidth { get; set; } = 3;
        public VBuffer<float> DecisionLogits { get; set; } =
            new(9, [0, 1, 2, 0, 1, 2, 0, 1, 2]);
        public VBuffer<float> DecisionActionProbabilities { get; set; } =
            new(6, [0.25f, 0.75f, 0.25f, 0.75f, 0.25f, 0.75f]);
    }

    private sealed class PortableBundleFixture : IDisposable
    {
        private readonly TypedDecisionNativeDataViewTests.NativeBundleFixture? _nativeFixture;


        private PortableBundleFixture(string root)
        {
            Root = root;
            ModelAssetsPath = Path.Combine(root, "bundle");
            Directory.CreateDirectory(ModelAssetsPath);

            File.WriteAllText(
                Path.Combine(ModelAssetsPath, "laya_config.json"),
                """{"max_len":64,"head_max_len":16,"temperature":[1.25,0.75,2.0]}""");
            var tokenizer = Path.Combine(ModelAssetsPath, "tokenizer");
            Directory.CreateDirectory(tokenizer);
            File.WriteAllText(
                Path.Combine(tokenizer, "tokenizer.json"),
                """
                {
                  "version": "1.0",
                  "normalizer": {"type": "NFC"},
                  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": true},
                  "model": {
                    "type": "BPE", "unk_token": "<unk>",
                    "vocab": {
                      "<unk>": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
                      "Ã„Â ": 5, "a": 6, "b": 7, "c": 8, "d": 9, "e": 10, "f": 11,
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
        }

        private PortableBundleFixture(
            TypedDecisionNativeDataViewTests.NativeBundleFixture nativeFixture)
        {
            _nativeFixture = nativeFixture;
            Root = nativeFixture.Root;
            ModelAssetsPath = nativeFixture.ModelAssetsPath;
        }

        public string Root { get; }
        public string ModelAssetsPath { get; }

        public static PortableBundleFixture Create()
            => new(TypedDecisionNativeDataViewTests.NativeBundleFixture.Create());

        public static PortableBundleFixture CreateProfileOnly()
            => new(
                Path.Combine(
                    Path.GetTempPath(),
                    "typed-decisions-portable-profile-tests",
                    Guid.NewGuid().ToString("N")));

        public static PortableBundleFixture CreateExternal()
        {
            var fixture = new PortableBundleFixture(
                Path.Combine(
                    Path.GetTempPath(),
                    "typed-decisions-portable-external-tests",
                    Guid.NewGuid().ToString("N")));
            File.WriteAllBytes(
                Path.Combine(fixture.ModelAssetsPath, "laya.onnx"),
                Convert.FromBase64String(ExternalModelBase64));
            File.WriteAllBytes(
                Path.Combine(fixture.ModelAssetsPath, "weights.bin"),
                Convert.FromBase64String(ExternalWeightsBase64));
            TypedDecisionBundle.WriteManifest(
                fixture.ModelAssetsPath,
                new TypedDecisionBundleManifest
                {
                    ModelFile = "laya.onnx",
                    ExternalDataFiles = ["weights.bin"],
                    TokenizerDirectory = "tokenizer"
                });
            return fixture;
        }

        public OnnxTypedDecisionsOptions CreateFacadeOptions(
            IReadOnlyList<DecisionQuestion> questions)
            => new()
            {
                ModelAssetsPath = ModelAssetsPath,
                Questions = questions,
                StateColumnName = "Evidence",
                OutputPrefix = "Portable_",
                ResultsColumnName = "PortableResults",
                BatchSize = 1
            };

        public DecisionInputPreparationOptions CreatePreparationOptions(
            IReadOnlyList<DecisionQuestion> questions)
            => new()
            {
                ModelAssetsPath = ModelAssetsPath,
                Questions = questions,
                StateColumnName = "Evidence",
                InputIdsColumnName = "PortableInputIds",
                AttentionMaskColumnName = "PortableAttentionMask",
                MarkerPositionsColumnName = "PortableMarkerPositions",
                MarkerMaskColumnName = "PortableMarkerMask",
                QuestionTypesColumnName = "PortableQuestionTypes",
                BatchSizeColumnName = "PortableBatchSize",
                SequenceLengthColumnName = "PortableSequenceLength",
                MarkerWidthColumnName = "PortableMarkerWidth"
            };

        public OnnxDecisionModelScorerOptions CreateScoringOptions()
            => new()
            {
                ModelAssetsPath = ModelAssetsPath,
                InputIdsColumnName = "PortableInputIds",
                AttentionMaskColumnName = "PortableAttentionMask",
                MarkerPositionsColumnName = "PortableMarkerPositions",
                MarkerMaskColumnName = "PortableMarkerMask",
                QuestionTypesColumnName = "PortableQuestionTypes",
                BatchSizeColumnName = "PortableBatchSize",
                SequenceLengthColumnName = "PortableSequenceLength",
                MarkerWidthColumnName = "PortableMarkerWidth",
                LogitsColumnName = "PortableLogits",
                ActionProbabilitiesColumnName = "PortableActionProbabilities",
                BatchSize = 2
            };

        public DecisionDecodingOptions CreateDecodingOptions(
            IReadOnlyList<DecisionQuestion> questions,
            string? modelAssetsPath = null)
            => new()
            {
                ModelAssetsPath = modelAssetsPath ?? ModelAssetsPath,
                Questions = questions,
                InputIdsColumnName = "PortableInputIds",
                AttentionMaskColumnName = "PortableAttentionMask",
                MarkerPositionsColumnName = "PortableMarkerPositions",
                MarkerMaskColumnName = "PortableMarkerMask",
                QuestionTypesColumnName = "PortableQuestionTypes",
                BatchSizeColumnName = "PortableBatchSize",
                SequenceLengthColumnName = "PortableSequenceLength",
                MarkerWidthColumnName = "PortableMarkerWidth",
                LogitsColumnName = "PortableLogits",
                ActionProbabilitiesColumnName = "PortableActionProbabilities",
                OutputPrefix = "Portable_",
                ResultsColumnName = "PortableResults"
            };

        public void DeleteSourceAssets()
        {
            if (Directory.Exists(ModelAssetsPath))
                Directory.Delete(ModelAssetsPath, recursive: true);
        }

        public void SetTemperature(double temperature)
        {
            File.WriteAllText(
                Path.Combine(ModelAssetsPath, "laya_config.json"),
                $$"""{"max_len":64,"head_max_len":16,"temperature":[{{temperature}},{{temperature}},{{temperature}}]}""");
            using var bundle = TypedDecisionBundle.OpenDirectory(
                ModelAssetsPath,
                requirements: TypedDecisionBundleLoadRequirements.None);
            TypedDecisionBundle.WriteManifest(ModelAssetsPath, bundle.Manifest);
        }

        public void Dispose()
        {
            if (_nativeFixture is not null)
            {
                _nativeFixture.Dispose();
                return;
            }

            if (Directory.Exists(Root))
                Directory.Delete(Root, recursive: true);
        }

        private const string ExternalModelBase64 = "CAkSDnBvcnRhYmxlLXRlc3RzOpgICjISC3JlZHVjZV9heGVzIghDb25zdGFudCoZCgV2YWx1ZSoNCAEQBzoBAUIEYXhlc6ABBAovEgN0d28iCENvbnN0YW50Kh4KBXZhbHVlKhIIARAHOgECQgl0d29fdmFsdWWgAQQKKwoKbWFya2VyX3BvcxIMbWFya2VyX2Zsb2F0IgRDYXN0KgkKAnRvGAGgAQIKMwoMbWFya2VyX2Zsb2F0CgxtYXJrZXJfc2NhbGUSEG1hcmtlcl9jb21wb25lbnQiA011bAo/CglpbnB1dF9pZHMKC3JlZHVjZV9heGVzEglzdGF0ZV9zdW0iCVJlZHVjZVN1bSoPCghrZWVwZGltcxgBoAECCikKCXN0YXRlX3N1bRILc3RhdGVfZmxvYXQiBENhc3QqCQoCdG8YAaABAgowCgtzdGF0ZV9mbG9hdAoLc3RhdGVfc2NhbGUSD3N0YXRlX2NvbXBvbmVudCIDTXVsCjAKEG1hcmtlcl9jb21wb25lbnQKD3N0YXRlX2NvbXBvbmVudBIGbG9naXRzIgNBZGQKFgoFcXR5cGUSBnFzaGFwZSIFU2hhcGUKMAoGcXNoYXBlCgN0d28SDGFjdGlvbl9zaGFwZSIGQ29uY2F0KgsKBGF4aXMYAKABAgouCgthY3Rpb25fYmFzZQoMYWN0aW9uX3NoYXBlEglhY3RfcHJvYnMiBkV4cGFuZBIedHlwZWRfZXh0ZXJuYWxfc3RhdGVfc2Vuc2l0aXZlKkUQAUIMbWFya2VyX3NjYWxlahcKCGxvY2F0aW9uEgt3ZWlnaHRzLmJpbmoLCgZvZmZzZXQSATBqCwoGbGVuZ3RoEgE0cAEqRBABQgtzdGF0ZV9zY2FsZWoXCghsb2NhdGlvbhILd2VpZ2h0cy5iaW5qCwoGb2Zmc2V0EgE0agsKBmxlbmd0aBIBNHABKkgIAQgCEAFCC2FjdGlvbl9iYXNlahcKCGxvY2F0aW9uEgt3ZWlnaHRzLmJpbmoLCgZvZmZzZXQSAThqCwoGbGVuZ3RoEgE4cAFaKAoJaW5wdXRfaWRzEhsKGQgHEhUKBxIFYmF0Y2gKChIIc2VxdWVuY2VaLQoOYXR0ZW50aW9uX21hc2sSGwoZCAcSFQoHEgViYXRjaAoKEghzZXF1ZW5jZVooCgptYXJrZXJfcG9zEhoKGAgHEhQKBxIFYmF0Y2gKCRIHbWFya2Vyc1opCgttYXJrZXJfbWFzaxIaChgICRIUCgcSBWJhdGNoCgkSB21hcmtlcnNaGAoFcXR5cGUSDwoNCAcSCQoHEgViYXRjaGIkCgZsb2dpdHMSGgoYCAESFAoHEgViYXRjaAoJEgdtYXJrZXJzYiAKCWFjdF9wcm9icxITChEIARINCgcSBWJhdGNoCgIIAkIECgAQEg==";

        private const string ExternalWeightsBase64 = "exQuPqabRDsAAIA+AABAPw==";
    }
}
