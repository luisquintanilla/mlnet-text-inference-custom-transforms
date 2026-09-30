using System.Security.Cryptography;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace DecisionArrowPredictor;

public static class ConsumerFitControls
{
    private sealed record Candidate(string Arm, double L2, Metrics Metrics, double Threshold, double BudgetThreshold,
        string FeatureSchemaSha256, CursorTrace[] SourceTrace, CursorTrace[] LearnerTrace);
    private sealed record Fitted(ITransformer Model, Candidate Candidate, Prediction[] Predictions);

    public static async Task VerifyAsync(string[] importPaths, string output, string? onlyArm = null,
        double? onlyL2 = null, CancellationToken token = default)
    {
        var legacy = await ArrowFeatureReader.ImportAsync(importPaths[0], importPaths[1], importPaths[2], importPaths[3],
            importPaths[4], importPaths[5], importPaths[6], token);
        using var compact = await ArrowFeatureReader.ImportCompactAsync(importPaths[0], importPaths[1], importPaths[2],
            importPaths[3], importPaths[4], importPaths[5], importPaths[6], token: token);
        Verify(legacy, compact, output, onlyArm, onlyL2);
    }

    public static void Verify(ImportedStudy legacy, StudyData compact, string output, string? onlyArm = null,
        double? onlyL2 = null, string? originalProjectionReferenceReceiptSha256 = null)
    {
        bool singleCandidate = onlyArm is not null && onlyL2.HasValue;
        if ((onlyArm is not null) != onlyL2.HasValue ||
            (singleCandidate && (!PredictorTraining.LearnedArms.Contains(onlyArm, StringComparer.Ordinal) ||
                !PredictorTraining.L2Grid.Contains(onlyL2!.Value))))
            throw new ArgumentException("A single-candidate control needs both a declared learned arm and an exact frozen L2.");
        if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
            throw new IOException("Bounded fit-control output must be empty; failed evidence is retained.");
        Directory.CreateDirectory(output);
        var training = compact.Partition("train").Subset(32);
        var validation = compact.Partition("validation").Subset(32);
        if (training.Count > 128 || validation.Count > 128)
            throw new InvalidDataException("Whole-group bounded fit controls exceed 128 rows; no silent group truncation.");
        long[] trainIds = training.SourceIds(), validationIds = validation.SourceIds();
        var legacyTrain = Select(legacy, trainIds);
        var legacyValidation = Select(legacy, validationIds);
        if (!PredictorTraining.Subset(legacy.Rows.Where(r => legacy.Split.Rows
                .Any(s => s.RowId == r.RowId && s.Split == "train")).ToArray(), 32)
            .Select(r => r.RowId).SequenceEqual(trainIds))
            throw new InvalidDataException("Bounded grouped subset policy changed.");
        var controls = new List<object>();
        try
        {
            foreach (string arm in singleCandidate ? [onlyArm!] : PredictorTraining.LearnedArms)
            {
                var candidates = new List<(Fitted Original, Fitted Compact)>();
                foreach (double l2 in singleCandidate ? [onlyL2!.Value] : PredictorTraining.L2Grid)
                {
                    var original = Fit(arm, l2, false, false);
                    var sameModelCompact = Predict(original.Model);
                    ArtifactFiles.Write(Path.Combine(output, $"{arm}-{l2:R}.same-model.json"), new
                    {
                        arm, l2, comparison = Compare(original.Predictions, sameModelCompact),
                        sourceModel = "Same original independently fitted candidate; no compact fit yet.",
                        compact.FeatureFingerprint, compact.DatasetManifestSha256
                    });
                    RequirePredictions(original.Predictions, sameModelCompact);
                    var optimized = Fit(arm, l2, true, false);
                    var originalTraced = Fit(arm, l2, false, true);
                    var optimizedTraced = Fit(arm, l2, true, true);
                    ArtifactFiles.Write(Path.Combine(output, $"{arm}-{l2:R}.trace.raw.json"), new
                    {
                        arm, l2, original = original.Candidate, compact = optimized.Candidate,
                        tracedOriginal = originalTraced.Candidate, tracedCompact = optimizedTraced.Candidate,
                        sameFittedModelComparison = Compare(original.Predictions, sameModelCompact),
                        untracedComparison = Compare(original.Predictions, optimized.Predictions),
                        tracedComparison = Compare(originalTraced.Predictions, optimizedTraced.Predictions),
                        originalObserverComparison = Compare(original.Predictions, originalTraced.Predictions),
                        compactObserverComparison = Compare(optimized.Predictions, optimizedTraced.Predictions)
                    });
                    RequirePredictions(original.Predictions, optimized.Predictions);
                    RequireMetrics(original.Candidate.Metrics, optimized.Candidate.Metrics);
                    RequireCandidate(original.Candidate, optimized.Candidate);

                    RequirePredictions(original.Predictions, originalTraced.Predictions);
                    RequirePredictions(optimized.Predictions, optimizedTraced.Predictions);
                    RequireCandidate(original.Candidate, originalTraced.Candidate);
                    RequireCandidate(optimized.Candidate, optimizedTraced.Candidate);
                    RequireCandidate(originalTraced.Candidate, optimizedTraced.Candidate);
                    RequireTrace(originalTraced.Candidate.SourceTrace, optimizedTraced.Candidate.SourceTrace);
                    RequireTrace(originalTraced.Candidate.LearnerTrace, optimizedTraced.Candidate.LearnerTrace);
                    controls.Add(new
                    {
                        arm, l2, original = original.Candidate, compact = optimized.Candidate,
                        tracedOriginal = originalTraced.Candidate, tracedCompact = optimizedTraced.Candidate,
                        independentFitReplayPass = true, instrumentationDidNotChangePredictions = true
                    });
                    ArtifactFiles.Write(Path.Combine(output, $"{arm}-{l2:R}.candidate.json"), controls[^1]);
                    candidates.Add((original, optimized));
                }
                if (singleCandidate) continue;
                int originalBest = Best(candidates.Select(c => c.Original.Candidate).ToArray());
                int compactBest = Best(candidates.Select(c => c.Compact.Candidate).ToArray());
                if (originalBest != compactBest) throw new InvalidDataException("Validation-selected L2 changed.");
                SaveReplay(arm, "original", candidates[originalBest].Original,
                    new MLContext(1).Data.LoadFromEnumerable(legacyTrain).Schema);
                SaveReplay(arm, "compact", candidates[compactBest].Compact, training.View().Schema);
            }
            ArtifactFiles.Write(Path.Combine(output, "fit-control.receipt.json"), new
            {
                schemaVersion = 1, status = singleCandidate ? "BOUNDED_SINGLE_CANDIDATE_PARITY_PASS" :
                    "BOUNDED_INDEPENDENT_FIT_PARITY_PASS",
                compact.FeatureFingerprint, compact.DatasetManifestSha256, compact.SplitSha256, compact.QuestionsSha256,
                sourceAssemblySha256 = ArtifactFiles.Hash(typeof(ConsumerFitControls).Assembly.Location),
                originalProjectionReferenceReceiptSha256,
                trainingRows = training.Count, validationRows = validation.Count,
                trainingIdOrderSha256 = HashIds(trainIds), validationIdOrderSha256 = HashIds(validationIds),
                settings = "ML.NET5/seed1/thread1/max100/defaultShuffle/L2[.0001,.001,.01]; unchanged grouped subset; validation only.",
                trace = "Default-off public forwarding source and at-SDCA-input wrappers; local DataViewRowId/order/active columns/feature bits, no raw text; supplied Random never advanced by observer.",
                cacheShuffleAudit = "ML.NET5 StreamingDataView nonshuffle/localIDs; SDCA injects standard RowShufflingTransformer pool1000/host-derived forced seed. No manual cache, seed or shuffle policy.",
                sourceHostLifecycle = "Real StudyDataView registered once per candidate through public MLContext/IHostEnvironment before estimator construction, matching legacy source host draw order.",
                onlyArm, onlyL2, completeGridAndWinningSaveReplay = !singleCandidate,
                predictionTolerance = 1e-6, metricTolerance = 1e-12, exactThresholdsSlotsL2AndTrace = true,
                holdoutOpened = false, fullStudyReady = false, controls
            });
        }
        catch (Exception error) when (error is InvalidDataException or InvalidOperationException or ArgumentException)
        {
            ArtifactFiles.Write(Path.Combine(output, "fit-control.failure.json"), new
            {
                schemaVersion = 1, status = "BOUNDED_FIT_PARITY_FAILED", error = error.GetType().Name,
                reason = error.Message, completedCandidates = controls.Count, holdoutOpened = false,
                sourceAssemblySha256 = ArtifactFiles.Hash(typeof(ConsumerFitControls).Assembly.Location)
            });
            throw;
        }

        static object Compare(Prediction[] expected, Prediction[] actual) => new
        {
            expectedRows = expected.Length, actualRows = actual.Length,
            associationExact = expected.Length == actual.Length &&
                expected.Select((p, i) => p.RowId == actual[i].RowId && p.GroupId == actual[i].GroupId &&
                    p.Label == actual[i].Label).All(value => value),
            maximumAbsoluteProbabilityDifference = expected.Length == actual.Length ?
                expected.Select((p, i) => Math.Abs(p.Probability - actual[i].Probability)).DefaultIfEmpty(0).Max() : (double?)null
        };

        Fitted Fit(string arm, double l2, bool optimized, bool instrument)
        {
            var context = new MLContext(PredictorTraining.Seed);
            IDataView source = optimized ? training.View(context) : context.Data.LoadFromEnumerable(legacyTrain);
            DataViewTrace? sourceTrace = instrument ? new(source, trainIds) : null;
            DataViewTrace? learnerTrace = null;
            Func<IDataView, IDataView>? observe = instrument ? input =>
            {
                learnerTrace = new DataViewTrace(input, trainIds);
                return learnerTrace;
            } : null;
            var estimator = optimized ? CompactPredictorTraining.Estimator(context, arm, l2, observe) :
                LegacyEstimator(context, arm, l2, observe);
            var model = estimator.Fit(sourceTrace ?? source);
            var predictions = optimized ? Predict(model) : PredictorTraining.Predict(model, legacyValidation);
            double threshold = PredictorEvaluation.SelectThreshold(predictions, false);
            double budget = PredictorEvaluation.SelectThreshold(predictions, true);
            var candidate = new Candidate(arm, l2, PredictorEvaluation.Calculate(predictions, .5), threshold, budget,
                SchemaHash(model.Transform(source)), sourceTrace?.Snapshot() ?? [], learnerTrace?.Snapshot() ?? []);
            return new(model, candidate, predictions);
        }

        // Independent construction follows the frozen legacy Fit, not the optimized factory.
        static IEstimator<ITransformer> LegacyEstimator(MLContext context, string arm, double l2,
            Func<IDataView, IDataView>? observe)
        {
            IEstimator<ITransformer> features = arm switch
            {
                "text" => context.Transforms.Text.FeaturizeText("Features", "Text"),
                "semantic" => context.Transforms.CopyColumns("Features", "Semantic"),
                "combined" => context.Transforms.Text.FeaturizeText("TextFeatures", "Text")
                    .Append(context.Transforms.Concatenate("Features", "TextFeatures", "Semantic")),
                _ => throw new ArgumentException("Unknown control arm.", nameof(arm))
            };
            IEstimator<ITransformer> learner = context.BinaryClassification.Trainers.SdcaLogisticRegression(
                new SdcaLogisticRegressionBinaryTrainer.Options
                {
                    LabelColumnName = "Label", FeatureColumnName = "Features", L2Regularization = (float)l2,
                    MaximumNumberOfIterations = 100, NumberOfThreads = 1
                });
            return features.Append(observe is null ? learner : new TracedEstimator(learner, observe));
        }

        Prediction[] Predict(ITransformer model)
        {
            var buffer = new PredictionBuffer(validation.Count);
            buffer.Fill(model, validation.View(), validation);
            return buffer.Snapshot();
        }

        void SaveReplay(string arm, string pathName, Fitted fitted, DataViewSchema actualInputSchema)
        {
            string path = Path.Combine(output, $"{arm}-{pathName}.mlnet");
            var context = new MLContext(PredictorTraining.Seed);
            using (var file = new FileStream(path, FileMode.CreateNew)) context.Model.Save(fitted.Model, actualInputSchema, file);
            ITransformer loaded;
            DataViewSchema savedSchema;
            using (var file = File.OpenRead(path)) loaded = context.Model.Load(file, out savedSchema);
            if (actualInputSchema.Count != savedSchema.Count ||
                actualInputSchema.Where((column, i) => column.Name != savedSchema[i].Name ||
                    !column.Type.Equals(savedSchema[i].Type)).Any())
                throw new InvalidDataException("Standard save/load actual input schema changed.");
            RequirePredictions(fitted.Predictions, Predict(loaded));
            ArtifactFiles.Write(path + ".replay.json", new
            {
                arm, pathName, modelSha256 = ArtifactFiles.Hash(path), bytes = new FileInfo(path).Length,
                rows = validation.Count, completeSourceGroupLabelReplayPass = true, tolerance = 1e-6
            });
        }
    }

    private static LearningRow[] Select(ImportedStudy study, long[] ids)
    {
        var rows = study.Rows.ToDictionary(r => r.RowId);
        return ids.Select(id => rows.TryGetValue(id, out var row) ? row :
            throw new InvalidDataException("Original source is missing a declared control ID.")).ToArray();
    }

    private static string HashIds(long[] ids) => Convert.ToHexStringLower(SHA256.HashData(
        System.Runtime.InteropServices.MemoryMarshal.AsBytes(ids.AsSpan())));

    private static int Best(Candidate[] candidates) => Enumerable.Range(0, candidates.Length)
        .OrderByDescending(i => candidates[i].Metrics.Auprc).ThenBy(i => candidates[i].Metrics.LogLoss).First();

    private static string SchemaHash(IDataView output)
    {
        var column = output.Schema["Features"];
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hash.AppendData(ArtifactFiles.Utf8.GetBytes(column.Type.ToString() ??
            throw new InvalidDataException("Feature type has no stable description.")));
        if (column.Annotations.Schema.GetColumnOrNull("SlotNames").HasValue)
        {
            VBuffer<ReadOnlyMemory<char>> slots = default;
            column.Annotations.GetValue("SlotNames", ref slots);
            ReadOnlySpan<byte> separator = stackalloc byte[] { 0 };
            foreach (var slot in slots.DenseValues())
            {
                hash.AppendData(ArtifactFiles.Utf8.GetBytes(slot.ToString()));
                hash.AppendData(separator);
            }
        }
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    private static void RequireCandidate(Candidate expected, Candidate actual)
    {
        if (expected.Arm != actual.Arm || expected.L2 != actual.L2 || expected.Threshold != actual.Threshold ||
            expected.BudgetThreshold != actual.BudgetThreshold || expected.FeatureSchemaSha256 != actual.FeatureSchemaSha256)
            throw new InvalidDataException("Control vocabulary/slot/schema/L2/validation threshold changed.");
        RequireMetrics(expected.Metrics, actual.Metrics);
    }

    private static void RequireTrace(CursorTrace[] expected, CursorTrace[] actual)
    {
        if (expected.Length == 0 || expected.Length != actual.Length)
            throw new InvalidDataException("Actual fit cursor request count changed or trace is empty.");
        for (int i = 0; i < expected.Length; i++)
        {
            var a = expected[i]; var b = actual[i];
            if (a.Request != b.Request || a.Cursor != b.Cursor || a.Method != b.Method ||
                a.RequestedCount != b.RequestedCount || a.RandomSupplied != b.RandomSupplied ||
                !a.ActiveColumns.SequenceEqual(b.ActiveColumns) || a.Rows != b.Rows || a.Complete != b.Complete ||
                a.RowOrderSha256 != b.RowOrderSha256 || a.FeatureGetterCalls != b.FeatureGetterCalls ||
                a.FeatureBitsSha256 != b.FeatureBitsSha256 || a.LabelGetterCalls != b.LabelGetterCalls ||
                a.LabelBitsSha256 != b.LabelBitsSha256)
                throw new InvalidDataException("Actual learner/source requests, local IDs/order or feature bits changed.");
        }
    }

    private static void RequirePredictions(Prediction[] expected, Prediction[] actual)
    {
        if (expected.Length != actual.Length) throw new InvalidDataException("Independent fit prediction count changed.");
        for (int i = 0; i < expected.Length; i++)
            if (expected[i].RowId != actual[i].RowId || expected[i].GroupId != actual[i].GroupId ||
                expected[i].Label != actual[i].Label || !double.IsFinite(expected[i].Probability) ||
                expected[i].Probability is < 0 or > 1 ||
                !double.IsFinite(actual[i].Probability) || actual[i].Probability is < 0 or > 1 ||
                Math.Abs(expected[i].Probability - actual[i].Probability) > 1e-6)
                throw new InvalidDataException("Independent fit full association/prediction parity failed.");
    }

    private static void RequireMetrics(Metrics a, Metrics b)
    {
        bool Equal(double? x, double? y) => x.HasValue == y.HasValue && (!x.HasValue || Math.Abs(x.Value - y!.Value) <= 1e-12);
        if (a.Rows != b.Rows || a.Spam != b.Spam || a.Ham != b.Ham || a.Threshold != b.Threshold ||
            a.TruePositive != b.TruePositive || a.FalsePositive != b.FalsePositive ||
            a.TrueNegative != b.TrueNegative || a.FalseNegative != b.FalseNegative ||
            !Equal(a.Auprc, b.Auprc) || !Equal(a.RocAuc, b.RocAuc) || !Equal(a.LogLoss, b.LogLoss) ||
            !Equal(a.Brier, b.Brier) || !Equal(a.Precision, b.Precision) ||
            !Equal(a.Recall, b.Recall) || !Equal(a.FalsePositiveRate, b.FalsePositiveRate))
            throw new InvalidDataException("Independent fit integer/undefined/metric parity failed (unchanged 1e-12).");
    }
}
