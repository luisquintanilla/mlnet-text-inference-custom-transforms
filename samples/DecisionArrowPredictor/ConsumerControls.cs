using System.Diagnostics;
using Apache.Arrow;
using Apache.Arrow.Ipc;
using DecisionInference.Arrow;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DecisionArrowPredictor;

public sealed record ConsumerMeasurement(string Scope, string Path, int Rows, int BatchSize, int Pair, string Order,
    long ManagedBytes, int Gen0, int Gen1, int Gen2, double WallMilliseconds, double CpuMilliseconds,
    long WorkingSetBytes, long PeakWorkingSetBytes, long NumericCapacityBytes);

public static class ConsumerControls
{
    private static readonly int[] MatrixRows = [257, 4097, 65537, 1048577];
    private static readonly int[] MatrixBatches = [1, 256, 4096];

    private static ConsumerMeasurement Measure(string scope, string path, int rows, int batch, int pair,
        string order, Action action, long numericCapacity = 0)
    {
        using var process = Process.GetCurrentProcess();
        int gen0 = GC.CollectionCount(0), gen1 = GC.CollectionCount(1), gen2 = GC.CollectionCount(2);
        var cpu = process.TotalProcessorTime;
        long allocated = GC.GetTotalAllocatedBytes(true);
        long start = Stopwatch.GetTimestamp();
        action();
        long finish = Stopwatch.GetTimestamp();
        long bytes = GC.GetTotalAllocatedBytes(true) - allocated;
        double elapsed = Stopwatch.GetElapsedTime(start, finish).TotalMilliseconds;
        process.Refresh();
        return new(scope, path, rows, batch, pair, order, bytes, GC.CollectionCount(0) - gen0,
            GC.CollectionCount(1) - gen1, GC.CollectionCount(2) - gen2, elapsed,
            (process.TotalProcessorTime - cpu).TotalMilliseconds, process.WorkingSet64, process.PeakWorkingSet64, numericCapacity);
    }

    public static async Task ProjectionAsync(string referenceDirectory, string output, int? onlyRows = null, int? onlyBatch = null)
    {
        if (File.Exists(output)) throw new IOException("Control measurements are immutable; choose a new output.");
        if (onlyRows.HasValue && !MatrixRows.Contains(onlyRows.Value) ||
            onlyBatch.HasValue && !MatrixBatches.Contains(onlyBatch.Value))
            throw new ArgumentException("Use predeclared rows 257/4097/65537/1048577 and batches 1/256/4096.");
        using var oracle = new OriginalConsumerOracle(referenceDirectory);
        string fixture = Path.Combine(AppContext.BaseDirectory, "fixtures", "HighPrecision");
        var contract = DecisionArrowSchema.FromCanonicalJson(File.ReadAllText(Path.Combine(fixture, "contract.json")));
        using var reader = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(fixture, "manifest.json"), contract);
        using var template = await reader.ReadNextRecordBatchAsync();
        if (template is null) throw new InvalidDataException("Missing synthetic fixture template.");
        var measurements = new List<ConsumerMeasurement>();
        foreach (int rows in onlyRows.HasValue ? [onlyRows.Value] : MatrixRows)
            foreach (int batchSize in onlyBatch.HasValue ? [onlyBatch.Value] : MatrixBatches)
            {
                // Exclude one warmup for each path. Synthetic construction/Arrow slice setup is outside projection timing.
                _ = RunProjection(template, oracle, rows, batchSize, -1, "warmup", true);
                _ = RunProjection(template, oracle, rows, batchSize, -1, "warmup", false);
                for (int pair = 0; pair < 5; pair++)
                {
                    bool firstLegacy = pair % 2 == 0;
                    string order = firstLegacy ? "AB" : "BA";
                    measurements.Add(RunProjection(template, oracle, rows, batchSize, pair, order, firstLegacy));
                    measurements.Add(RunProjection(template, oracle, rows, batchSize, pair, order, !firstLegacy));
                }
            }
        ArtifactFiles.Write(output, new
        {
            version = 1, status = "MEASURED", referenceReceiptSha256 = oracle.ReceiptSha256,
            sourceAssemblySha256 = ArtifactFiles.Hash(typeof(ConsumerControls).Assembly.Location),
            runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
            scope = "Managed projection/materialization only; equivalent ten Single/direct Double content. " +
                "Synthetic Arrow construction/slicing excluded, accessor binding and numeric capacity allocation included. " +
                "Initial IO/public-reader full scan separate. Managed GC bytes are not native/process memory. " +
                "Projection wall time sums timed projection intervals; CPU/WS include excluded synthetic setup and are " +
                "whole-run diagnostics, not projection CPU estimates. No extractor inference or model fitting. " +
                "Five balanced AB/BA pairs; warmups excluded.",
            synthetic = "Frozen fixture probabilities; gapped nonmonotonic IDs Id(i)=even?10000000+3*i:-5*i, " +
                "4096-row parents sliced into requested batches and final short batch; parent offsets included.",
            measurements
        });
    }

    private static ConsumerMeasurement RunProjection(RecordBatch template, OriginalConsumerOracle oracle,
        int rows, int batchSize, int pair, string order, bool legacy)
    {
        using var process = Process.GetCurrentProcess();
        var cpuStart = process.TotalProcessorTime;
        ProbabilityStore? store = null;
        long managed = 0; int gen0 = 0, gen1 = 0, gen2 = 0;
        double wall = 0;
        long ws = 0, peak = 0;
        Timed(() => { if (!legacy) store = new ProbabilityStore(rows); });
        try
        {
            for (int chunkStart = 0; chunkStart < rows; chunkStart += 4096)
            {
                int chunkRows = Math.Min(4096, rows - chunkStart);
                using var parent = SyntheticBatch(template, chunkStart, chunkRows);
                for (int offset = 0; offset < chunkRows; offset += batchSize)
                {
                    int count = Math.Min(batchSize, chunkRows - offset);
                    using var batch = parent.Slice(offset, count);
                    int destination = chunkStart + offset;
                    System.Array? observations = null;
                    Timed(() =>
                    {
                        if (legacy) observations = oracle.Project(batch);
                        else
                        {
                            var accessor = new ProbabilityBatchAccessor(batch);
                            for (int row = 0; row < count; row++)
                            {
                                long id = accessor.RowId(row);
                                if (id != SyntheticId(destination + row)) throw new InvalidDataException("Synthetic source order drift.");
                                store!.SetDirect(destination + row, accessor.CopyRow(row, store.WritableSemantic(destination + row)));
                            }
                        }
                    });
                    GC.KeepAlive(observations);
                }
                process.Refresh();
                ws = Math.Max(ws, process.WorkingSet64); peak = Math.Max(peak, process.PeakWorkingSet64);
            }
            return new("projection", legacy ? "original" : "compact", rows, batchSize, pair, order,
                managed, gen0, gen1, gen2, wall, (process.TotalProcessorTime - cpuStart).TotalMilliseconds,
                ws, peak, store?.NumericCapacityBytes ?? 0);
        }
        finally { store?.Dispose(); }

        void Timed(Action action)
        {
            int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
            long allocated = GC.GetTotalAllocatedBytes(true);
            long start = Stopwatch.GetTimestamp();
            action();
            long finish = Stopwatch.GetTimestamp();
            managed = checked(managed + GC.GetTotalAllocatedBytes(true) - allocated);
            wall += Stopwatch.GetElapsedTime(start, finish).TotalMilliseconds;
            gen0 += GC.CollectionCount(0) - g0; gen1 += GC.CollectionCount(1) - g1; gen2 += GC.CollectionCount(2) - g2;
        }
    }

    private static long SyntheticId(int row) => row % 2 == 0 ? 10000000L + 3L * row : -5L * row;
    private static DoubleArray Repeated(double? value, int count)
    {
        var builder = new DoubleArray.Builder();
        builder.Reserve(count);
        for (int i = 0; i < count; i++)
            if (value.HasValue) builder.Append(value.Value); else builder.AppendNull();
        return builder.Build();
    }

    private static RecordBatch SyntheticBatch(RecordBatch template, int start, int count)
    {
        var pressure = (StructArray)template.Column(3);
        var purpose = (StructArray)template.Column(4);
        var time = (FixedSizeListArray)pressure.Fields[4];
        var intent = (FixedSizeListArray)purpose.Fields[1];
        var timeBuilder = new DoubleArray.Builder().Reserve(checked(count * 3));
        var intentBuilder = new DoubleArray.Builder().Reserve(checked(count * 5));
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < 3; j++) timeBuilder.Append(((DoubleArray)time.Values).GetValue(j)!.Value);
            for (int j = 0; j < 5; j++) intentBuilder.Append(((DoubleArray)intent.Values).GetValue(j)!.Value);
        }
        if (purpose.Fields[0] is not Int32Array selectedOrdinals)
            throw new InvalidDataException("Expected the pinned Arrow selected-candidate Int32 ordinal field.");
        var selected = new Int32Array.Builder();
        for (int i = 0; i < count; i++) selected.Append(selectedOrdinals.GetValue(0)!.Value);
        using var authored = new RecordBatch(template.Schema,
        [
            new Int64Array.Builder().AppendRange(Enumerable.Range(start, count).Select(i => SyntheticId(i))).Build(),
            Repeated(((DoubleArray)template.Column(1)).GetValue(0), count),
            Repeated(((DoubleArray)template.Column(2)).GetValue(0), count),
            new StructArray(pressure.Data.DataType, count,
                [.. pressure.Fields.Take(4).Select(a => Repeated(((DoubleArray)a).GetValue(0), count)),
                    new FixedSizeListArray(time.Data.DataType, count, timeBuilder.Build(), ArrowBuffer.Empty)], ArrowBuffer.Empty),
            new StructArray(purpose.Data.DataType, count,
                [selected.Build(), new FixedSizeListArray(intent.Data.DataType, count, intentBuilder.Build(), ArrowBuffer.Empty)], ArrowBuffer.Empty),
            Repeated(((DoubleArray)template.Column(5)).GetValue(0), count)
        ], count);
        // The original oracle consumes IPC reader batches, not builder-owned native buffers.
        using var memory = new MemoryStream();
        using (var writer = new ArrowStreamWriter(memory, authored.Schema, leaveOpen: true))
        {
            writer.WriteStartAsync().GetAwaiter().GetResult();
            writer.WriteRecordBatchAsync(authored).GetAwaiter().GetResult();
            writer.WriteEndAsync().GetAwaiter().GetResult();
        }
        memory.Position = 0;
        using var ipc = new ArrowStreamReader(memory, leaveOpen: true);
        return ipc.ReadNextRecordBatchAsync().GetAwaiter().GetResult() ??
            throw new InvalidDataException("Synthetic IPC roundtrip returned no batch.");
    }

    public static async Task SameModelAsync(string referenceDirectory, string[] importPaths,
        string modelPath, string modelReceiptPath, string output, int predictionCursors = 1,
        bool partitionedSourceControl = false)
    {
        if (File.Exists(output)) throw new IOException("Control receipts are immutable.");
        using var oracle = new OriginalConsumerOracle(referenceDirectory);
        var original = await oracle.ImportAsync(importPaths);
        using var compact = await ArrowFeatureReader.ImportCompactAsync(importPaths[0], importPaths[1], importPaths[2],
            importPaths[3], importPaths[4], importPaths[5], importPaths[6]);
        oracle.RequireContent(original, compact);
        var receipt = ArtifactFiles.Read<ModelReceipt>(modelReceiptPath);
        var model = PredictorTraining.Load(modelPath, receipt, compact.FeatureFingerprint);
        var expected = OriginalConsumerOracle.Predictions(oracle.Predict(model, oracle.Rows(original)));
        var selection = compact.All();
        var predictionView = partitionedSourceControl ? selection.PartitionedPredictionControlView() : selection.View();
        var buffer = new PredictionBuffer(selection.Count);
        buffer.Fill(model, predictionView, selection, requestedCursors: predictionCursors);
        buffer.RequireReplay(expected);
        var loaded = PredictorTraining.Load(modelPath, receipt, compact.FeatureFingerprint);
        buffer.Fill(loaded, predictionView, selection, requestedCursors: predictionCursors);
        buffer.RequireReplay(expected);
        var measurements = new List<ConsumerMeasurement>();
        _ = oracle.Predict(model, oracle.Rows(original));
        buffer.Fill(model, predictionView, selection, requestedCursors: predictionCursors);
        for (int pair = 0; pair < 5; pair++)
        {
            bool ab = pair % 2 == 0;
            string order = ab ? "AB" : "BA";
            for (int run = 0; run < 2; run++)
            {
                bool legacy = ab == (run == 0);
                measurements.Add(Measure("prediction-materialization", legacy ? "original" : "compact",
                    selection.Count, 0, pair, order, () =>
                    {
                        if (legacy) GC.KeepAlive(oracle.Predict(model, oracle.Rows(original)));
                        else buffer.Fill(model, predictionView, selection, requestedCursors: predictionCursors);
                    }, compact.NumericCapacityBytes));
            }
        }
        var view = selection.View();
        var column = view.Schema["Semantic"];
        void Pass()
        {
            using var cursor = view.GetRowCursor([column]);
            var get = cursor.GetGetter<VBuffer<float>>(column);
            VBuffer<float> value = default;
            while (cursor.MoveNext()) get(ref value);
        }
        Pass();
        measurements.Add(Measure("requested-semantic-getter", "compact", selection.Count, 0, 0, "warm", Pass, compact.NumericCapacityBytes));
        ArtifactFiles.Write(output, new
        {
            version = 1, status = "SAME_MODEL_REPLAY_PASS", rows = selection.Count,
            compact.FeatureFingerprint, compact.DatasetManifestSha256, compact.SplitSha256, compact.QuestionsSha256,
            modelSha256 = ArtifactFiles.Hash(modelPath), referenceReceiptSha256 = oracle.ReceiptSha256,
            sourceAssemblySha256 = ArtifactFiles.Hash(typeof(ConsumerControls).Assembly.Location),
            exactAssociation = true, predictionTolerance = 1e-6, inference = "Existing ML.NET saved head only; no new model extraction or fitting.",
            exactFeatureBitsAndOriginalText = true,
            predictionExecutionProfile = new { requestedCursors = predictionCursors,
                partitionedSourceControl,
                publicOutputCursorSet = true, disjointSourceIdsRequired = true, selectionRankOrderRestored = true,
                customScorer = false, predictionCache = false, trainingPolicyChanged = false },
            measurements
        });
    }

    public static async Task AllocationDiagnosticsAsync(string[] importPaths, string modelPath,
        string modelReceiptPath, string output)
    {
        if (File.Exists(output)) throw new IOException("Diagnostic receipts are immutable.");
        using var compact = await ArrowFeatureReader.ImportCompactAsync(importPaths[0], importPaths[1], importPaths[2],
            importPaths[3], importPaths[4], importPaths[5], importPaths[6]);
        var receipt = ArtifactFiles.Read<ModelReceipt>(modelReceiptPath);
        var model = PredictorTraining.Load(modelPath, receipt, compact.FeatureFingerprint);
        var selected = compact.All();
        var view = selected.View();
        var outputView = model.Transform(view);
        var input = Diagnose(view, selected);
        var scored = Diagnose(outputView, selected);
        var buffer = new PredictionBuffer(selected.Count);
        buffer.Fill(model, view, selected);
        var fillStages = new List<object>();
        long processStart = GC.GetTotalAllocatedBytes(true);
        long threadStart = GC.GetAllocatedBytesForCurrentThread();
        buffer.Fill(model, view, selected, (stage, bytes) => fillStages.Add(new { stage, bytes }));
        long fillThreadBytes = GC.GetAllocatedBytesForCurrentThread() - threadStart;
        long fillProcessBytes = GC.GetTotalAllocatedBytes(true) - processStart;
        ArtifactFiles.Write(output, new
        {
            version = 1, status = "DIAGNOSTIC_ONLY", rows = selected.Count, compact.FeatureFingerprint,
            modelSha256 = ArtifactFiles.Hash(modelPath),
            scope = "Thread-local managed bytes by MoveNext/getter/active-column check; not a timing or acceptance run. " +
                "First row excluded as warmup. No new fit/extractor inference.",
            input, scored, fillStages, fillThreadBytes, fillProcessBytes
        });
    }

    public static void JuliaSameModel(StudyData compact, ImportedStudy original, string referenceDirectory,
        string trainingFreezePath, string trainingFreezeSha256, string evaluationPath, string evaluationSha256,
        string output, int predictionCursors = 1, bool partitionedSourceControl = false)
    {
        if (File.Exists(output) || PredictorTraining.LearnedArms.Any(name =>
            File.Exists(output + $".{name}.same-model.json") || File.Exists(output + $".{name}.measurements.json")))
            throw new IOException("Julia benchmark receipts, including partial evidence, are immutable.");
        JuliaStudyImport.RequireLegacyParity(original, compact);
        if (compact.FeatureFingerprint != JuliaFixtureInterop.HighPrecisionFingerprint ||
            compact.Metadata.Count != 5574 || compact.Partition("train").Count != 3344 ||
            predictionCursors is < 1 or > PredictionBuffer.MaximumControlCursors)
            throw new InvalidDataException("Julia benchmark requires the full frozen CPU source and a declared cursor profile.");
        ArtifactFiles.RequireHash(trainingFreezePath, trainingFreezeSha256);
        ArtifactFiles.RequireHash(evaluationPath, evaluationSha256);
        var freeze = ArtifactFiles.Read<TrainingFreeze>(trainingFreezePath);
        var evaluation = ArtifactFiles.Read<StudyEvaluation>(evaluationPath);
        if (freeze.Version != 1 || freeze.Status != "complete" || evaluation.Version != 1 ||
            evaluation.Status != "complete" || evaluation.TrainingFreezeSha256 != trainingFreezeSha256 ||
            freeze.FeatureFingerprint != compact.FeatureFingerprint || evaluation.FeatureFingerprint != compact.FeatureFingerprint ||
            freeze.DatasetManifestSha256 != compact.DatasetManifestSha256 ||
            evaluation.DatasetManifestSha256 != compact.DatasetManifestSha256 ||
            freeze.SplitSha256 != compact.SplitSha256 || evaluation.SplitSha256 != compact.SplitSha256 ||
            freeze.QuestionsSha256 != compact.QuestionsSha256 || freeze.Conversion != FeatureContract.Conversion ||
            freeze.Arms.Length != 15 || evaluation.Arms.Length != 15 ||
            evaluation.Arms.Any(arm => arm.Holdout.Rows != 1114))
            throw new InvalidDataException("Julia benchmarks follow the completed frozen study; never open holdout during selection.");
        using var oracle = new OriginalConsumerOracle(referenceDirectory);
        var rows = oracle.BridgeRows(original);
        oracle.RequireRowsContent(rows, compact);
        var selection = compact.All();
        var view = partitionedSourceControl ? selection.PartitionedPredictionControlView() : selection.View();
        var buffer = new PredictionBuffer(selection.Count);
        var measurements = new List<ConsumerMeasurement>();
        var models = new List<object>();
        string directory = Path.GetDirectoryName(Path.GetFullPath(trainingFreezePath))!;
        foreach (string name in PredictorTraining.LearnedArms)
        {
            var arm = freeze.Arms.Single(arm => arm.Arm == name && arm.TargetRows == 3344);
            string filename = arm.ModelReceiptFile ?? throw new InvalidDataException("Missing frozen Julia model receipt.");
            if (Path.GetFileName(filename) != filename) throw new InvalidDataException("Model receipt must be a sibling file.");
            string receiptPath = Path.Combine(directory, filename);
            ArtifactFiles.RequireHash(receiptPath, arm.ModelReceiptSha256 ?? "");
            var receipt = ArtifactFiles.Read<ModelReceipt>(receiptPath);
            if (receipt.Arm != name || receipt.TargetRows != 3344 || receipt.ActualRows != 3344 ||
                receipt.DatasetManifestSha256 != compact.DatasetManifestSha256 ||
                receipt.SplitSha256 != compact.SplitSha256 || receipt.QuestionsSha256 != compact.QuestionsSha256 ||
                receipt.Threshold != arm.Threshold || receipt.BudgetThreshold != arm.BudgetThreshold ||
                Path.GetFileName(receipt.ModelFile) != receipt.ModelFile)
                throw new InvalidDataException("Julia benchmark model differs from the completed frozen study.");
            string path = Path.Combine(directory, receipt.ModelFile);
            var model = PredictorTraining.Load(path, receipt, compact.FeatureFingerprint, view.Schema);
            var expected = OriginalConsumerOracle.Predictions(oracle.Predict(model, rows));
            buffer.Fill(model, view, selection, requestedCursors: predictionCursors);
            var actual = buffer.Values;
            bool complete = expected.Length == actual.Count;
            bool finite = expected.All(row => double.IsFinite(row.Probability) && row.Probability is >= 0 and <= 1);
            double? difference = complete && finite ? Enumerable.Range(0, expected.Length)
                .Select(i => Math.Abs(expected[i].Probability - actual[i].Probability)).DefaultIfEmpty(0).Max() : null;
            ArtifactFiles.Write(output + $".{name}.same-model.json", new
            {
                schemaVersion = 1, status = "SAME_MODEL_COMPARISON_RECORDED", arm = name, receipt.ModelSha256,
                expectedRows = expected.Length, actualRows = actual.Count, finiteOriginalProbabilities = finite,
                completeAssociation = complete && Enumerable.Range(0, expected.Length).All(i =>
                    expected[i].RowId == actual[i].RowId && expected[i].GroupId == actual[i].GroupId &&
                    expected[i].Label == actual[i].Label),
                maximumAbsoluteProbabilityDifference = difference, predictionTolerance = 1e-6
            });
            if (!finite) throw new InvalidDataException("Frozen original predictor returned invalid probabilities.");
            buffer.RequireReplay(expected);
            buffer.Fill(PredictorTraining.Load(path, receipt, compact.FeatureFingerprint, view.Schema),
                view, selection, requestedCursors: predictionCursors);
            buffer.RequireReplay(expected);
            _ = oracle.Predict(model, rows);
            buffer.Fill(model, view, selection, requestedCursors: predictionCursors);
            for (int pair = 0; pair < 5; pair++)
            {
                bool ab = pair % 2 == 0;
                string order = ab ? "AB" : "BA";
                for (int run = 0; run < 2; run++)
                {
                    bool legacy = ab == (run == 0);
                    measurements.Add(Measure($"julia-{name}-prediction-materialization", legacy ? "original" : "compact",
                        selection.Count, 0, pair, order, () =>
                        {
                            if (legacy) GC.KeepAlive(oracle.Predict(model, rows));
                            else buffer.Fill(model, view, selection, requestedCursors: predictionCursors);
                        }, compact.NumericCapacityBytes));
                }
            }
            ArtifactFiles.Write(output + $".{name}.measurements.json", new
            {
                schemaVersion = 1, arm = name, receipt.ModelSha256, rows = selection.Count,
                predictionCursors, partitionedSourceControl,
                measurements = measurements.Where(measurement => measurement.Scope == $"julia-{name}-prediction-materialization")
            });
            buffer.RequireReplay(expected);
            models.Add(new { arm = name, receipt.ModelSha256, receipt.ModelBytes, receipt.L2,
                completeAssociationAndSaveLoadReplay = true, predictionTolerance = 1e-6 });
        }
        ArtifactFiles.Write(output, new
        {
            schemaVersion = 1, status = "JULIA_SAME_MODEL_BENCHMARK_REPLAY_PASS",
            compact.FeatureFingerprint, compact.DatasetManifestSha256, compact.SplitSha256, compact.QuestionsSha256,
            trainingFreezeSha256, evaluationSha256, referenceReceiptSha256 = oracle.ReceiptSha256,
            sourceAssemblySha256 = ArtifactFiles.Hash(typeof(ConsumerControls).Assembly.Location),
            rows = selection.Count, models, measurements,
            predictionExecutionProfile = new { requestedCursors = predictionCursors, partitionedSourceControl,
                publicOutputCursorSet = true, customScorer = false, predictionCache = false, trainingPolicyChanged = false },
            scope = "Frozen original Predict versus reusable compact columns through the same three genuinely new saved Julia heads; " +
                "full5574 rows, five balanced AB/BA pairs per head, warmups excluded. Full import/IO/public-reader validation, " +
                "model load, bridge construction and replay checks excluded from matched hot prediction scope. " +
                "Managed GC bytes are not native or process memory; CPU/WS remain separately reported. " +
                "70% reduction is a reported benchmark, never a retroactive PASS for v5/v6; no new fitting/extraction or holdout tuning."
        });
    }

    private static object Diagnose(IDataView view, RowSelection expected)
    {
        var schema = view.Schema;
        var idColumn = schema["RowId"]; var groupColumn = schema["GroupId"]; var labelColumn = schema["Label"];
        var probability = schema.GetColumnOrNull("Probability");
        bool hasProbability = probability.HasValue;
        var valueColumn = probability ?? schema["Semantic"];
        long setupStart = GC.GetAllocatedBytesForCurrentThread();
        using var cursor = view.GetRowCursor([idColumn, groupColumn, labelColumn, valueColumn]);
        long cursorSetupBytes = GC.GetAllocatedBytesForCurrentThread() - setupStart;
        setupStart = GC.GetAllocatedBytesForCurrentThread();
        var id = cursor.GetGetter<long>(idColumn); var group = cursor.GetGetter<long>(groupColumn);
        var label = cursor.GetGetter<bool>(labelColumn);
        var scalar = hasProbability ? cursor.GetGetter<float>(valueColumn) : null;
        var vector = scalar is null ? cursor.GetGetter<VBuffer<float>>(valueColumn) : null;
        long getterSetupBytes = GC.GetAllocatedBytesForCurrentThread() - setupStart;
        long moveBytes = 0, idBytes = 0, groupBytes = 0, labelBytes = 0, valueBytes = 0, activeBytes = 0;
        long rowId = 0, groupId = 0; bool isSpam = false; float score = 0; VBuffer<float> values = default;
        int rows = 0;
        while (true)
        {
            long start = GC.GetAllocatedBytesForCurrentThread();
            bool next = cursor.MoveNext();
            long move = GC.GetAllocatedBytesForCurrentThread() - start;
            if (!next) break;
            start = GC.GetAllocatedBytesForCurrentThread(); id(ref rowId);
            long i = GC.GetAllocatedBytesForCurrentThread() - start;
            start = GC.GetAllocatedBytesForCurrentThread(); group(ref groupId);
            long g = GC.GetAllocatedBytesForCurrentThread() - start;
            start = GC.GetAllocatedBytesForCurrentThread(); label(ref isSpam);
            long l = GC.GetAllocatedBytesForCurrentThread() - start;
            start = GC.GetAllocatedBytesForCurrentThread();
            if (scalar is null) vector!(ref values); else scalar(ref score);
            long v = GC.GetAllocatedBytesForCurrentThread() - start;
            start = GC.GetAllocatedBytesForCurrentThread(); _ = cursor.IsColumnActive(valueColumn);
            long a = GC.GetAllocatedBytesForCurrentThread() - start;
            if (rows > 0) { moveBytes += move; idBytes += i; groupBytes += g; labelBytes += l; valueBytes += v; activeBytes += a; }
            var metadata = expected.Owner.Metadata[expected[rows]];
            if (rowId != metadata.RowId || groupId != metadata.GroupId || isSpam != metadata.Label)
                throw new InvalidDataException("Diagnostic source association changed.");
            rows++;
        }
        if (rows != expected.Count) throw new InvalidDataException("Diagnostic row count differs.");
        return new { rows, cursorSetupBytes, getterSetupBytes, moveBytes, idBytes, groupBytes, labelBytes, valueBytes, activeBytes };
    }
}
