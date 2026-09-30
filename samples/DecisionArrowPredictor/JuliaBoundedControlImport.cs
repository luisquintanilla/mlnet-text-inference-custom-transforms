using System.Text.Json;
using Apache.Arrow;
using DecisionInference.Arrow;

namespace DecisionArrowPredictor;

public static class JuliaBoundedControlImport
{
    public const string RuntimeSourceCommit = "fd14d3b1caf89886a5f6cc29fa6c01dcef199e01";
    public const string ManifestSha256 = "f00fbeda4045a2a8d0c17a54efd569b382bba9d1047ffe22bf9cfaf12d99845c";
    public const string DataSha256 = "444969cad288ebdb868528b66884a835c126b8ba8572a195f1ed960b6cfa0620";
    public const string ControlStatesSha256 = "46c32520da66d08180221c301bc9f955a63a4155b8f06bf47c19b8effda0cfa4";
    public const string SelectionSha256 = "dfba474494e9bc54a077a76d2254b1675ebaa1fce2cf7d5748a5e7372ed7608c";
    public const string QualificationSha256 = "6602c1f86a251855eb2f647dd700c6402870a0c1960c81d45d7484c2cfc50f07";
    private const string FrozenStatesSha256 = "c00636dc21f7a4e3bb099c508fbf24a407af546851e060c5852cb9aca54639fa";
    private const string FrozenSplitSha256 = "1f5967c9e6d228fda4322845a77ee1464dff6ec87ea578abafa16f679127cf6f";

    public static async Task VerifyPinned128Async(string dataset, string selectionPath, string controlStatesPath,
        string qualificationPath, string preparationPath, string splitPath, string statesPath, string questionsPath,
        string output, CancellationToken token = default)
    {
        if (File.Exists(output)) throw new IOException("Bounded Julia control receipts are immutable.");
        RequireFile(Path.Combine(dataset, "manifest.json"), 7068, ManifestSha256);
        RequireFile(Path.Combine(dataset, "contract.json"), 2448, JuliaFixtureInterop.HighPrecisionFingerprint);
        RequireFile(Path.Combine(dataset, "decisions.arrow"), 23520, DataSha256);
        ArtifactFiles.RequireHash(selectionPath, SelectionSha256);
        ArtifactFiles.RequireHash(controlStatesPath, ControlStatesSha256);
        ArtifactFiles.RequireHash(qualificationPath, QualificationSha256);
        using var qualification = JsonDocument.Parse(File.ReadAllBytes(qualificationPath));
        var profile = qualification.RootElement;
        if (profile.GetProperty("status").GetString() != "QUALIFIED-CPU-FALLBACK-FULL-GATED" ||
            profile.GetProperty("selectedMode").GetString() != "scalar-cpu" ||
            profile.GetProperty("featureFingerprint").GetString() != JuliaFixtureInterop.HighPrecisionFingerprint ||
            profile.GetProperty("runtimeSourceCommit").GetString() != RuntimeSourceCommit ||
            profile.GetProperty("packageSourceCommit").GetString() != "65880cdfd33580d23d0e833ac977f79acb18b1b0" ||
            profile.GetProperty("cpu").GetProperty("controls128").GetString() != "PASS")
            throw new InvalidDataException("Bounded controls need the exact selected producer CPU qualification, not full-study GO.");
        var (frozen, preparation) = ArrowFeatureReader.ReadFrozenMetadata(
            preparationPath, splitPath, statesPath, questionsPath);
        if (frozen.Count != 5574 || preparation.StatesSha256 != FrozenStatesSha256 ||
            preparation.SplitSha256 != FrozenSplitSha256)
            throw new InvalidDataException("Bounded controls require the original complete frozen5574 source and split.");
        using var selection = JsonDocument.Parse(File.ReadAllBytes(selectionPath));
        var selected = selection.RootElement;
        if (selected.GetProperty("status").GetString() != "FROZEN" ||
            selected.GetProperty("sourceHashes").GetProperty("states.v1.jsonl").GetString() != FrozenStatesSha256 ||
            selected.GetProperty("sourceHashes").GetProperty("split.v1.json").GetString() != FrozenSplitSha256)
            throw new InvalidDataException("Control selection differs from the frozen source/split receipt.");
        long[] ids = selected.GetProperty("sets").GetProperty("128").EnumerateArray().Select(value => value.GetInt64()).ToArray();
        var controls = ArrowFeatureReader.ReadStates(controlStatesPath);
        var metadata = Associate(frozen, ids, controls);
        using var imported = await ImportControlAsync(Path.Combine(dataset, "manifest.json"),
            Path.Combine(dataset, "contract.json"), questionsPath, metadata, ControlStatesSha256,
            preparation.SplitSha256, token: token);
        ArtifactFiles.Write(output, new
        {
            schemaVersion = 1, status = "BOUNDED_JULIA_CPU128_IMPORT_PARITY_PASS",
            sourceAssemblySha256 = ArtifactFiles.Hash(typeof(JuliaBoundedControlImport).Assembly.Location),
            adapterAssemblySha256 = ArtifactFiles.Hash(typeof(DecisionArrowDatasetReader).Assembly.Location),
            manifestSha256 = ManifestSha256, dataSha256 = DataSha256, selectionSha256 = SelectionSha256,
            qualificationSha256 = QualificationSha256, controlStatesSha256 = ControlStatesSha256,
            imported.FeatureFingerprint, imported.QuestionsSha256, imported.SplitSha256,
            rows = imported.Metadata.Count, imported.NumericCapacityBytes,
            exactSelectionOrderOriginalTextSourceGroupLabelAssociation = true,
            exactLegacyVersusCompactTenSingleAndSeparateDoubleBits = true,
            fullPublicReaderSchemaIdsHashEos = true, arrowNativeLeasesAfterImport = 0,
            partitions = Enumerable.Range(0, metadata.Count).Select(i => metadata[i].Partition)
                .GroupBy(name => name).ToDictionary(group => group.Key, group => group.Count()),
            inferenceOccurred = false, modelLoadedOrPredictedOrFitted = false, performanceMeasured = false,
            holdoutSelected = false, fullJuliaStudyImportEnabled = false, fullStudyReady = false,
            costScope = "Bounded producer control export only; no consumer-invented model-load measurement or full extraction cost."
        });
    }

    public static StudyMetadata Associate(StudyMetadata frozen, long[] ids, IReadOnlyDictionary<long, string> controls)
    {
        if (ids.Length is < 1 or > 128 || ids.Distinct().Count() != ids.Length || controls.Count != ids.Length ||
            !ids.ToHashSet().SetEquals(controls.Keys))
            throw new InvalidDataException("Bounded controls need1..128 unique independently selected source IDs and matching label-free states.");
        var metadata = new StudyRowMetadata[ids.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            var row = frozen[frozen.Ordinal(ids[i])];
            if (row.Partition == "holdout" || !string.Equals(row.Text, controls[ids[i]], StringComparison.Ordinal))
                throw new InvalidDataException("Control source text changed or selected a holdout row.");
            metadata[i] = row;
        }
        return new(metadata);
    }

    public static async Task<StudyData> ImportControlAsync(string manifestPath, string contractPath, string questionsPath,
        StudyMetadata metadata, string expectedInputSha256, string splitSha256,
        long numericCapBytes = ProbabilityStore.DefaultNumericCapBytes, CancellationToken token = default)
    {
        if (metadata.Count is < 1 or > 128 ||
            Enumerable.Range(0, metadata.Count).Any(i => metadata[i].Partition == "holdout"))
            throw new InvalidDataException("This control-only importer cannot open a full corpus or holdout selection.");
        var contract = ArrowFeatureReader.ExpectedContract(contractPath,
            JuliaFixtureInterop.HighPrecisionFingerprint, questionsPath);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(manifestPath, contract,
            Enumerable.Range(0, metadata.Count).Select(i => metadata[i].RowId), token);
        var provenance = reader.Manifest.Provenance;
        if (reader.Manifest.RowCount != metadata.Count || reader.Manifest.OutputBatchSize != 256 ||
            provenance.SourceRepository != "luisquintanilla/typesafe-meai" ||
            provenance.SourceCommit != RuntimeSourceCommit || provenance.ExecutionMode != "scalar-cpu" ||
            provenance.InputSha256 != expectedInputSha256 || provenance.QuestionsSha256 != FeatureContract.QuestionsV1Sha256 ||
            provenance.NativeStateBatchSize != 1 || provenance.CpuThreads != 4)
            throw new InvalidDataException("Bounded Julia input/source/count/mode must match the declared scalar-cpu profile exactly.");
        var measures = provenance.Measurements ??
            throw new InvalidDataException("Bounded producer export measurements are missing.");
        if (!measures.TryGetValue("exportBeforeManifestPublicationMilliseconds", out double export) ||
            !double.IsFinite(export) || export < 0)
            throw new InvalidDataException("Bounded producer export measurement is invalid.");
        var store = new ProbabilityStore(metadata.Count, numericCapBytes);
        try
        {
            int count = 0;
            RecordBatch? batch;
            while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
            {
                using (batch)
                {
                    var original = ArrowFeatureReader.Project(batch);
                    var accessor = new ProbabilityBatchAccessor(batch);
                    for (int row = 0; row < batch.Length; row++)
                    {
                        int ordinal = checked(count + row);
                        if (ordinal >= metadata.Count || accessor.RowId(row) != metadata[ordinal].RowId ||
                            original[row].RowId != metadata[ordinal].RowId)
                            throw new InvalidDataException("Bounded Arrow IDs/order differ from the independent control selection.");
                        var destination = store.WritableSemantic(ordinal);
                        double direct = accessor.CopyRow(row, destination);
                        if (BitConverter.DoubleToInt64Bits(direct) != BitConverter.DoubleToInt64Bits(original[row].SpamBaseline))
                            throw new InvalidDataException("Bounded Julia direct Double bits differ from legacy projection.");
                        for (int column = 0; column < FeatureContract.Width; column++)
                            if (BitConverter.SingleToInt32Bits(destination[column]) !=
                                BitConverter.SingleToInt32Bits(original[row].Semantic[column]))
                                throw new InvalidDataException("Bounded Julia ten Single bits/order differ from legacy projection.");
                        store.SetDirect(ordinal, direct);
                    }
                    count = checked(count + batch.Length);
                }
            }
            if (count != metadata.Count) throw new InvalidDataException("Bounded Julia fullEOS/count gate failed.");
            token.ThrowIfCancellationRequested();
            return new(metadata, store, contract.FeatureFingerprint, ArtifactFiles.Hash(manifestPath), splitSha256,
                FeatureContract.QuestionsV1Sha256, new ExtractionCost(provenance.ExecutionMode, count,
                    new Dictionary<string, double>(measures, StringComparer.Ordinal),
                    "Bounded producer control export only; not complete corpus extraction or model-load cost."));
        }
        catch
        {
            store.Dispose();
            throw;
        }
    }

    private static void RequireFile(string path, long bytes, string hash)
    {
        if (new FileInfo(path).Length != bytes)
            throw new InvalidDataException("Bounded Julia artifact size differs from independent pin.");
        ArtifactFiles.RequireHash(path, hash);
    }
}
