using System.Text.Json;
using Apache.Arrow;
using DecisionInference.Arrow;

namespace DecisionArrowPredictor;

public static class JuliaStudyImport
{
    public const string FrozenStatesSha256 = "c00636dc21f7a4e3bb099c508fbf24a407af546851e060c5852cb9aca54639fa";
    public const string FrozenSplitSha256 = "1f5967c9e6d228fda4322845a77ee1464dff6ec87ea578abafa16f679127cf6f";
    public const string PortableSourceCommit = "65880cdfd33580d23d0e833ac977f79acb18b1b0";
    public const string FullQualificationSha256 = "d6be7ed9322d869155cf833d61cff58da6a238329617a0a0b7b78794b1543847";
    public const string SelectedProfileSha256 = "d03cbcf0f230275356ba8adea870d7f34f0dfe8e2f76520e19e676e7209cf2ec";
    public const string FrozenPreparedSha256 = "8be3531e3196a9bf23658acec302244244429e6e36ddc040ef6392a0810e235c";
    private const string HandoffStatus = "FULL-EXPORT-VERIFIED-PUBLIC-PACKAGE-READER";

    private sealed record AuthorizedDataset(string Manifest, string Contract, ExtractionCost Extraction);

    public static async Task<StudyData> ImportAsync(string artifactRoot, string handoffPath, string handoffSha256,
        string preparationPath, string splitPath, string statesPath, string questionsPath,
        long numericCapBytes = ProbabilityStore.DefaultNumericCapBytes, CancellationToken token = default)
    {
        var authorized = ReadAuthorization(artifactRoot, handoffPath, handoffSha256);
        var metadata = ReadFullMetadata(preparationPath, splitPath, statesPath, questionsPath);
        return await ImportVerifiedAsync(authorized.Manifest, authorized.Contract, questionsPath, metadata,
            FrozenStatesSha256, FrozenSplitSha256, authorized.Extraction, numericCapBytes, token);
    }

    // Explicit POCO oracle only; the compact importer never projects through per-row observations.
    public static async Task<ImportedStudy> ImportLegacyOracleAsync(string artifactRoot, string handoffPath,
        string handoffSha256, string preparationPath, string splitPath, string statesPath, string questionsPath,
        CancellationToken token = default, string? originalReferenceDirectory = null)
    {
        var authorized = ReadAuthorization(artifactRoot, handoffPath, handoffSha256);
        var metadata = ReadFullMetadata(preparationPath, splitPath, statesPath, questionsPath);
        var contract = ArrowFeatureReader.ExpectedContract(authorized.Contract,
            JuliaFixtureInterop.HighPrecisionFingerprint, questionsPath);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(authorized.Manifest, contract,
            Enumerable.Range(0, metadata.Count).Select(i => metadata[i].RowId), token);
        RequireProvenance(reader, metadata.Count, FrozenStatesSha256);
        using var oracle = originalReferenceDirectory is null ? null : new OriginalConsumerOracle(originalReferenceDirectory);
        var rows = new LearningRow[metadata.Count];
        int count = 0;
        RecordBatch? batch;
        while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
        {
            using (batch)
            {
                RequireJuliaNativeNull(batch);
                foreach (var projected in oracle is null ? ArrowFeatureReader.Project(batch) :
                    OriginalConsumerOracle.Observations(oracle.Project(batch)))
                {
                    if (count >= metadata.Count || projected.RowId != metadata[count].RowId)
                        throw new InvalidDataException("Julia POCO oracle source order/count differs from frozen original IDs.");
                    var original = metadata[count];
                    rows[count++] = new LearningRow
                    {
                        RowId = original.RowId, GroupId = original.GroupId, Label = original.Label,
                        Text = original.Text, Semantic = projected.Semantic, SpamBaseline = projected.SpamBaseline
                    };
                }
            }
        }
        if (count != metadata.Count) throw new InvalidDataException("Julia POCO oracle full EOS/count gate failed.");
        token.ThrowIfCancellationRequested();
        var study = new ImportedStudy(contract.FeatureFingerprint, ArtifactFiles.Hash(authorized.Manifest),
            FrozenSplitSha256, FeatureContract.QuestionsV1Sha256, rows, ArtifactFiles.Read<SplitManifest>(splitPath),
            authorized.Extraction);
        StudyWorkflow.Validate(study);
        return study;
    }

    public static void RequireLegacyParity(ImportedStudy legacy, StudyData compact)
    {
        compact.RequireOpen();
        if (legacy.Rows.Length != compact.Metadata.Count || legacy.FeatureFingerprint != compact.FeatureFingerprint ||
            legacy.DatasetManifestSha256 != compact.DatasetManifestSha256 ||
            legacy.SplitSha256 != compact.SplitSha256 || legacy.QuestionsSha256 != compact.QuestionsSha256)
            throw new InvalidDataException("Julia original/compact import identities or full counts differ.");
        var partitions = legacy.Split.Rows.ToDictionary(row => row.RowId, row => row.Split);
        Span<float> values = stackalloc float[FeatureContract.Width];
        for (int i = 0; i < legacy.Rows.Length; i++)
        {
            var row = legacy.Rows[i];
            var metadata = compact.Metadata[i];
            compact.Probabilities.CopySemantic(i, values);
            if (row.RowId != metadata.RowId || row.GroupId != metadata.GroupId || row.Label != metadata.Label ||
                !string.Equals(row.Text, metadata.Text, StringComparison.Ordinal) ||
                !partitions.TryGetValue(row.RowId, out var partition) || partition != metadata.Partition ||
                row.Semantic.Length != FeatureContract.Width ||
                BitConverter.DoubleToInt64Bits(row.SpamBaseline) !=
                    BitConverter.DoubleToInt64Bits(compact.Probabilities.Direct(i)))
                throw new InvalidDataException("Julia original/compact source/text/group/label/direct bits differ.");
            for (int column = 0; column < values.Length; column++)
                if (BitConverter.SingleToInt32Bits(row.Semantic[column]) != BitConverter.SingleToInt32Bits(values[column]))
                    throw new InvalidDataException("Julia original/compact ten feature bits/order differ.");
        }
    }

    private static StudyMetadata ReadFullMetadata(string preparationPath, string splitPath, string statesPath,
        string questionsPath)
    {
        ArtifactFiles.RequireHash(statesPath, FrozenStatesSha256);
        ArtifactFiles.RequireHash(splitPath, FrozenSplitSha256);
        var (metadata, preparation) = ArrowFeatureReader.ReadFrozenMetadata(
            preparationPath, splitPath, statesPath, questionsPath);
        if (metadata.Count != 5574 ||
            preparation.CorpusSha256 != "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d" ||
            Enumerable.Range(0, metadata.Count).Any(i => metadata[i].RowId != i + 1L) ||
            Enumerable.Range(0, metadata.Count).Select(i => metadata[i].GroupId).Distinct().Count() != 5102 ||
            Enumerable.Range(0, metadata.Count).Count(i => metadata[i].Partition == "train") != 3344 ||
            Enumerable.Range(0, metadata.Count).Count(i => metadata[i].Partition == "validation") != 1116 ||
            Enumerable.Range(0, metadata.Count).Count(i => metadata[i].Partition == "holdout") != 1114)
            throw new InvalidDataException("Full Julia study requires the unchanged5574/5102 source and3344/1116/1114 grouped split.");
        return metadata;
    }

    private static AuthorizedDataset ReadAuthorization(string artifactRoot, string path, string expectedHash)
    {
        try { return ReadAuthorizationCore(artifactRoot, path, expectedHash); }
        catch (Exception error) when (error is KeyNotFoundException or InvalidOperationException or FormatException)
        {
            throw new InvalidDataException("Full Julia handoff is missing or has invalid required schema fields.", error);
        }
    }

    private static AuthorizedDataset ReadAuthorizationCore(string artifactRoot, string path, string expectedHash)
    {
        ArtifactFiles.RequireHash(path, expectedHash);
        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = document.RootElement;
        RequireHandoffHeader(root);
        var artifacts = root.GetProperty("artifacts");
        var verified = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string name in new[] { "manifest", "contract", "data", "runtime", "preflight", "prepared",
            "selectedProfile", "qualificationReceipt", "priorQualificationReceipt", "portableReceipt",
            "ownerAuthorization", "publicReaderReceipt" })
        {
            verified.Add(name, VerifyArtifact(artifactRoot, artifacts.GetProperty(name), name));
        }
        string dataset = Path.GetFullPath(root.GetProperty("datasetDirectory").GetString() ??
            throw new InvalidDataException("Full Julia dataset directory is missing."));
        foreach (var (name, filename) in new[] { ("manifest", "manifest.json"), ("contract", "contract.json"),
            ("data", "decisions.arrow"), ("runtime", "runtime.receipt.json") })
            if (!string.Equals(verified[name], Path.Combine(dataset, filename), StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Full Julia handoff mixes dataset artifact directories.");
        ArtifactFiles.RequireHash(verified["contract"], JuliaFixtureInterop.HighPrecisionFingerprint);
        ArtifactFiles.RequireHash(verified["qualificationReceipt"], FullQualificationSha256);
        ArtifactFiles.RequireHash(verified["priorQualificationReceipt"], JuliaBoundedControlImport.QualificationSha256);
        ArtifactFiles.RequireHash(verified["selectedProfile"], SelectedProfileSha256);
        ArtifactFiles.RequireHash(verified["portableReceipt"],
            "1baa85ce0b1be99d306713d8a19f287b6a8a599bcf5726169c965f842d94b03c");
        ArtifactFiles.RequireHash(verified["prepared"], FrozenPreparedSha256);
        if (artifacts.TryGetProperty("originalPrepared", out var originalPrepared))
            ArtifactFiles.RequireHash(VerifyArtifact(artifactRoot, originalPrepared, "originalPrepared"),
                FrozenPreparedSha256);
        using var runtimeDocument = JsonDocument.Parse(File.ReadAllBytes(verified["runtime"]));
        return new(verified["manifest"], verified["contract"], ReadRuntimeCost(runtimeDocument.RootElement));
    }

    internal static void RequireHandoffHeader(JsonElement root)
    {
        try { RequireHandoffHeaderCore(root); }
        catch (Exception error) when (error is KeyNotFoundException or InvalidOperationException or FormatException)
        {
            throw new InvalidDataException("Full Julia handoff header is missing or has invalid required schema fields.", error);
        }
    }

    private static void RequireHandoffHeaderCore(JsonElement root)
    {
        var sources = root.GetProperty("acceptedSources");
        var ids = root.GetProperty("expectedIds");
        if (root.GetProperty("schemaVersion").GetInt32() != 1 ||
            root.GetProperty("status").GetString() != HandoffStatus || root.GetProperty("rowCount").GetInt32() != 5574 ||
            root.GetProperty("preparedTensorCount").GetInt32() != 5574 ||
            !root.GetProperty("preparedMatchesFrozenOriginalBytes").GetBoolean() ||
            root.GetProperty("featureFingerprint").GetString() != JuliaFixtureInterop.HighPrecisionFingerprint ||
            sources.GetProperty("runtimeSourceCommit").GetString() != JuliaBoundedControlImport.RuntimeSourceCommit ||
            sources.GetProperty("portableSourceCommit").GetString() != PortableSourceCommit ||
            sources.GetProperty("originalSourceCommit").GetString() != "6dda59d19a60fdad2fc14152e3774d1d5ad9b345" ||
            ids.GetProperty("first").GetInt64() != 1 || ids.GetProperty("last").GetInt64() != 5574 ||
            ids.GetProperty("count").GetInt32() != 5574 || ids.GetProperty("order").GetString() != "exact")
            throw new InvalidDataException("A completed independently pinned full5574 CPU Julia handoff is required.");
    }

    internal static string VerifyArtifact(string artifactRoot, JsonElement artifact, string name)
    {
        try
        {
            string resolved = ResolveArtifact(artifactRoot, artifact.GetProperty("path").GetString() ??
                throw new InvalidDataException("Handoff artifact path is missing."));
            long size = artifact.GetProperty("sizeBytes").GetInt64();
            if (size <= 0 || new FileInfo(resolved).Length != size)
                throw new InvalidDataException($"Full Julia handoff artifact size mismatch: {name}.");
            ArtifactFiles.RequireHash(resolved, artifact.GetProperty("sha256").GetString() ?? "");
            return resolved;
        }
        catch (Exception error) when (error is KeyNotFoundException or InvalidOperationException or FormatException)
        {
            throw new InvalidDataException($"Full Julia artifact has invalid required fields: {name}.", error);
        }
    }

    internal static string ResolveArtifact(string artifactRoot, string relative)
    {
        if (string.IsNullOrWhiteSpace(relative) || Path.IsPathRooted(relative))
            throw new InvalidDataException("Full Julia handoff artifact paths must be relative to the supplied namespace.");
        string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(artifactRoot));
        string resolved = Path.GetFullPath(Path.Combine(root, relative));
        if (!resolved.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Full Julia handoff artifact path escapes the supplied namespace.");
        return resolved;
    }

    internal static ExtractionCost ReadRuntimeCost(JsonElement root)
    {
        try { return ReadRuntimeCostCore(root); }
        catch (Exception error) when (error is KeyNotFoundException or InvalidOperationException or FormatException)
        {
            throw new InvalidDataException("Full Julia runtime receipt is missing or has invalid required schema fields.", error);
        }
    }

    private static ExtractionCost ReadRuntimeCostCore(JsonElement root)
    {
        var profile = root.GetProperty("profile");
        var runtime = root.GetProperty("runtime");
        if (root.GetProperty("schemaVersion").GetInt32() != 1 || !root.GetProperty("full").GetBoolean() ||
            root.GetProperty("diagnostic").GetBoolean() || root.GetProperty("rowCount").GetInt32() != 5574 ||
            root.GetProperty("inputHash").GetString() != FrozenStatesSha256 ||
            root.GetProperty("featureFingerprint").GetString() != JuliaFixtureInterop.HighPrecisionFingerprint ||
            profile.GetProperty("mode").GetString() != "scalar-cpu" ||
            profile.GetProperty("status").GetString() != "SELECTED" ||
            profile.GetProperty("qualificationReceiptSha256").GetString() != FullQualificationSha256 ||
            profile.GetProperty("intraThreads").GetInt32() != 4 || profile.GetProperty("interThreads").GetInt32() != 1 ||
            runtime.GetProperty("mode").GetString() != "scalar-cpu" ||
            runtime.GetProperty("arithmetic").GetString() != "ort1.23.2-cpu-fp32-intra4-inter1-default-sync-v1" ||
            runtime.GetProperty("tf32").GetBoolean() || runtime.GetProperty("deviceUuid").ValueKind != JsonValueKind.Null ||
            runtime.GetProperty("deviceIndex").ValueKind != JsonValueKind.Null ||
            runtime.GetProperty("providerOptions").ValueKind != JsonValueKind.Null ||
            runtime.GetProperty("nativeOrt").GetString() != "1.23.2" ||
            root.GetProperty("precisionTolerance").GetDouble() != 1e-6)
            throw new InvalidDataException("Full Julia runtime must match the qualified CPU-only profile and unchanged tolerance.");
        var measures = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (string name in new[] { "loadMilliseconds", "warmupMilliseconds", "extractionAndValidationMilliseconds" })
        {
            double value = root.GetProperty(name).GetDouble();
            if (!double.IsFinite(value) || value < 0)
                throw new InvalidDataException($"Invalid full Julia runtime measurement: {name}.");
            measures.Add(name, value);
        }
        return new("scalar-cpu", 5574, measures,
            "Producer runtime receipt: load and warmup are separate; extractionAndValidationMilliseconds includes " +
            "preparation/synchronized Run and transfers/Double decode/builder/IPC/full validation. " +
            "Excludes runtime receipt/module hashing; manifest writer stages overlap and must not be summed.");
    }

    internal static async Task<StudyData> ImportVerifiedAsync(string manifestPath, string contractPath, string questionsPath,
        StudyMetadata metadata, string expectedInputSha256, string splitSha256, ExtractionCost extraction,
        long numericCapBytes = ProbabilityStore.DefaultNumericCapBytes, CancellationToken token = default)
    {
        if (extraction.ExecutionMode != "scalar-cpu" || extraction.Rows != metadata.Count)
            throw new InvalidDataException("Julia extraction receipt count/mode differs from the imported source.");
        var contract = ArrowFeatureReader.ExpectedContract(contractPath,
            JuliaFixtureInterop.HighPrecisionFingerprint, questionsPath);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(manifestPath, contract,
            Enumerable.Range(0, metadata.Count).Select(i => metadata[i].RowId), token);
        RequireProvenance(reader, metadata.Count, expectedInputSha256);
        var store = new ProbabilityStore(metadata.Count, numericCapBytes);
        try
        {
            int count = 0;
            RecordBatch? batch;
            while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
            {
                using (batch)
                {
                    RequireJuliaNativeNull(batch);
                    var accessor = new ProbabilityBatchAccessor(batch);
                    for (int row = 0; row < accessor.Count; row++)
                    {
                        int ordinal = checked(count + row);
                        if (ordinal >= metadata.Count || accessor.RowId(row) != metadata[ordinal].RowId)
                            throw new InvalidDataException("Full Julia IDs/order differ from independently supplied original source.");
                        store.SetDirect(ordinal, accessor.CopyRow(row, store.WritableSemantic(ordinal)));
                    }
                    count = checked(count + accessor.Count);
                }
            }
            if (count != metadata.Count || reader.Manifest.RowCount != count)
                throw new InvalidDataException("Full Julia public-reader EOS/count gate failed.");
            token.ThrowIfCancellationRequested();
            return new(metadata, store, contract.FeatureFingerprint, ArtifactFiles.Hash(manifestPath), splitSha256,
                FeatureContract.QuestionsV1Sha256, extraction);
        }
        catch
        {
            store.Dispose();
            throw;
        }
    }

    internal static void RequireJuliaNativeNull(RecordBatch batch)
    {
        if (batch.Column(3) is not StructArray pressure || pressure.Fields.Count != 5 ||
            pressure.Fields[3] is not DoubleArray native)
            throw new InvalidDataException("Julia HighPrecision score native values must remain absent/null.");
        // Fields already applies the parent Struct offset in the pinned Arrow API.
        if (native.Length != batch.Length)
            throw new InvalidDataException("Julia native score child bounds differ from the batch.");
        if (native.NullCount == native.Length) return;
        for (int row = 0; row < batch.Length; row++)
            if (!native.IsNull(row))
                throw new InvalidDataException("Julia HighPrecision score native values must remain absent/null.");
    }

    private static void RequireProvenance(DecisionArrowDatasetReader reader, int count, string inputSha256)
    {
        var source = reader.Manifest.Provenance;
        if (reader.Manifest.RowCount != count || reader.Manifest.OutputBatchSize != 256 ||
            source.SourceRepository != "luisquintanilla/typesafe-meai" ||
            source.SourceCommit != JuliaBoundedControlImport.RuntimeSourceCommit ||
            source.ExecutionMode != "scalar-cpu" || source.InputSha256 != inputSha256 ||
            source.QuestionsSha256 != FeatureContract.QuestionsV1Sha256 ||
            source.NativeStateBatchSize != 1 || source.CpuThreads != 4)
            throw new InvalidDataException("Full Julia manifest source/input/mode/count must match the frozen scalar-cpu profile.");
    }
}
