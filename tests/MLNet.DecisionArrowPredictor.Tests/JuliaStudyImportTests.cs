using System.Text.Json;
using System.Text.Json.Nodes;
using Apache.Arrow;
using DecisionInference.Arrow;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class JuliaStudyImportTests
{
    private static string Questions => Path.Combine(AppContext.BaseDirectory, "questions.v1.json");
    private const string InputHash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private const string SplitHash = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    private static ExtractionCost Cost => new("scalar-cpu", 257,
        new Dictionary<string, double> { ["extractionAndValidationMilliseconds"] = 17 }, "Authored; no inference.");

    [TestMethod]
    public void RequireHandoffHeader_AcceptsOnlyDeclaredFullCpuSourceAndExactOriginalIds()
    {
        using var document = JsonDocument.Parse(Header().ToJsonString());
        JuliaStudyImport.RequireHandoffHeader(document.RootElement);
        Assert.AreEqual(5574, document.RootElement.GetProperty("expectedIds").GetProperty("count").GetInt32());
    }

    [TestMethod]
    [DataRow("bounded")]
    [DataRow("rows")]
    [DataRow("runtime")]
    [DataRow("package")]
    [DataRow("original")]
    [DataRow("fingerprint")]
    [DataRow("ids")]
    [DataRow("order")]
    [DataRow("prepared-count")]
    [DataRow("prepared-bytes")]
    [DataRow("missing")]
    public void RequireHandoffHeader_RejectsControlOrMixedSourceIdentityAndMissingFields(string mutation)
    {
        var header = Header();
        switch (mutation)
        {
            case "bounded": header["status"] = "BOUNDED_JULIA_CPU128_IMPORT_PARITY_PASS"; break;
            case "rows": header["rowCount"] = 128; break;
            case "runtime": header["acceptedSources"]!["runtimeSourceCommit"] = new string('a', 40); break;
            case "package": header["acceptedSources"]!["portableSourceCommit"] = new string('a', 40); break;
            case "original": header["acceptedSources"]!["originalSourceCommit"] = new string('a', 40); break;
            case "fingerprint": header["featureFingerprint"] = ArrowFeatureReader.LegacyLayaFingerprint; break;
            case "ids": header["expectedIds"]!["first"] = 2; break;
            case "order": header["expectedIds"]!["order"] = "set-only"; break;
            case "prepared-count": header["preparedTensorCount"] = 128; break;
            case "prepared-bytes": header["preparedMatchesFrozenOriginalBytes"] = false; break;
            case "missing": header.Remove("acceptedSources"); break;
        }
        using var document = JsonDocument.Parse(header.ToJsonString());
        Assert.ThrowsExactly<InvalidDataException>(() => JuliaStudyImport.RequireHandoffHeader(document.RootElement));
    }

    [TestMethod]
    public void ReadRuntimeCost_FullQualifiedCpuPreservesIndependentLoadWarmupAndExtractionScopes()
    {
        using var document = JsonDocument.Parse(Runtime().ToJsonString());
        var cost = JuliaStudyImport.ReadRuntimeCost(document.RootElement);
        Assert.AreEqual("scalar-cpu", cost.ExecutionMode);
        Assert.AreEqual(5574, cost.Rows);
        Assert.AreEqual(3, cost.Measurements.Count);
        Assert.AreEqual(123d, cost.Measurements["loadMilliseconds"]);
        Assert.AreEqual(456d, cost.Measurements["warmupMilliseconds"]);
        Assert.AreEqual(789d, cost.Measurements["extractionAndValidationMilliseconds"]);
        Assert.IsFalse(cost.Measurements.ContainsKey("exportBeforeManifestPublicationMilliseconds"),
            "Do not invent a manifest writer measurement or combine overlapping producer scopes.");
        StringAssert.Contains(cost.Scope, "must not be summed");
    }

    [TestMethod]
    [DataRow("bounded")]
    [DataRow("diagnostic")]
    [DataRow("rows")]
    [DataRow("input")]
    [DataRow("fingerprint")]
    [DataRow("mode")]
    [DataRow("not-go")]
    [DataRow("intra")]
    [DataRow("inter")]
    [DataRow("runtime-mode")]
    [DataRow("arithmetic")]
    [DataRow("tf32")]
    [DataRow("device")]
    [DataRow("provider")]
    [DataRow("ort")]
    [DataRow("tolerance")]
    [DataRow("cost")]
    [DataRow("missing")]
    public void ReadRuntimeCost_RejectsBoundedMixedOrChangedCpuProfileAndInvalidCosts(string mutation)
    {
        var runtime = Runtime();
        switch (mutation)
        {
            case "bounded": runtime["full"] = false; break;
            case "diagnostic": runtime["diagnostic"] = true; break;
            case "rows": runtime["rowCount"] = 128; break;
            case "input": runtime["inputHash"] = InputHash; break;
            case "fingerprint": runtime["featureFingerprint"] = ArrowFeatureReader.LegacyLayaFingerprint; break;
            case "mode": runtime["profile"]!["mode"] = "scalar-cuda"; break;
            case "not-go": runtime["profile"]!["status"] = "QUALIFIED"; break;
            case "intra": runtime["profile"]!["intraThreads"] = 1; break;
            case "inter": runtime["profile"]!["interThreads"] = 2; break;
            case "runtime-mode": runtime["runtime"]!["mode"] = "scalar-cuda"; break;
            case "arithmetic": runtime["runtime"]!["arithmetic"] = "different"; break;
            case "tf32": runtime["runtime"]!["tf32"] = true; break;
            case "device": runtime["runtime"]!["deviceUuid"] = "GPU"; break;
            case "provider": runtime["runtime"]!["providerOptions"] = new JsonObject(); break;
            case "ort": runtime["runtime"]!["nativeOrt"] = "1.24"; break;
            case "tolerance": runtime["precisionTolerance"] = 1e-4; break;
            case "cost": runtime["loadMilliseconds"] = -1d; break;
            case "missing": runtime.Remove("loadMilliseconds"); break;
        }
        using var document = JsonDocument.Parse(runtime.ToJsonString());
        Assert.ThrowsExactly<InvalidDataException>(() => JuliaStudyImport.ReadRuntimeCost(document.RootElement));
    }

    [TestMethod]
    [DataRow("")]
    [DataRow("..\\outside.json")]
    [DataRow("nested\\..\\..\\outside.json")]
    [DataRow("C:\\outside.json")]
    public void ResolveArtifact_RejectsMissingRootedAndEscapingNamespacePaths(string relative)
    {
        using var temp = new TempDirectory();
        Assert.ThrowsExactly<InvalidDataException>(() => JuliaStudyImport.ResolveArtifact(
            temp.FilePath("namespace"), relative));
    }

    [TestMethod]
    public void ResolveArtifact_RetainsExactRelativeArtifactWithinNamespace()
    {
        using var temp = new TempDirectory();
        string root = temp.FilePath("namespace");
        Assert.AreEqual(Path.GetFullPath(Path.Combine(root, "dataset", "manifest.json")),
            JuliaStudyImport.ResolveArtifact(root, "dataset\\manifest.json"));
    }

    [TestMethod]
    public void VerifyArtifact_RequiresExactNamespaceBytesSizeAndIndependentlyDeclaredHash()
    {
        using var temp = new TempDirectory();
        string path = temp.FilePath("artifact.json");
        File.WriteAllBytes(path, [1, 2, 3]);
        using var document = JsonDocument.Parse(new JsonObject
        {
            ["path"] = "artifact.json", ["sizeBytes"] = 3,
            ["sha256"] = ArtifactExpectations.HashFile(path)
        }.ToJsonString());
        Assert.AreEqual(path, JuliaStudyImport.VerifyArtifact(Path.GetDirectoryName(path)!,
            document.RootElement, "authored"));
    }

    [TestMethod]
    [DataRow("size")]
    [DataRow("hash")]
    [DataRow("content")]
    [DataRow("missing-field")]
    [DataRow("escape")]
    [DataRow("zero")]
    public void VerifyArtifact_RejectsReplacedBytesSizeHashMissingFieldsAndEscapingPaths(string mutation)
    {
        using var temp = new TempDirectory();
        string path = temp.FilePath("artifact.json");
        File.WriteAllBytes(path, [1, 2, 3]);
        var artifact = new JsonObject
        {
            ["path"] = "artifact.json", ["sizeBytes"] = 3, ["sha256"] = ArtifactExpectations.HashFile(path)
        };
        switch (mutation)
        {
            case "size": artifact["sizeBytes"] = 4; break;
            case "hash": artifact["sha256"] = InputHash; break;
            case "content": File.WriteAllBytes(path, [3, 2, 1]); break;
            case "missing-field": artifact.Remove("sha256"); break;
            case "escape": artifact["path"] = "..\\outside.json"; break;
            case "zero": artifact["sizeBytes"] = 0; break;
        }
        using var document = JsonDocument.Parse(artifact.ToJsonString());
        Assert.ThrowsExactly<InvalidDataException>(() => JuliaStudyImport.VerifyArtifact(
            Path.GetDirectoryName(path)!, document.RootElement, "authored"));
    }

    [TestMethod]
    public async Task ImportVerified_PublicFullEosFinalShortBatchRetainsExactBitsMetadataAndZeroNativeLeases()
    {
        using var temp = new TempDirectory();
        var (manifest, contract, metadata) = await Fixture(temp);
        var extraction = Cost;
        using var study = await JuliaStudyImport.ImportVerifiedAsync(manifest, contract, Questions,
            metadata, InputHash, SplitHash, extraction, numericCapBytes: 257 * 48);
        Assert.AreSame(metadata, study.Metadata);
        Assert.AreSame(extraction, study.Extraction);
        Assert.AreEqual(JuliaFixtureInterop.HighPrecisionFingerprint, study.FeatureFingerprint);
        Assert.AreEqual(12336L, study.NumericCapacityBytes);
        Assert.AreEqual(12336L, study.NumericCapBytes);
        using (File.Open(Path.Combine(Path.GetDirectoryName(manifest)!, "decisions.arrow"),
            FileMode.Open, FileAccess.ReadWrite, FileShare.None)) { }
        var view = study.All().View();
        using var cursor = view.GetRowCursor(view.Schema);
        var id = cursor.GetGetter<long>(view.Schema["RowId"]);
        var semantic = cursor.GetGetter<VBuffer<float>>(view.Schema["Semantic"]);
        var baseline = cursor.GetGetter<double>(view.Schema["SpamBaseline"]);
        VBuffer<float> value = default;
        long source = 0; double direct = 0;
        int count = 0;
        float[] expected = [.42f, .42f, 0, 0, 1, 0, 0, 0, 0, 1];
        while (cursor.MoveNext())
        {
            id(ref source); semantic(ref value); baseline(ref direct);
            Assert.AreEqual(metadata[count].RowId, source);
            for (int column = 0; column < expected.Length; column++)
                Assert.AreEqual(BitConverter.SingleToInt32Bits(expected[column]),
                    BitConverter.SingleToInt32Bits(value.GetValues()[column]));
            Assert.AreEqual(BitConverter.DoubleToInt64Bits(.42), BitConverter.DoubleToInt64Bits(direct));
            count++;
        }
        Assert.AreEqual(257, count);
        Assert.AreEqual(0, study.ActiveCursors);
        Assert.AreEqual(85, study.Partition("holdout").Count,
            "Only the full-study importer accepts frozen holdout metadata; bounded importer remains rejecting.");
    }

    [TestMethod]
    [DataRow("scalar-cuda")]
    [DataRow("scalar")]
    [DataRow("native-cpu")]
    [DataRow("source")]
    [DataRow("input")]
    [DataRow("threads")]
    [DataRow("order")]
    [DataRow("missing")]
    [DataRow("extra")]
    [DataRow("native")]
    public async Task ImportVerified_RejectsModeSourceInputAndNonexactFullAssociation(string mutation)
    {
        using var temp = new TempDirectory();
        var (manifest, contract, metadata) = await Fixture(temp, mutation);
        await Assert.ThrowsAsync<InvalidDataException>(() => JuliaStudyImport.ImportVerifiedAsync(
            manifest, contract, Questions, metadata, InputHash, SplitHash, Cost));
    }

    [TestMethod]
    [DataRow("mode")]
    [DataRow("rows")]
    public async Task ImportVerified_RejectsExtractionReceiptCountOrModeAssociation(string mutation)
    {
        using var temp = new TempDirectory();
        var (manifest, contract, metadata) = await Fixture(temp);
        var cost = mutation == "mode" ? Cost with { ExecutionMode = "scalar-cuda" } : Cost with { Rows = 128 };
        await Assert.ThrowsExactlyAsync<InvalidDataException>(() => JuliaStudyImport.ImportVerifiedAsync(
            manifest, contract, Questions, metadata, InputHash, SplitHash, cost));
    }

    [TestMethod]
    [DataRow("null")]
    [DataRow("outside")]
    [DataRow("inside")]
    public async Task RequireJuliaNativeNull_NonzeroStructAndChildOffsetsValidateOnlyVisibleRows(string mutation)
    {
        string fixture = Path.Combine(AppContext.BaseDirectory, "fixtures", "HighPrecision");
        using var reader = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(fixture, "manifest.json"),
            DecisionArrowSchema.FromCanonicalJson(File.ReadAllText(Path.Combine(fixture, "contract.json"))));
        using var template = await reader.ReadNextRecordBatchAsync();
        Assert.IsNotNull(template);
        var pressure = (StructArray)template.Column(3);
        var builder = new DoubleArray.Builder().Append(2).Append(2);
        for (int row = 0; row < template.Length; row++)
            if (mutation == "outside" && row == 0 || mutation == "inside" && row == 3) builder.Append(2);
            else builder.AppendNull();
        using var prefixed = builder.Build();
        using var native = (DoubleArray)prefixed.Slice(2, template.Length);
        using var batch = new RecordBatch(template.Schema,
            [Clone(template.Column(0)), Clone(template.Column(1)), Clone(template.Column(2)),
                new StructArray(pressure.Data.DataType, template.Length,
                    [.. pressure.Fields.Take(3).Select(Clone), native, Clone(pressure.Fields[4])], ArrowBuffer.Empty, 0),
                Clone(template.Column(4)), Clone(template.Column(5))], template.Length);
        using var slice = batch.Slice(2, 3);
        Assert.AreEqual(2, ((StructArray)slice.Column(3)).Offset);
        Assert.AreEqual(2, native.Offset);
        if (mutation == "inside")
            Assert.ThrowsExactly<InvalidDataException>(() => JuliaStudyImport.RequireJuliaNativeNull(slice));
        else
            JuliaStudyImport.RequireJuliaNativeNull(slice);

        static IArrowArray Clone(IArrowArray array) => ArrowArrayFactory.BuildArray(
            array.Data.Clone(Apache.Arrow.Memory.MemoryAllocator.Default.Value));
    }

    [TestMethod]
    public async Task ImportVerified_CapCancellationAndRehashedTruncationCannotPublishReadyOrRetainFileLease()
    {
        using var temp = new TempDirectory();
        var (manifest, contract, metadata) = await Fixture(temp);
        await Assert.ThrowsExactlyAsync<InvalidDataException>(() => JuliaStudyImport.ImportVerifiedAsync(
            manifest, contract, Questions, metadata, InputHash, SplitHash, Cost, numericCapBytes: 12335));
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAsync<OperationCanceledException>(() => JuliaStudyImport.ImportVerifiedAsync(
            manifest, contract, Questions, metadata, InputHash, SplitHash, Cost, token: cancellation.Token));
        string data = Path.Combine(Path.GetDirectoryName(manifest)!, "decisions.arrow");
        byte[] bytes = File.ReadAllBytes(data);
        File.WriteAllBytes(data, bytes[..^32]);
        var changed = JsonNode.Parse(File.ReadAllText(manifest))!;
        changed["dataSize"] = new FileInfo(data).Length;
        changed["dataSha256"] = ArtifactExpectations.HashFile(data);
        File.WriteAllText(manifest, changed.ToJsonString());
        await Assert.ThrowsAsync<InvalidDataException>(() => JuliaStudyImport.ImportVerifiedAsync(
            manifest, contract, Questions, metadata, InputHash, SplitHash, Cost));
        using (File.Open(data, FileMode.Open, FileAccess.ReadWrite, FileShare.None)) { }
    }

    [TestMethod]
    [DataRow("text")]
    [DataRow("label")]
    [DataRow("group")]
    [DataRow("feature-bit")]
    [DataRow("direct-bit")]
    [DataRow("identity")]
    [DataRow("partition")]
    public void RequireLegacyParity_RejectsAnyOriginalAssociationOrNumericBitDrift(string mutation)
    {
        var legacy = StudyFixture.Authored();
        using var compact = StudyData.FromLegacy(legacy);
        JuliaStudyImport.RequireLegacyParity(legacy, compact);
        switch (mutation)
        {
            case "text": legacy.Rows[0].Text += " "; break;
            case "label": legacy.Rows[0].Label = !legacy.Rows[0].Label; break;
            case "group": legacy.Rows[0].GroupId++; break;
            case "feature-bit": legacy.Rows[0].Semantic[0] = MathF.BitIncrement(legacy.Rows[0].Semantic[0]); break;
            case "direct-bit": legacy.Rows[0].SpamBaseline = Math.BitIncrement(legacy.Rows[0].SpamBaseline); break;
            case "identity": legacy = legacy with { FeatureFingerprint = ArrowFeatureReader.LegacyLayaFingerprint }; break;
            case "partition":
                var assignments = legacy.Split.Rows.ToArray();
                assignments[0] = assignments[0] with { Split = "validation" };
                legacy = legacy with { Split = legacy.Split with { Rows = assignments } };
                break;
        }
        Assert.ThrowsExactly<InvalidDataException>(() => JuliaStudyImport.RequireLegacyParity(legacy, compact));
    }

    private static JsonObject Runtime() => new()
    {
        ["schemaVersion"] = 1, ["full"] = true, ["diagnostic"] = false, ["rowCount"] = 5574,
        ["inputHash"] = JuliaStudyImport.FrozenStatesSha256,
        ["featureFingerprint"] = JuliaFixtureInterop.HighPrecisionFingerprint,
        ["profile"] = new JsonObject
        {
            ["mode"] = "scalar-cpu", ["status"] = "SELECTED", ["intraThreads"] = 4, ["interThreads"] = 1,
            ["qualificationReceiptSha256"] = JuliaStudyImport.FullQualificationSha256
        },
        ["runtime"] = new JsonObject
        {
            ["mode"] = "scalar-cpu", ["arithmetic"] = "ort1.23.2-cpu-fp32-intra4-inter1-default-sync-v1",
            ["tf32"] = false, ["deviceIndex"] = null, ["deviceUuid"] = null, ["providerOptions"] = null,
            ["nativeOrt"] = "1.23.2"
        },
        ["precisionTolerance"] = 1e-6, ["loadMilliseconds"] = 123d, ["warmupMilliseconds"] = 456d,
        ["extractionAndValidationMilliseconds"] = 789d
    };

    private static JsonObject Header() => new()
    {
        ["schemaVersion"] = 1, ["status"] = "FULL-EXPORT-VERIFIED-PUBLIC-PACKAGE-READER",
        ["rowCount"] = 5574, ["featureFingerprint"] = JuliaFixtureInterop.HighPrecisionFingerprint,
        ["preparedTensorCount"] = 5574, ["preparedMatchesFrozenOriginalBytes"] = true,
        ["acceptedSources"] = new JsonObject
        {
            ["runtimeSourceCommit"] = JuliaBoundedControlImport.RuntimeSourceCommit,
            ["portableSourceCommit"] = JuliaStudyImport.PortableSourceCommit,
            ["originalSourceCommit"] = "6dda59d19a60fdad2fc14152e3774d1d5ad9b345"
        },
        ["expectedIds"] = new JsonObject { ["first"] = 1, ["last"] = 5574, ["count"] = 5574, ["order"] = "exact" }
    };

    private static async Task<(string Manifest, string Contract, StudyMetadata Metadata)> Fixture(
        TempDirectory temp, string? mutation = null)
    {
        string fixture = Path.Combine(AppContext.BaseDirectory, "fixtures", "HighPrecision");
        var canonical = JsonNode.Parse(File.ReadAllText(Path.Combine(fixture, "contract.json")))!;
        canonical["provider"] = "julia";
        canonical["modelId"] = "SupersonicLabs/Julia-1-ONNX";
        canonical["assetHashes"] = new JsonObject
        {
            ["model.onnx"] = "97141d0cfb1da6204e9f8f24d581af72eaeb82cda21149d83eaa6df7160fbcd9",
            ["model.onnx.data"] = "fd915be810d7ebfb80fb05a48dd33c9484d17ae1b6bcb9e1f544cbaaa913ded1",
            ["tokenizer.json"] = "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"
        };
        canonical["preprocessing"] = "julia-python-json-strict1024-head256-option48-min16-reserved0to4-pad8-v1";
        canonical["decoder"] = "julia-valid-fp32-logits-double-stable-full-softmax-firstargmax-binary-p1-zero-based-highprecision-native-null-v1;ort1.23.2-cpu-fp32-intra4-inter1-default-sync-v1";
        var contract = DecisionArrowSchema.FromCanonicalJson(canonical.ToJsonString());
        Assert.AreEqual(JuliaFixtureInterop.HighPrecisionFingerprint, contract.FeatureFingerprint);
        long[] ids = Enumerable.Range(0, 257).Select(i => 10000L - i * 7).ToArray();
        var metadata = new StudyMetadata(ids.Select((id, i) =>
            new StudyRowMetadata(id, id, i % 2 == 0, $"Authored {i}", SplitManifest.Names[i % 3])));
        if (mutation == "missing") metadata = new StudyMetadata(Enumerable.Range(0, 256).Select(i => metadata[i]));
        if (mutation == "extra") metadata = new StudyMetadata([
            .. Enumerable.Range(0, metadata.Count).Select(i => metadata[i]), new(99, 99, false, "Extra", "train")]);
        if (mutation == "order") (ids[0], ids[1]) = (ids[1], ids[0]);
        var provenance = new DecisionArrowProvenance
        {
            SourceRepository = "luisquintanilla/typesafe-meai",
            SourceCommit = mutation == "source" ? new string('c', 40) : JuliaBoundedControlImport.RuntimeSourceCommit,
            InputSha256 = mutation == "input" ? SplitHash : InputHash, QuestionsSha256 = FeatureContract.QuestionsV1Sha256,
            ExecutionMode = mutation is "scalar-cuda" or "scalar" or "native-cpu" ? mutation : "scalar-cpu",
            NativeStateBatchSize = 1, CpuThreads = mutation == "threads" ? 1 : 4,
            Machine = "Authored fixture", Runtime = "Offline; no inference", Packages = []
        };
        string output = temp.FilePath("authored-full");
        await DecisionArrowWriter.WriteBatchesAsync(output, contract, Batches(), provenance, 256);
        return (Path.Combine(output, "manifest.json"), Path.Combine(output, "contract.json"), metadata);

        async IAsyncEnumerable<RecordBatch> Batches()
        {
            using var reader = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(fixture, "manifest.json"),
                DecisionArrowSchema.FromCanonicalJson(File.ReadAllText(Path.Combine(fixture, "contract.json"))));
            int offset = 0;
            RecordBatch? template;
            while ((template = await reader.ReadNextRecordBatchAsync()) is not null)
            {
                using (template)
                {
                    var pressure = (StructArray)template.Column(3);
                    var native = new DoubleArray.Builder();
                    for (int row = 0; row < template.Length; row++)
                        if (mutation == "native") native.Append(2); else native.AppendNull();
                    var score = new StructArray(pressure.Data.DataType, template.Length,
                        [.. pressure.Fields.Take(3).Select(Clone), native.Build(), Clone(pressure.Fields[4])],
                        ArrowBuffer.Empty, 0);
                    var batch = new RecordBatch(contract.Schema,
                        [new Int64Array.Builder().AppendRange(ids.AsSpan(offset, template.Length).ToArray()).Build(),
                            Clone(template.Column(1)), Clone(template.Column(2)), score,
                            Clone(template.Column(4)), Clone(template.Column(5))], template.Length);
                    offset += template.Length;
                    yield return batch;
                }
            }

            static IArrowArray Clone(IArrowArray array) => ArrowArrayFactory.BuildArray(
                array.Data.Clone(Apache.Arrow.Memory.MemoryAllocator.Default.Value));
        }
    }
}
