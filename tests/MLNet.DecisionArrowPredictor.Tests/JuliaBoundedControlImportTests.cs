using System.Text.Json.Nodes;
using Apache.Arrow;
using DecisionInference.Arrow;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class JuliaBoundedControlImportTests
{
    private static string Questions => Path.Combine(AppContext.BaseDirectory, "questions.v1.json");
    private const string InputHash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private const string SplitHash = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    private static long[] Ids => [9, 2, 7];
    private static StudyMetadata Metadata => new([
        new(9, 9, true, "Authored first", "train"),
        new(2, 2, false, "Authored second", "validation"),
        new(7, 7, false, "Authored third", "train")
    ]);

    [TestMethod]
    public void Associate_IndependentOrderJoinsOriginalTextGroupLabelAndSharesStrings()
    {
        var frozen = Metadata;
        var states = new Dictionary<long, string>
        {
            [7] = new string(frozen[2].Text.AsSpan()),
            [9] = new string(frozen[0].Text.AsSpan()),
            [2] = new string(frozen[1].Text.AsSpan())
        };
        var selected = JuliaBoundedControlImport.Associate(frozen, [7, 2, 9], states);
        CollectionAssert.AreEqual(new long[] { 7, 2, 9 },
            Enumerable.Range(0, selected.Count).Select(i => selected[i].RowId).ToArray());
        for (int i = 0; i < selected.Count; i++)
        {
            var original = frozen[frozen.Ordinal(selected[i].RowId)];
            Assert.AreEqual(original, selected[i]);
            Assert.AreSame(original.Text, selected[i].Text);
        }
    }

    [TestMethod]
    [DataRow("duplicate")]
    [DataRow("extra-state")]
    [DataRow("missing-state")]
    [DataRow("changed-text")]
    [DataRow("holdout")]
    [DataRow("unknown-id")]
    [DataRow("empty")]
    [DataRow("oversize")]
    public void Associate_RejectsChangedSelectionTextMissingExtraIdsAndHoldout(string mutation)
    {
        var rows = Enumerable.Range(0, Metadata.Count).Select(i => Metadata[i]).ToArray();
        var ids = Ids;
        var states = rows.ToDictionary(row => row.RowId, row => row.Text);
        switch (mutation)
        {
            case "duplicate": ids = [9, 2, 9]; break;
            case "extra-state": states[999] = "Extra authored state"; break;
            case "missing-state": states.Remove(7); break;
            case "changed-text": states[2] += " "; break;
            case "holdout": rows[1] = rows[1] with { Partition = "holdout" }; break;
            case "unknown-id": ids = [9, 2, 999]; states.Remove(7); states[999] = "Unknown"; break;
            case "empty": ids = []; states.Clear(); break;
            case "oversize":
                ids = Enumerable.Range(0, 129).Select(i => (long)i).ToArray();
                states = ids.ToDictionary(id => id, _ => "Authored");
                break;
        }
        Assert.ThrowsExactly<InvalidDataException>(() =>
            JuliaBoundedControlImport.Associate(new StudyMetadata(rows), ids, states));
    }

    [TestMethod]
    public async Task ImportControl_PublicCompletedEosPreservesNonmonotonicAssociationsAndExactProjectionBits()
    {
        using var temp = new TempDirectory();
        var paths = await WriteFixture(temp);
        using var imported = await JuliaBoundedControlImport.ImportControlAsync(paths.Manifest, paths.Contract,
            Questions, Metadata, InputHash, SplitHash, numericCapBytes: 144);
        Assert.AreEqual(JuliaFixtureInterop.HighPrecisionFingerprint, imported.FeatureFingerprint);
        Assert.AreEqual(144L, imported.NumericCapacityBytes);
        Assert.AreEqual(SplitHash, imported.SplitSha256);
        var extraction = imported.Extraction;
        Assert.IsNotNull(extraction);
        Assert.AreEqual("scalar-cpu", extraction.ExecutionMode);
        Assert.IsFalse(extraction.Measurements.ContainsKey("loadMilliseconds"));
        var view = imported.All().View();
        using var cursor = view.GetRowCursor(view.Schema);
        var id = cursor.GetGetter<long>(view.Schema["RowId"]);
        var features = cursor.GetGetter<VBuffer<float>>(view.Schema["Semantic"]);
        var direct = cursor.GetGetter<double>(view.Schema["SpamBaseline"]);
        VBuffer<float> vector = default; long source = 0; double spam = 0;
        float[] golden = [.42f, .42f, 0, 0, 1, 0, 0, 0, 0, 1];
        int rank = 0;
        while (cursor.MoveNext())
        {
            id(ref source); features(ref vector); direct(ref spam);
            Assert.AreEqual(Ids[rank], source);
            Assert.AreEqual(Metadata[rank], imported.Metadata[rank]);
            CollectionAssert.AreEqual(golden.Select(BitConverter.SingleToInt32Bits).ToArray(),
                vector.GetValues().ToArray().Select(BitConverter.SingleToInt32Bits).ToArray());
            Assert.AreEqual(BitConverter.DoubleToInt64Bits(.42), BitConverter.DoubleToInt64Bits(spam));
            rank++;
        }
        Assert.AreEqual(3, rank);
        Assert.AreEqual(0, imported.ActiveCursors);
    }

    [TestMethod]
    [DataRow("scalar")]
    [DataRow("native-cpu")]
    [DataRow("scalar-cuda")]
    [DataRow("julia-offline-fixture-not-real")]
    [DataRow("unknown")]
    [DataRow("source")]
    [DataRow("input")]
    [DataRow("threads")]
    [DataRow("state-batch")]
    [DataRow("output-batch")]
    [DataRow("order")]
    public async Task ImportControl_RejectsMixedOrUndeclaredRuntimeSourceInputCountsAndOrder(string mutation)
    {
        using var temp = new TempDirectory();
        var paths = await WriteFixture(temp, mutation);
        await Assert.ThrowsExactlyAsync<InvalidDataException>(() =>
            JuliaBoundedControlImport.ImportControlAsync(paths.Manifest, paths.Contract,
                Questions, Metadata, InputHash, SplitHash));
    }

    [TestMethod]
    public async Task ImportControl_CapFailureAndCancellationNeverPublishAReadyStore()
    {
        using var temp = new TempDirectory();
        var paths = await WriteFixture(temp);
        var error = await Assert.ThrowsExactlyAsync<InvalidDataException>(() =>
            JuliaBoundedControlImport.ImportControlAsync(paths.Manifest, paths.Contract,
                Questions, Metadata, InputHash, SplitHash, numericCapBytes: 143));
        StringAssert.Contains(error.Message, "no unlimited fallback");
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            JuliaBoundedControlImport.ImportControlAsync(paths.Manifest, paths.Contract,
                Questions, Metadata, InputHash, SplitHash, token: cancellation.Token));
    }

    [TestMethod]
    public async Task ImportControl_CorruptOrIncompleteFileCannotPassFullPublicReaderGate()
    {
        using var temp = new TempDirectory();
        var paths = await WriteFixture(temp);
        string data = Path.Combine(Path.GetDirectoryName(paths.Manifest)!, "decisions.arrow");
        byte[] bytes = File.ReadAllBytes(data);
        File.WriteAllBytes(data, bytes[..^32]);
        await Assert.ThrowsExactlyAsync<InvalidDataException>(() =>
            JuliaBoundedControlImport.ImportControlAsync(paths.Manifest, paths.Contract,
                Questions, Metadata, InputHash, SplitHash));
    }

    [TestMethod]
    public async Task ImportControl_RehashedTruncatedBatchStillFailsFullEosBeforeReadyPublication()
    {
        using var temp = new TempDirectory();
        var paths = await WriteFixture(temp);
        string data = Path.Combine(Path.GetDirectoryName(paths.Manifest)!, "decisions.arrow");
        byte[] bytes = File.ReadAllBytes(data);
        File.WriteAllBytes(data, bytes[..^32]);
        var manifest = JsonNode.Parse(File.ReadAllText(paths.Manifest))!;
        manifest["dataSize"] = new FileInfo(data).Length;
        manifest["dataSha256"] = ArtifactExpectations.HashFile(data);
        File.WriteAllText(paths.Manifest, manifest.ToJsonString());
        await Assert.ThrowsAsync<InvalidDataException>(() =>
            JuliaBoundedControlImport.ImportControlAsync(paths.Manifest, paths.Contract,
                Questions, Metadata, InputHash, SplitHash));
    }

    private static async Task<(string Manifest, string Contract)> WriteFixture(TempDirectory temp, string? mutation = null)
    {
        string originalFixture = Path.Combine(AppContext.BaseDirectory, "fixtures", "HighPrecision");
        var canonical = JsonNode.Parse(File.ReadAllText(Path.Combine(originalFixture, "contract.json")))!;
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
        Assert.AreEqual(JuliaFixtureInterop.HighPrecisionFingerprint, contract.FeatureFingerprint,
            "The producer-published identity is an external golden, not an identity invented by the consumer.");
        using var reader = await DecisionArrowDatasetReader.OpenAsync(
            Path.Combine(originalFixture, "manifest.json"),
            DecisionArrowSchema.FromCanonicalJson(File.ReadAllText(Path.Combine(originalFixture, "contract.json"))));
        using var template = await reader.ReadNextRecordBatchAsync();
        Assert.IsNotNull(template);
        using var slice = template.Slice(0, 3);
        var batch = new RecordBatch(contract.Schema,
            [new Int64Array.Builder().AppendRange(mutation == "order" ? [7L, 2L, 9L] : Ids).Build(),
                .. slice.Arrays.Skip(1).Select(array => ArrowArrayFactory.BuildArray(
                    array.Data.Clone(Apache.Arrow.Memory.MemoryAllocator.Default.Value)))], 3);
        string mode = mutation is "scalar" or "native-cpu" or "scalar-cuda" or "julia-offline-fixture-not-real" or "unknown" ?
            mutation : "scalar-cpu";
        var provenance = new DecisionArrowProvenance
        {
            SourceRepository = "luisquintanilla/typesafe-meai",
            SourceCommit = mutation == "source" ? new string('c', 40) : JuliaBoundedControlImport.RuntimeSourceCommit,
            InputSha256 = mutation == "input" ? SplitHash : InputHash,
            QuestionsSha256 = FeatureContract.QuestionsV1Sha256,
            ExecutionMode = mode, NativeStateBatchSize = mutation == "state-batch" ? 2 : 1,
            CpuThreads = mutation == "threads" ? 1 : 4,
            Machine = "Authored offline test; no inference", Runtime = ".NET authored fixture", Packages = []
        };
        string directory = temp.FilePath("authored-control");
        await DecisionArrowWriter.WriteBatchesAsync(directory, contract, Batches(), provenance,
            mutation == "output-batch" ? 3 : 256);
        return (Path.Combine(directory, "manifest.json"), Path.Combine(directory, "contract.json"));

        async IAsyncEnumerable<RecordBatch> Batches()
        {
            await Task.CompletedTask;
            yield return batch;
        }
    }
}
