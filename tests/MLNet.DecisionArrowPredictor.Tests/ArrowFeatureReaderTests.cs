using System.Text.Json;
using System.Text.Json.Nodes;
using Apache.Arrow;
using Apache.Arrow.Ipc;
using DecisionInference.Arrow;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class ArrowFeatureReaderTests
{
    private static string Questions => Path.Combine(AppContext.BaseDirectory, "questions.v1.json");
    private static string Fixture(string precision, string name) =>
        Path.Combine(AppContext.BaseDirectory, "fixtures", precision, name);

    // External goldens published in the immutable producer fixtures, not a
    // second implementation of the producer's canonicalization/fingerprint.
    private static string Fingerprint(string precision) => precision switch
    {
        "HighPrecision" => "06654530d526e33db86ca9f59e937d79dda9d83f1447d675edb69856a9017513",
        "FourDecimalPlaces" => "66e0d62923ab097c24016eb711bd65d3d07a5f3a77fef8bec3b95e735fc8a2c1",
        "TwoDecimalPlaces" => "b717b6c1c8250773be6d5594395405a81c64bae64d416d1c029a5e8ec3ed41e4",
        _ => throw new ArgumentOutOfRangeException(nameof(precision))
    };

    private static DecisionArrowSchema Contract(string precision = "HighPrecision") =>
        DecisionArrowSchema.FromCanonicalJson(File.ReadAllText(Fixture(precision, "contract.json")));

    [TestMethod]
    [DataRow("HighPrecision")]
    [DataRow("FourDecimalPlaces")]
    [DataRow("TwoDecimalPlaces")]
    public void ExpectedContract_OfficialPrecisionsPreserveExternalIdentityAndExactQuestions(string precision)
    {
        byte[] before = File.ReadAllBytes(Fixture(precision, "contract.json"));

        var actual = ArrowFeatureReader.ExpectedContract(Fixture(precision, "contract.json"),
            Fingerprint(precision), Questions);

        Assert.AreEqual(Fingerprint(precision), actual.FeatureFingerprint);
        Assert.AreEqual(precision, actual.Precision.ToString());
        Assert.AreEqual(5, actual.Questions.Count);
        using var canonical = JsonDocument.Parse(actual.CanonicalJson);
        using var frozen = JsonDocument.Parse(File.ReadAllBytes(Questions));
        Assert.IsTrue(JsonElement.DeepEquals(frozen.RootElement.GetProperty("questions"),
            canonical.RootElement.GetProperty("questions")), "Every question string must match, not just its ID/kind.");
        CollectionAssert.AreEqual(QuestionFixtures.Projection, actual.Identity.FeatureProjection.ToArray());
        Assert.AreEqual("float64", canonical.RootElement.GetProperty("numericRepresentation").GetString());
        Assert.IsFalse(actual.Identity.FeatureProjection.Contains("spam_baseline"));
        ArtifactExpectations.Bytes(before, Fixture(precision, "contract.json"));
    }

    [TestMethod]
    [DataRow("empty")]
    [DataRow("blank")]
    [DataRow("uppercase")]
    [DataRow("padding")]
    [DataRow("other-precision")]
    public void ExpectedContract_FingerprintMustMatchExactly(string mutation)
    {
        string expected = mutation switch
        {
            "empty" => "",
            "blank" => " ",
            "uppercase" => Fingerprint("HighPrecision").ToUpperInvariant(),
            "padding" => Fingerprint("HighPrecision") + " ",
            "other-precision" => Fingerprint("FourDecimalPlaces"),
            _ => throw new ArgumentOutOfRangeException(nameof(mutation))
        };
        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            ArrowFeatureReader.ExpectedContract(Fixture("HighPrecision", "contract.json"), expected, Questions));
        Assert.AreEqual("External producer feature-contract identity mismatch.", error.Message);
        Assert.AreEqual(Fingerprint("HighPrecision"), Contract().FeatureFingerprint);
    }

    [TestMethod]
    [DataRow("instructions")]
    [DataRow("true-description")]
    [DataRow("false-description")]
    [DataRow("purpose-description")]
    [DataRow("rubric-string")]
    [DataRow("purpose-id")]
    [DataRow("projection-order")]
    public void ExpectedContract_SelfConsistentButChangedQuestionStringsOrProjectionAreRejected(string mutation)
    {
        using var temp = new TempDirectory();
        var changed = JsonNode.Parse(File.ReadAllText(Fixture("HighPrecision", "contract.json")))!;
        var questions = changed["questions"]!;
        switch (mutation)
        {
            case "instructions": questions[0]!["instructions"] = "Different instructions."; break;
            case "true-description": questions[1]!["trueDescription"] = "Different true description."; break;
            case "false-description": questions[4]!["falseDescription"] = "Different false description."; break;
            case "purpose-description": questions[3]!["candidates"]![2]!["description"] = "Different description."; break;
            case "rubric-string": questions[2]!["rubric"]![0] = " none"; break;
            case "purpose-id": questions[3]!["candidates"]![4]!["id"] = "alternative"; break;
            case "projection-order":
                changed["featureProjection"]![0] = "requested_contact_action";
                changed["featureProjection"]![1] = "commercial_solicitation";
                break;
        }
        // The package, not test code, computes a valid identity for the changed contract.
        var changedContract = DecisionArrowSchema.FromCanonicalJson(changed.ToJsonString());
        string path = temp.PutText("changed-contract.json", changedContract.CanonicalJson);
        Assert.AreNotEqual(Fingerprint("HighPrecision"), changedContract.FeatureFingerprint);

        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            ArrowFeatureReader.ExpectedContract(path, changedContract.FeatureFingerprint, Questions));

        Assert.AreEqual("Producer expected contract differs from exact frozen question strings/projection.", error.Message);
        Assert.AreEqual(Fingerprint("HighPrecision"), Contract().FeatureFingerprint);
    }

    [TestMethod]
    public void ExpectedContract_WhitespaceOnlyQuestionByteDriftIsRejected()
    {
        using var temp = new TempDirectory();
        string drifted = temp.Put("questions.json", [.. File.ReadAllBytes(Questions), (byte)' ']);
        Assert.AreNotEqual(FeatureContract.QuestionsV1Sha256, ArtifactExpectations.HashFile(drifted));
        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            ArrowFeatureReader.ExpectedContract(Fixture("HighPrecision", "contract.json"),
                Fingerprint("HighPrecision"), drifted));
        StringAssert.Contains(error.Message, "SHA-256 mismatch:");
        Assert.AreEqual(FeatureContract.QuestionsV1Sha256, ArtifactExpectations.HashFile(Questions));
    }

    [TestMethod]
    [DataRow("HighPrecision")]
    [DataRow("FourDecimalPlaces")]
    [DataRow("TwoDecimalPlaces")]
    public async Task SmokeAsync_OfficialSyntheticPrecisionsReadAll257RowsWithoutChangingArtifacts(string precision)
    {
        var snapshot = new[] { "contract.json", "manifest.json", "decisions.arrow" }
            .ToDictionary(name => name, name => File.ReadAllBytes(Fixture(precision, name)));

        int actual = await ArrowFeatureReader.SmokeAsync(Fixture(precision, "manifest.json"),
            Fixture(precision, "contract.json"), Fingerprint(precision), Questions);

        Assert.AreEqual(257, actual, "Includes the final one-row batch after the 256-row batch.");
        foreach (var (name, bytes) in snapshot) ArtifactExpectations.Bytes(bytes, Fixture(precision, name));
    }

    [TestMethod]
    [DataRow("HighPrecision", 0d)]
    [DataRow("FourDecimalPlaces", .0001d)]
    [DataRow("TwoDecimalPlaces", .01d)]
    public async Task Project_OfficialRoundedValuesAndSlicesStayUnchanged(string precision, double mild)
    {
        byte[] before = File.ReadAllBytes(Fixture(precision, "decisions.arrow"));
        double[] golden = [.42, .42, 0, mild, 1, 0, 0, 0, 0, 1];
        using var reader = await DecisionArrowDatasetReader.OpenAsync(Fixture(precision, "manifest.json"), Contract(precision));
        using var first = await reader.ReadNextRecordBatchAsync();
        Assert.IsNotNull(first);
        Assert.AreEqual(256, first.Length);

        var actual = ArrowFeatureReader.Project(first);
        Assert.AreEqual(256, actual.Length);
        for (int row = 0; row < first.Length; row++)
            AssertObservation(row + 1, golden, .42, actual[row], first, row);
        foreach (var (offset, count) in new[] { (1, 3), (255, 1), (256, 0) })
        {
            using var slice = first.Slice(offset, count);
            var projected = ArrowFeatureReader.Project(slice);
            Assert.AreEqual(count, projected.Length);
            for (int row = 0; row < count; row++)
                AssertObservation(offset + row + 1, golden, .42, projected[row], slice, row);
        }
        using var last = await reader.ReadNextRecordBatchAsync();
        Assert.IsNotNull(last);
        Assert.AreEqual(1, last.Length);
        AssertObservation(257, golden, .42, ArrowFeatureReader.Project(last).Single(), last, 0);
        Assert.IsNull(await reader.ReadNextRecordBatchAsync());
        // Rounded producer vectors deliberately need not sum to one. No renormalization.
        Assert.AreEqual(1 + mild, actual[0].Semantic.Skip(2).Take(3).Sum(x => (double)x),
            1e-8, "Preserve the producer's rounded probabilities.");
        ArtifactExpectations.Bytes(before, Fixture(precision, "decisions.arrow"));
    }

    [TestMethod]
    public async Task Project_DistinctTenCoordinatesExcludeSpamBaselineAndRespectNonzeroSliceOffsets()
    {
        using var reader = await DecisionArrowDatasetReader.OpenAsync(Fixture("HighPrecision", "manifest.json"), Contract());
        using var template = await reader.ReadNextRecordBatchAsync();
        Assert.IsNotNull(template);
        // Project consumes IPC reader batches in production. Preserve that
        // ownership model for the distinct-coordinate fixture as well.
        using var authored = DistinctBatch(template);
        using var memory = new MemoryStream();
        using (var writer = new ArrowStreamWriter(memory, authored.Schema, leaveOpen: true))
        {
            await writer.WriteStartAsync();
            await writer.WriteRecordBatchAsync(authored);
            await writer.WriteEndAsync();
        }
        memory.Position = 0;
        using var ipc = new ArrowStreamReader(memory, leaveOpen: true);
        using var batch = await ipc.ReadNextRecordBatchAsync();
        Assert.IsNotNull(batch);
        var original = Enumerable.Range(0, 4).Select(row => RawVector(batch, row)).ToArray();

        foreach (var (offset, count) in new[] { (0, 4), (1, 2), (3, 1), (4, 0) })
        {
            using var slice = batch.Slice(offset, count);
            var actual = ArrowFeatureReader.Project(slice);
            Assert.AreEqual(count, actual.Length);
            for (int row = 0; row < count; row++)
            {
                int index = offset + row;
                AssertObservation(1001 + index, DistinctVector(index), .91 - .01 * index,
                    actual[row], slice, row);
                Assert.IsFalse(actual[row].Semantic.Contains((float)actual[row].SpamBaseline),
                    "A distinct direct baseline must not enter any of the ten coordinates.");
            }
        }
        for (int row = 0; row < 4; row++) CollectionAssert.AreEqual(original[row], RawVector(batch, row));
    }

    [TestMethod]
    [DataRow(0d, true)]
    [DataRow(1d, true)]
    [DataRow(double.NaN, false)]
    [DataRow(double.PositiveInfinity, false)]
    [DataRow(double.NegativeInfinity, false)]
    [DataRow(-double.Epsilon, false)]
    [DataRow(1.0000000000000002d, false)]
    public async Task Project_DirectBaselineAcceptsEndpointsButRejectsNonfiniteOrOutOfRange(double value, bool valid)
    {
        using var reader = await DecisionArrowDatasetReader.OpenAsync(Fixture("HighPrecision", "manifest.json"), Contract());
        using var template = await reader.ReadNextRecordBatchAsync();
        Assert.IsNotNull(template);
        using var one = template.Slice(0, 1);
        using var authored = new RecordBatch(one.Schema, [.. one.Arrays.Take(5).Select(Clone), Doubles([value])], 1);
        using var memory = new MemoryStream();
        using (var writer = new ArrowStreamWriter(memory, authored.Schema, leaveOpen: true))
        {
            await writer.WriteStartAsync();
            await writer.WriteRecordBatchAsync(authored);
            await writer.WriteEndAsync();
        }
        memory.Position = 0;
        using var ipc = new ArrowStreamReader(memory, leaveOpen: true);
        using var batch = await ipc.ReadNextRecordBatchAsync();
        Assert.IsNotNull(batch);

        if (valid)
            AssertObservation(1, [.42, .42, 0, 0, 1, 0, 0, 0, 0, 1], value,
                ArrowFeatureReader.Project(batch).Single(), batch, 0);
        else
        {
            var error = Assert.ThrowsExactly<InvalidDataException>(() => ArrowFeatureReader.Project(batch));
            Assert.AreEqual("Invalid direct spam baseline probability.", error.Message);
        }
        Assert.AreEqual(BitConverter.DoubleToInt64Bits(value),
            BitConverter.DoubleToInt64Bits(((DoubleArray)batch.Column(5)).GetValue(0)!.Value),
            "Even rejected Arrow probabilities must not be clamped or rewritten.");
    }

    [TestMethod]
    [DataRow("partial")]
    [DataRow("failed")]
    [DataRow("wrong-hash")]
    [DataRow("altered-file")]
    [DataRow("incomplete-ipc")]
    [DataRow("row-count")]
    [DataRow("non-synthetic")]
    public async Task SmokeAsync_PublicReaderRejectsIncompleteOrChangedFixture(string mutation)
    {
        using var temp = new TempDirectory();
        string path = CopyFixture(temp, "HighPrecision");
        var manifest = ReadManifest(path);
        switch (mutation)
        {
            case "partial": manifest = manifest with { Status = "partial" }; break;
            case "failed": manifest = manifest with { Status = "failed" }; break;
            case "wrong-hash": manifest = manifest with { DataSha256 = new string('0', 64) }; break;
            case "altered-file":
                byte[] bytes = File.ReadAllBytes(temp.FilePath("decisions.arrow"));
                bytes[100] ^= 1;
                File.WriteAllBytes(temp.FilePath("decisions.arrow"), bytes);
                break;
            case "incomplete-ipc":
                byte[] truncated = File.ReadAllBytes(temp.FilePath("decisions.arrow"))[..^8];
                File.WriteAllBytes(temp.FilePath("decisions.arrow"), truncated);
                manifest = manifest with { DataSize = truncated.Length, DataSha256 = ArtifactExpectations.Hash(truncated) };
                break;
            case "row-count": manifest = manifest with { RowCount = 258 }; break;
            case "non-synthetic":
                manifest = manifest with { Provenance = manifest.Provenance with { ExecutionMode = "scalar" } };
                break;
        }
        WriteManifest(path, manifest);

        var error = await Assert.ThrowsExactlyAsync<InvalidDataException>(() =>
            ArrowFeatureReader.SmokeAsync(path, temp.FilePath("contract.json"), Fingerprint("HighPrecision"), Questions));

        Assert.IsFalse(string.IsNullOrWhiteSpace(error.Message), "Reject, rather than returning a partial row count.");
        Assert.AreEqual(257, ReadManifest(Fixture("HighPrecision", "manifest.json")).RowCount);
        Assert.AreEqual(257, await ArrowFeatureReader.SmokeAsync(Fixture("HighPrecision", "manifest.json"),
            Fixture("HighPrecision", "contract.json"), Fingerprint("HighPrecision"), Questions));
    }

    [TestMethod]
    [DataRow(256, "Unexpected source row ID 257.")]
    [DataRow(258, "Dataset row count or source row IDs mismatch.")]
    public async Task PublicReader_ExactExternalSourceIdsRejectExtraAndMissingRows(int expectedCount, string message)
    {
        var error = await Assert.ThrowsExactlyAsync<InvalidDataException>(async () =>
        {
            using var reader = await DecisionArrowDatasetReader.OpenAsync(Fixture("HighPrecision", "manifest.json"),
                Contract(), Enumerable.Range(1, expectedCount).Select(id => (long)id));
            await Drain(reader);
        });

        Assert.AreEqual(message, error.Message);
        using var valid = await DecisionArrowDatasetReader.OpenAsync(Fixture("HighPrecision", "manifest.json"),
            Contract(), Enumerable.Range(1, 257).Select(id => (long)id));
        CollectionAssert.AreEqual(Enumerable.Range(1, 257).Select(id => (long)id).ToArray(), await Drain(valid));
    }

    [TestMethod]
    [DataRow("within-batch")]
    [DataRow("across-batches")]
    public async Task PublicReader_HashConsistentIpcStillRejectsDuplicateSourceIds(string placement)
    {
        using var temp = new TempDirectory();
        string path = CopyFixture(temp, "HighPrecision");
        using var original = await DecisionArrowDatasetReader.OpenAsync(Fixture("HighPrecision", "manifest.json"), Contract());
        using var first = await original.ReadNextRecordBatchAsync();
        using var last = await original.ReadNextRecordBatchAsync();
        Assert.IsNotNull(first);
        Assert.IsNotNull(last);
        using var changedFirst = new RecordBatch(first.Schema,
        [
            new Int64Array.Builder().AppendRange(Enumerable.Range(1, 256)
                .Select(id => placement == "within-batch" && id == 2 ? 1L : id)).Build(),
            .. first.Arrays.Skip(1).Select(Clone)
        ], 256);
        using var changedLast = placement == "across-batches" ? first.Slice(0, 1) : last.Clone();
        string data = temp.FilePath("decisions.arrow");
        using (var output = File.Create(data))
        using (var writer = new ArrowStreamWriter(output, first.Schema))
        {
            await writer.WriteStartAsync();
            await writer.WriteRecordBatchAsync(changedFirst);
            await writer.WriteRecordBatchAsync(changedLast);
            await writer.WriteEndAsync();
        }
        var manifest = ReadManifest(path) with
        {
            DataSha256 = ArtifactExpectations.HashFile(data), DataSize = new FileInfo(data).Length
        };
        WriteManifest(path, manifest);
        Assert.AreEqual(257L, manifest.RowCount);
        Assert.AreEqual(manifest.DataSha256, ArtifactExpectations.HashFile(data));

        var error = await Assert.ThrowsExactlyAsync<InvalidDataException>(async () =>
        {
            using var reader = await DecisionArrowDatasetReader.OpenAsync(path, Contract(),
                Enumerable.Range(1, 257).Select(id => (long)id));
            await Drain(reader);
        });

        StringAssert.Contains(error.Message, "Duplicate");
        StringAssert.Contains(error.Message, "1");
        Assert.AreEqual(Fingerprint("HighPrecision"), manifest.FeatureFingerprint);
        Assert.AreEqual("complete", manifest.Status, "This is ID rejection, not a partial/hash failure.");
    }

    [TestMethod]
    [DataRow("scalar")]
    [DataRow("native")]
    [DataRow("scalar-cpu")]
    [DataRow("native-cpu")]
    public async Task ImportAsync_ExactCpuQualifiedModesAndAliasesPreserveAllLocalUnitRows(string mode)
    {
        using var temp = new TempDirectory();
        var input = ImportInputs(temp, "HighPrecision");
        var manifest = ReadManifest(input.Manifest);
        // Local synthetic UNIT metadata only: this is not an authoritative
        // export or a substitute for UCI. No model or inference is involved.
        WriteManifest(input.Manifest, manifest with
        {
            Provenance = manifest.Provenance with
            {
                ExecutionMode = mode,
                Measurements = new Dictionary<string, double>
                {
                    ["loadMilliseconds"] = 7,
                    ["exportBeforeManifestPublicationMilliseconds"] = 19
                }
            }
        });
        var snapshot = new[] { input.Manifest, input.PreparationPath, input.Split, input.States,
            temp.FilePath("decisions.arrow"), temp.FilePath("contract.json") }
            .ToDictionary(path => path, File.ReadAllBytes);

        var actual = await Import(input, "HighPrecision");

        Assert.AreEqual(257, actual.Rows.Length);
        CollectionAssert.AreEqual(Enumerable.Range(1, 257).Select(id => (long)id).ToArray(),
            actual.Rows.Select(row => row.RowId).ToArray());
        float[] semantic = [.42f, .42f, 0, 0, 1, 0, 0, 0, 0, 1];
        foreach (var row in actual.Rows)
        {
            Assert.AreEqual(row.RowId, row.GroupId);
            Assert.AreEqual(row.RowId % 2 == 0, row.Label);
            Assert.AreEqual($"unit-{row.RowId}", row.Text);
            CollectionAssert.AreEqual(semantic, row.Semantic);
            Assert.AreEqual(.42, row.SpamBaseline);
        }
        Assert.AreEqual(129, actual.Rows.Count(row => !row.Label));
        Assert.AreEqual(128, actual.Rows.Count(row => row.Label));
        Assert.AreEqual(Fingerprint("HighPrecision"), actual.FeatureFingerprint);
        Assert.AreEqual(ArtifactExpectations.HashFile(input.Manifest), actual.DatasetManifestSha256);
        Assert.AreEqual(input.Preparation.SplitSha256, actual.SplitSha256);
        Assert.AreEqual(FeatureContract.QuestionsV1Sha256, actual.QuestionsSha256);
        CollectionAssert.AreEqual(ArtifactFiles.Read<SplitManifest>(input.Split).Rows, actual.Split.Rows);
        Assert.IsNotNull(actual.Extraction);
        Assert.AreEqual(mode, actual.Extraction.ExecutionMode, "Preserve the exact producer spelling; no alias rewriting.");
        Assert.AreEqual(257, actual.Extraction.Rows);
        CollectionAssert.AreEquivalent(new[] { "loadMilliseconds", "exportBeforeManifestPublicationMilliseconds" },
            actual.Extraction.Measurements.Keys.ToArray());
        Assert.AreEqual(7d, actual.Extraction.Measurements["loadMilliseconds"]);
        Assert.AreEqual(19d, actual.Extraction.Measurements["exportBeforeManifestPublicationMilliseconds"]);
        foreach (var (path, bytes) in snapshot) ArtifactExpectations.Bytes(bytes, path);
    }

    [TestMethod]
    [DataRow("HighPrecision")]
    [DataRow("FourDecimalPlaces")]
    [DataRow("TwoDecimalPlaces")]
    public async Task ImportAsync_RealImportRejectsOfficialSyntheticFixtureEvenWhenSourceHashesAndIdsMatch(string precision)
    {
        using var temp = new TempDirectory();
        var input = ImportInputs(temp, precision);
        var manifest = ReadManifest(input.Manifest);
        Assert.AreEqual("synthetic", manifest.Provenance.ExecutionMode);
        Assert.AreEqual(input.Preparation.StatesSha256, manifest.Provenance.InputSha256);
        Assert.AreEqual(input.Preparation.QuestionsSha256, manifest.Provenance.QuestionsSha256);
        using (var reader = await DecisionArrowDatasetReader.OpenAsync(input.Manifest, Contract(precision),
            Enumerable.Range(1, 257).Select(id => (long)id)))
            CollectionAssert.AreEqual(Enumerable.Range(1, 257).Select(id => (long)id).ToArray(), await Drain(reader));
        byte[] statesBefore = File.ReadAllBytes(input.States);
        byte[] splitBefore = File.ReadAllBytes(input.Split);

        var error = await Assert.ThrowsExactlyAsync<InvalidDataException>(() => Import(input, precision));

        Assert.AreEqual("Expected matching completed real scalar/native export, not synthetic or changed input.", error.Message);
        ArtifactExpectations.Bytes(statesBefore, input.States);
        ArtifactExpectations.Bytes(splitBefore, input.Split);
        Assert.AreEqual(257, await ArrowFeatureReader.SmokeAsync(input.Manifest, temp.FilePath("contract.json"),
            Fingerprint(precision), Questions), "A smoke-valid fixture must still not become a real imported study.");
    }

    [TestMethod]
    [DataRow("input-hash")]
    [DataRow("questions-hash")]
    [DataRow("mode-uppercase")]
    [DataRow("mode-unknown")]
    [DataRow("mode-empty")]
    [DataRow("mode-null")]
    public async Task ImportAsync_PublicImporterRejectsWrongProvenance(string mutation)
    {
        using var temp = new TempDirectory();
        var input = ImportInputs(temp, "HighPrecision");
        var original = ReadManifest(input.Manifest);
        // Deliberately forged negative metadata on a copied synthetic fixture.
        // Never treated as a real producer receipt or used for positive import.
        var provenance = original.Provenance with { ExecutionMode = "scalar" };
        provenance = mutation switch
        {
            "input-hash" => provenance with { InputSha256 = new string('0', 64) },
            "questions-hash" => provenance with { QuestionsSha256 = new string('0', 64) },
            "mode-uppercase" => provenance with { ExecutionMode = "Scalar" },
            "mode-unknown" => provenance with { ExecutionMode = "unknown" },
            "mode-empty" => provenance with { ExecutionMode = "" },
            "mode-null" => provenance with { ExecutionMode = null! },
            _ => throw new ArgumentOutOfRangeException(nameof(mutation))
        };
        WriteManifest(input.Manifest, original with { Provenance = provenance });

        var error = await Assert.ThrowsExactlyAsync<InvalidDataException>(() => Import(input, "HighPrecision"));

        Assert.AreEqual("Expected matching completed real scalar/native export, not synthetic or changed input.", error.Message);
        Assert.AreEqual(original.DataSha256, ArtifactExpectations.HashFile(temp.FilePath("decisions.arrow")));
        Assert.AreEqual("synthetic", ReadManifest(Fixture("HighPrecision", "manifest.json")).Provenance.ExecutionMode);
    }

    private static async Task<long[]> Drain(DecisionArrowDatasetReader reader)
    {
        var ids = new List<long>();
        RecordBatch? batch;
        while ((batch = await reader.ReadNextRecordBatchAsync()) is not null)
        {
            using (batch)
                ids.AddRange(Enumerable.Range(0, batch.Length).Select(row => ((Int64Array)batch.Column(0)).GetValue(row)!.Value));
        }
        return ids.ToArray();
    }

    private static DecisionArrowManifest ReadManifest(string path) =>
        JsonSerializer.Deserialize<DecisionArrowManifest>(File.ReadAllText(path), DecisionArrowManifest.JsonOptions)!;

    private static void WriteManifest(string path, DecisionArrowManifest value) =>
        File.WriteAllText(path, JsonSerializer.Serialize(value, DecisionArrowManifest.JsonOptions), ArtifactFiles.Utf8);

    private static string CopyFixture(TempDirectory temp, string precision)
    {
        foreach (string name in new[] { "manifest.json", "contract.json", "decisions.arrow" })
            temp.Put(name, File.ReadAllBytes(Fixture(precision, name)));
        return temp.FilePath("manifest.json");
    }

    private sealed record ImportInput(string Manifest, string PreparationPath, string Split, string States,
        PreparationReceipt Preparation);

    private static ImportInput ImportInputs(TempDirectory temp, string precision)
    {
        string manifestPath = CopyFixture(temp, precision);
        // Authored, label-free preparation for lightweight local unit tests.
        // NOT UCI evidence. Valid IDs/class support also ensure rejection tests
        // cannot pass vacuously because of an earlier preparation failure.
        var source = Enumerable.Range(1, 257).Select(id => new CorpusRow(id, id % 2 == 0, $"unit-{id}")).ToArray();
        string states = temp.PutText("states.jsonl", string.Join("\n",
            source.Select(row => JsonSerializer.Serialize(new { rowId = row.RowId, state = row.Text }))) + "\n");
        var rows = source.Select(row => new SplitRow(row.RowId, row.RowId, row.Label,
            SplitManifest.Names[(int)((row.RowId - 1) % 3)])).ToArray();
        var counts = SplitManifest.Names.Select(name =>
        {
            var selected = rows.Where(row => row.Split == name).ToArray();
            return new PartitionCount(name, selected.Length, selected.Count(row => !row.Label),
                selected.Count(row => row.Label), selected.Length);
        }).ToArray();
        var split = new SplitManifest(1, 1729, ArtifactExpectations.AbcHash, FeatureContract.QuestionsV1Sha256,
            ArtifactExpectations.EmptyHash, "test-only-rejection-fixture", counts, rows);
        split.Validate(source);
        string splitPath = temp.FilePath("split.json");
        ArtifactFiles.Write(splitPath, split);
        var preparation = new PreparationReceipt(1, "complete", ArtifactExpectations.EmptyHash, split.CorpusSha256,
            split.QuestionsSha256, split.GroupDiagnosticsSha256, ArtifactExpectations.HashFile(splitPath),
            ArtifactExpectations.HashFile(states), 257, 129, 128, 257, "authored, not UCI", counts);
        string preparationPath = temp.FilePath("preparation.json");
        ArtifactFiles.Write(preparationPath, preparation);
        var manifest = ReadManifest(manifestPath);
        // Test-only metadata patch; IPC and immutable source stay untouched.
        WriteManifest(manifestPath, manifest with
        {
            Provenance = manifest.Provenance with { InputSha256 = preparation.StatesSha256 }
        });
        return new(manifestPath, preparationPath, splitPath, states, preparation);
    }

    private static Task<ImportedStudy> Import(ImportInput input, string precision) =>
        ArrowFeatureReader.ImportAsync(input.Manifest, Fixture(precision, "contract.json"), Fingerprint(precision),
            input.PreparationPath, input.Split, input.States, Questions);

    private static void AssertObservation(long id, double[] vector, double spam, FeatureObservation actual,
        RecordBatch source, int row)
    {
        Assert.AreEqual(id, actual.RowId);
        Assert.AreEqual(10, actual.Semantic.Length);
        CollectionAssert.AreEqual(vector.Select(value => (float)value).ToArray(), actual.Semantic);
        Assert.AreEqual(spam, actual.SpamBaseline);
        CollectionAssert.AreEqual(vector, RawVector(source, row));
        Assert.AreEqual(spam, ((DoubleArray)source.Column(5)).GetValue(row));
        Assert.AreEqual(id, ((Int64Array)source.Column(0)).GetValue(row));
    }

    private static double[] RawVector(RecordBatch batch, int row)
    {
        using var time = (DoubleArray)((FixedSizeListArray)((StructArray)batch.Column(3)).Fields[4]).GetSlicedValues(row);
        using var purpose = (DoubleArray)((FixedSizeListArray)((StructArray)batch.Column(4)).Fields[1]).GetSlicedValues(row);
        return
        [
            ((DoubleArray)batch.Column(1)).GetValue(row)!.Value,
            ((DoubleArray)batch.Column(2)).GetValue(row)!.Value,
            .. Enumerable.Range(0, 3).Select(i => time.GetValue(i)!.Value),
            .. Enumerable.Range(0, 5).Select(i => purpose.GetValue(i)!.Value)
        ];
    }

    private static double[] DistinctVector(int row) =>
        [.07 + .01 * row, .09 + .01 * row, .1 + .01 * row, .2, .7 - .01 * row,
            .11 + .01 * row, .13, .17, .23, .36 - .01 * row];

    private static DoubleArray Doubles(IEnumerable<double> values) => new DoubleArray.Builder().AppendRange(values).Build();
    private static IArrowArray Clone(IArrowArray array) =>
        ArrowArrayFactory.BuildArray(array.Data.Clone(Apache.Arrow.Memory.MemoryAllocator.Default.Value));

    private static RecordBatch DistinctBatch(RecordBatch template)
    {
        const int count = 4;
        using var small = template.Slice(0, count);
        var pressure = (StructArray)small.Column(3);
        var purpose = (StructArray)small.Column(4);
        var time = new FixedSizeListArray(pressure.Fields[4].Data.DataType, count,
            Doubles(Enumerable.Range(0, count).SelectMany(row => DistinctVector(row).Skip(2).Take(3))), ArrowBuffer.Empty, 0);
        var intent = new FixedSizeListArray(purpose.Fields[1].Data.DataType, count,
            Doubles(Enumerable.Range(0, count).SelectMany(row => DistinctVector(row).Skip(5))), ArrowBuffer.Empty, 0);
        return new RecordBatch(template.Schema,
        [
            new Int64Array.Builder().AppendRange(Enumerable.Range(1001, count).Select(i => (long)i)).Build(),
            Doubles(Enumerable.Range(0, count).Select(row => DistinctVector(row)[0])),
            Doubles(Enumerable.Range(0, count).Select(row => DistinctVector(row)[1])),
            new StructArray(pressure.Data.DataType, count, [.. pressure.Fields.Take(4).Select(Clone), time], ArrowBuffer.Empty, 0),
            new StructArray(purpose.Data.DataType, count, [Clone(purpose.Fields[0]), intent], ArrowBuffer.Empty, 0),
            Doubles(Enumerable.Range(0, count).Select(row => .91 - .01 * row))
        ], count);
    }
}
