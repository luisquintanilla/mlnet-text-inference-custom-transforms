using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class PreparationWorkflowTests
{
    [TestMethod]
    public void ReviewAndFreeze_LocalEvidencePublishesCompleteHonestArtifacts()
    {
        using var fixture = new WorkflowFixture();
        var evidence = fixture.SnapshotEvidence();
        byte[] diagnosticsBytes = File.ReadAllBytes(fixture.DiagnosticsPath);
        Assert.AreEqual(60, fixture.Groups.Rows);
        Assert.AreEqual(30, fixture.Groups.Groups);
        Assert.AreEqual(2, fixture.Groups.LargestGroup);
        Assert.AreEqual(30, fixture.Groups.MixedLabelGroups);
        Assert.AreEqual(0, fixture.Groups.ChainedGroups);
        Assert.AreEqual("lower-invariant-whitespace/url-digit4-template/normalized-char5-jaccard90/components-source-id-v1",
            fixture.Groups.Algorithm);
        Assert.AreEqual(.90, fixture.Groups.Threshold);
        for (int i = 0; i < 30; i++)
        {
            var group = fixture.Groups.Membership[i];
            Assert.AreEqual(2L * i + 1, group.GroupId);
            CollectionAssert.AreEqual(new long[] { 2 * i + 1, 2 * i + 2 }, group.RowIds);
            Assert.AreEqual(1, group.Ham);
            Assert.AreEqual(1, group.Spam);
            Assert.IsFalse(group.Chained);
        }

        var receipt = fixture.Freeze();

        Assert.AreEqual(1, receipt.Version);
        Assert.AreEqual("complete", receipt.Status);
        Assert.AreEqual(60, receipt.ParsedRows);
        Assert.AreEqual(30, receipt.Ham);
        Assert.AreEqual(30, receipt.Spam);
        Assert.AreEqual(999, receipt.MetadataInstances);
        Assert.AreEqual("Parsed download count is authoritative; metadata=999, parsed=60. " +
            "Bundled README/source composition and line endings are reviewed separately; no rows added/dropped to match metadata.",
            receipt.CountDiscrepancy);
        Assert.AreEqual(ArtifactExpectations.Hash(evidence["acquisition.json"]), receipt.AcquisitionSha256);
        Assert.AreEqual(ArtifactExpectations.Hash(evidence["SMSSpamCollection"]), receipt.CorpusSha256);
        Assert.AreEqual(ArtifactExpectations.Hash(fixture.QuestionsBytes), receipt.QuestionsSha256);
        Assert.AreEqual(ArtifactExpectations.Hash(diagnosticsBytes), receipt.GroupsSha256);
        string splitPath = Path.Combine(fixture.Output, "split.v1.json");
        string statesPath = Path.Combine(fixture.Output, "states.v1.jsonl");
        Assert.AreEqual(ArtifactExpectations.HashFile(splitPath), receipt.SplitSha256);
        Assert.AreEqual(ArtifactExpectations.HashFile(statesPath), receipt.StatesSha256);
        ArtifactExpectations.Bytes(fixture.QuestionsBytes, Path.Combine(fixture.Output, "questions.v1.json"));
        var split = JsonSerializer.Deserialize<SplitManifest>(File.ReadAllBytes(splitPath),
            new JsonSerializerOptions(JsonSerializerDefaults.Web))!;
        Assert.AreEqual(receipt.CorpusSha256, split.CorpusSha256);
        Assert.AreEqual(receipt.QuestionsSha256, split.QuestionsSha256);
        Assert.AreEqual(receipt.GroupsSha256, split.GroupDiagnosticsSha256);
        CollectionAssert.AreEqual(new[]
        {
            new PartitionCount("train", 36, 18, 18, 18),
            new PartitionCount("validation", 12, 6, 6, 6),
            new PartitionCount("holdout", 12, 6, 6, 6)
        }, receipt.Partitions);
        CollectionAssert.AreEqual(receipt.Partitions, split.Counts);
        CollectionAssert.AreEqual(fixture.Source.Select(r => r.RowId).ToArray(), split.Rows.Select(r => r.RowId).ToArray());
        foreach (var row in fixture.Source)
        {
            var assignment = split.Rows.Single(r => r.RowId == row.RowId);
            Assert.AreEqual(row.Label, assignment.Label);
            Assert.AreEqual(row.RowId % 2 == 1 ? row.RowId : row.RowId - 1, assignment.GroupId);
        }
        foreach (var group in fixture.Groups.Membership)
            Assert.AreEqual(1, split.Rows.Where(r => group.RowIds.Contains(r.RowId)).Select(r => r.Split).Distinct().Count());
        split.Validate(fixture.Source);
        string[] lines = File.ReadAllLines(statesPath, new UTF8Encoding(false, true));
        Assert.AreEqual(60, lines.Length);
        for (int i = 0; i < lines.Length; i++)
        {
            using var json = JsonDocument.Parse(lines[i]);
            Assert.AreEqual(2, json.RootElement.EnumerateObject().Count());
            Assert.AreEqual(fixture.Source[i].RowId, json.RootElement.GetProperty("rowId").GetInt64());
            Assert.AreEqual(fixture.Source[i].Text, json.RootElement.GetProperty("state").GetString());
        }
        Assert.IsTrue(lines[0].Contains("\\t", StringComparison.Ordinal), "Original inner tabs must survive JSON escaping.");
        Assert.AreEqual("  ToPiC-00\t ", fixture.Source[0].Text);
        var published = JsonSerializer.Deserialize<PreparationReceipt>(
            File.ReadAllBytes(Path.Combine(fixture.Output, "preparation.v1.json")),
            new JsonSerializerOptions(JsonSerializerDefaults.Web))!;
        Assert.AreEqual("complete", published.Status);
        Assert.AreEqual(60, published.ParsedRows);
        Assert.AreEqual(receipt.AcquisitionSha256, published.AcquisitionSha256);
        Assert.AreEqual(receipt.StatesSha256, published.StatesSha256);
        Assert.AreEqual(receipt.SplitSha256, published.SplitSha256);
        CollectionAssert.AreEqual(receipt.Partitions, published.Partitions);
        CollectionAssert.AreEqual(new[] { "preparation.v1.json", "questions.v1.json", "split.v1.json", "states.v1.jsonl" },
            Directory.GetFiles(fixture.Output).Select(Path.GetFileName).Order(StringComparer.Ordinal).ToArray());
        fixture.AssertEvidence(evidence);
        ArtifactExpectations.Bytes(diagnosticsBytes, fixture.DiagnosticsPath);
    }

    [TestMethod]
    [DataRow("uci-page.html")]
    [DataRow("uci-metadata.json")]
    [DataRow("sms-spam-collection.zip")]
    [DataRow("readme")]
    [DataRow("SMSSpamCollection")]
    public void ReviewAndFreeze_TamperedPinnedEvidenceIsRejected(string name)
    {
        using var fixture = new WorkflowFixture();
        var evidence = fixture.SnapshotEvidence();
        byte[] diagnostics = File.ReadAllBytes(fixture.DiagnosticsPath);
        byte[] changed = evidence[name].ToArray();
        changed[changed.Length / 2] ^= 1;
        string path = fixture.Directory.Put(name, changed);
        string secondReview = fixture.Directory.FilePath("rejected-review");

        var reviewError = Assert.ThrowsExactly<InvalidDataException>(() => PreparationWorkflow.Review(
            fixture.Acquisition.ReceiptPath, fixture.Acquisition.CorpusPath, secondReview));
        var freezeError = Assert.ThrowsExactly<InvalidDataException>(() => fixture.Freeze());

        Assert.AreEqual("SHA-256 mismatch: " + path, reviewError.Message);
        Assert.AreEqual("SHA-256 mismatch: " + path, freezeError.Message);
        Assert.IsFalse(Directory.Exists(secondReview));
        Assert.IsFalse(Directory.Exists(fixture.Output));
        Assert.AreNotEqual(ArtifactExpectations.Hash(evidence[name]), ArtifactExpectations.HashFile(path));
        evidence[name] = changed;
        fixture.AssertEvidence(evidence);
        ArtifactExpectations.Bytes(diagnostics, fixture.DiagnosticsPath);
    }

    [TestMethod]
    [DataRow("stale-hash")]
    [DataRow("membership")]
    [DataRow("membership-order")]
    [DataRow("algorithm")]
    [DataRow("group-count")]
    [DataRow("class-count")]
    public void Freeze_StaleReviewedHashAndFalseRehashedDiagnosticsAreRejected(string mutation)
    {
        using var fixture = new WorkflowFixture();
        var evidence = fixture.SnapshotEvidence();
        byte[] original = File.ReadAllBytes(fixture.DiagnosticsPath);
        var document = JsonNode.Parse(original)!.AsObject();
        var membership = document["membership"]!.AsArray();
        switch (mutation)
        {
            case "membership": membership[0]!["rowIds"]![0] = 9999; break;
            case "membership-order":
                var first = membership[0]!.DeepClone();
                membership[0] = membership[1]!.DeepClone();
                membership[1] = first;
                break;
            case "algorithm": document["algorithm"] = "unreviewed-algorithm"; break;
            case "group-count": document["groups"] = 31; break;
            case "class-count": membership[0]!["ham"] = 2; break;
        }
        string path = mutation == "stale-hash" ? fixture.DiagnosticsPath :
            fixture.Directory.PutText("false-diagnostics.json", document.ToJsonString());
        string hash = mutation == "stale-hash" ? new string('0', 64) : ArtifactExpectations.HashFile(path);

        var error = Assert.ThrowsExactly<InvalidDataException>(() => fixture.Freeze(diagnostics: path, reviewedHash: hash));

        Assert.AreEqual(mutation == "stale-hash" ? "SHA-256 mismatch: " + path :
            "Reviewed grouping diagnostics do not match the pinned corpus/algorithm.", error.Message);
        if (mutation != "stale-hash")
        {
            Assert.AreEqual(hash, ArtifactExpectations.HashFile(path), "Rehashing must not bypass recomputation.");
            Assert.AreNotEqual(fixture.ReviewedHash, hash);
        }
        Assert.IsFalse(Directory.Exists(fixture.Output));
        fixture.AssertEvidence(evidence);
        ArtifactExpectations.Bytes(original, fixture.DiagnosticsPath);
    }

    [TestMethod]
    [DataRow("projection")]
    [DataRow("rubric")]
    [DataRow("candidates")]
    [DataRow("whitespace-only")]
    public void Freeze_ChangedQuestionContractIsRejected(string mutation)
    {
        using var fixture = new WorkflowFixture();
        var evidence = fixture.SnapshotEvidence();
        var document = JsonNode.Parse(fixture.QuestionsBytes)!.AsObject();
        switch (mutation)
        {
            case "projection": document["featureProjection"]![9] = "spam_baseline"; break;
            case "rubric": document["questions"]![2]!["rubric"] = new JsonArray("mild", "none", "explicit urgent deadline"); break;
            case "candidates": document["questions"]![3]!["candidates"]![0]!["id"] = "promotion"; break;
        }
        byte[] changed = mutation == "whitespace-only" ? [.. fixture.QuestionsBytes, 0x20] :
            Encoding.UTF8.GetBytes(document.ToJsonString());
        string path = fixture.Directory.Put("changed-questions.json", changed);

        var error = Assert.ThrowsExactly<InvalidDataException>(() => fixture.Freeze(questions: path));

        Assert.AreEqual("SHA-256 mismatch: " + path, error.Message, "Current Freeze pins exact question bytes before shape validation.");
        Assert.AreNotEqual(ArtifactExpectations.Hash(fixture.QuestionsBytes), ArtifactExpectations.HashFile(path));
        Assert.IsFalse(Directory.Exists(fixture.Output));
        ArtifactExpectations.Bytes(changed, path);
        fixture.AssertEvidence(evidence);
    }

    [TestMethod]
    public void Freeze_RepeatedPublicationPreservesOriginalsWithoutRollbackGuarantee()
    {
        using var fixture = new WorkflowFixture();
        var evidence = fixture.SnapshotEvidence();
        var receipt = fixture.Freeze();
        var outputs = Directory.GetFiles(fixture.Output).ToDictionary(p => p, File.ReadAllBytes, StringComparer.Ordinal);

        Assert.ThrowsExactly<IOException>(() => fixture.Freeze());

        foreach (var (path, bytes) in outputs) ArtifactExpectations.Bytes(bytes, path);
        Assert.AreEqual(4, outputs.Count);
        string partial = Path.Combine(fixture.Output, "questions.v1.json.partial");
        ArtifactExpectations.Bytes(fixture.QuestionsBytes, partial);
        Assert.AreEqual(receipt.SplitSha256, ArtifactExpectations.HashFile(Path.Combine(fixture.Output, "split.v1.json")));
        Assert.AreEqual(receipt.StatesSha256, ArtifactExpectations.HashFile(Path.Combine(fixture.Output, "states.v1.jsonl")));
        Assert.AreEqual(5, Directory.GetFiles(fixture.Output).Length);
        fixture.AssertEvidence(evidence);
    }

    [TestMethod]
    public void Review_RepeatedPublicationPreservesOriginalDiagnostics()
    {
        using var fixture = new WorkflowFixture();
        var evidence = fixture.SnapshotEvidence();
        byte[] original = File.ReadAllBytes(fixture.DiagnosticsPath);

        Assert.ThrowsExactly<IOException>(() => PreparationWorkflow.Review(
            fixture.Acquisition.ReceiptPath, fixture.Acquisition.CorpusPath, fixture.ReviewOutput));

        ArtifactExpectations.Bytes(original, fixture.DiagnosticsPath);
        ArtifactExpectations.Bytes(original, fixture.DiagnosticsPath + ".partial");
        Assert.AreEqual(fixture.ReviewedHash, ArtifactExpectations.HashFile(fixture.DiagnosticsPath));
        Assert.AreEqual(2, Directory.GetFiles(fixture.ReviewOutput).Length);
        fixture.AssertEvidence(evidence);
    }

    [TestMethod]
    [DataRow("split.v1.json", 1)]
    [DataRow("states.v1.jsonl", 2)]
    [DataRow("preparation.v1.json", 3)]
    public void Freeze_PublicationCollisionRetainsEarlierArtifactsAndOriginalDestination(string collision, int earlierCount)
    {
        using var fixture = new WorkflowFixture();
        var evidence = fixture.SnapshotEvidence();
        byte[] reserved = Encoding.UTF8.GetBytes("reserved test-owned destination; not a complete receipt");
        string path = fixture.Directory.Put("frozen/" + collision, reserved);

        Assert.ThrowsExactly<IOException>(() => fixture.Freeze());

        ArtifactExpectations.Bytes(reserved, path);
        Assert.IsTrue(File.Exists(path + ".partial"));
        Assert.IsTrue(new FileInfo(path + ".partial").Length > 0);
        Assert.AreEqual(earlierCount + 2, Directory.GetFiles(fixture.Output).Length);
        ArtifactExpectations.Bytes(fixture.QuestionsBytes, Path.Combine(fixture.Output, "questions.v1.json"));
        if (earlierCount >= 2)
        {
            var split = JsonSerializer.Deserialize<SplitManifest>(File.ReadAllBytes(Path.Combine(fixture.Output, "split.v1.json")),
                new JsonSerializerOptions(JsonSerializerDefaults.Web))!;
            Assert.AreEqual(60, split.Rows.Length);
            CollectionAssert.AreEqual(fixture.Source.Select(r => r.RowId).ToArray(), split.Rows.Select(r => r.RowId).ToArray());
        }
        if (earlierCount >= 3)
            Assert.AreEqual(60, File.ReadAllLines(Path.Combine(fixture.Output, "states.v1.jsonl")).Length);
        if (collision != "preparation.v1.json")
            Assert.IsFalse(File.Exists(Path.Combine(fixture.Output, "preparation.v1.json")));
        else
            Assert.AreNotEqual("complete", Encoding.UTF8.GetString(File.ReadAllBytes(path)));
        fixture.AssertEvidence(evidence);
    }
}
