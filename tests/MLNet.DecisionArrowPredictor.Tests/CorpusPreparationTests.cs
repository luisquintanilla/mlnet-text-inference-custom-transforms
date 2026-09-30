using System.IO.Compression;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class CorpusPreparationTests
{
    [TestMethod]
    public void Parse_FirstTabPreservesTextIdsAndLabels()
    {
        using var reader = new StringReader(CorpusFixtures.FirstTab);

        var rows = CorpusPreparation.Parse(reader);

        CollectionAssert.AreEqual(new long[] { 1, 2 }, rows.Select(r => r.RowId).ToArray());
        CollectionAssert.AreEqual(new[] { false, true }, rows.Select(r => r.Label).ToArray());
        CollectionAssert.AreEqual(new[] { "  Keep\tinner tab  ", "Offer \t now  " }, rows.Select(r => r.Text).ToArray());
        Assert.AreEqual(2, rows.Count);
        Assert.IsNull(reader.ReadLine());
        Assert.AreEqual("ham\t  Keep\tinner tab  \nspam\tOffer \t now  \n", CorpusFixtures.FirstTab);
    }

    [TestMethod]
    public void Parse_WhitespaceOnlyMessageIsRetained()
    {
        using var reader = new StringReader("ham\t   \r\nspam\tx\r\n");

        var rows = CorpusPreparation.Parse(reader);

        Assert.AreEqual(new CorpusRow(1, false, "   "), rows[0]);
        Assert.AreEqual(new CorpusRow(2, true, "x"), rows[1]);
        Assert.AreEqual(3, rows[0].Text.Length);
        Assert.AreEqual(2, rows.Count);
    }

    [TestMethod]
    [DataRow("empty-row", "", false)]
    [DataRow("no-tab", "PRIVATE-MESSAGE", false)]
    [DataRow("leading-tab", "\tPRIVATE-MESSAGE", false)]
    [DataRow("empty-message", "ham\t", false)]
    [DataRow("uppercase-ham", "HAM\tPRIVATE-MESSAGE", true)]
    [DataRow("mixed-case-spam", "Spam\tPRIVATE-MESSAGE", true)]
    [DataRow("unknown-label", "PRIVATE-LABEL\tPRIVATE-MESSAGE", true)]
    public void Parse_MalformedRowReportsOnlyRowAndRule(string caseId, string badRow, bool labelRule)
    {
        using var reader = new StringReader("ham\tvalid original\n" + badRow + "\n");

        var error = Assert.ThrowsExactly<InvalidDataException>(() => CorpusPreparation.Parse(reader));

        Assert.AreEqual(labelRule ? "Row 2: label must be exactly ham or spam." :
            "Row 2: expected label, first tab, and nonempty original message.", error.Message, caseId);
        Assert.IsFalse(error.Message.Contains("PRIVATE", StringComparison.Ordinal));
        Assert.IsFalse(error.Message.Contains("HAM", StringComparison.Ordinal));
        Assert.IsFalse(error.Message.Contains("Spam", StringComparison.Ordinal));
        Assert.IsFalse(error.Message.Contains("valid original", StringComparison.Ordinal));
    }

    [TestMethod]
    public void Parse_EmptyCorpusIsRejected()
    {
        using var reader = new StringReader("");

        var error = Assert.ThrowsExactly<InvalidDataException>(() => CorpusPreparation.Parse(reader));

        Assert.AreEqual("Corpus has no records.", error.Message);
        Assert.AreEqual(-1, reader.Peek());
    }

    [TestMethod]
    public void ReadPinned_VerifiesHashBeforeStrictUtf8Parsing()
    {
        using var temp = new TempDirectory();
        byte[] original = Encoding.UTF8.GetBytes(CorpusFixtures.FirstTab);
        string path = temp.Put("corpus", original);
        string hash = ArtifactExpectations.Hash(original);

        var rows = CorpusPreparation.ReadPinned(path, hash);

        CollectionAssert.AreEqual(new long[] { 1, 2 }, rows.Select(r => r.RowId).ToArray());
        CollectionAssert.AreEqual(new[] { "  Keep\tinner tab  ", "Offer \t now  " }, rows.Select(r => r.Text).ToArray());
        CollectionAssert.AreEqual(new[] { false, true }, rows.Select(r => r.Label).ToArray());
        var changed = original.ToArray();
        changed[^2] = (byte)'!';
        File.WriteAllBytes(path, changed);
        var mismatch = Assert.ThrowsExactly<InvalidDataException>(() => CorpusPreparation.ReadPinned(path, hash));
        Assert.AreEqual("SHA-256 mismatch: " + path, mismatch.Message);
        ArtifactExpectations.Bytes(changed, path);
        byte[] invalid = [0xff, 0xfe, 0xff];
        File.WriteAllBytes(path, invalid);
        var firstGuard = Assert.ThrowsExactly<InvalidDataException>(() => CorpusPreparation.ReadPinned(path, hash));
        Assert.AreEqual("SHA-256 mismatch: " + path, firstGuard.Message);
        ArtifactExpectations.Bytes(invalid, path);
    }

    [TestMethod]
    [DataRow("invalid-utf8")]
    [DataRow("bom")]
    public void ReadPinned_BomAndInvalidUtf8AreNotAccepted(string caseId)
    {
        using var temp = new TempDirectory();
        byte[] bytes = caseId == "bom"
            ? [0xef, 0xbb, 0xbf, .. Encoding.UTF8.GetBytes("ham\toriginal\n")]
            : [.. Encoding.UTF8.GetBytes("ham\t"), 0xc3, 0x28, 0x0a];
        string path = temp.Put("invalid-corpus", bytes);
        string hash = ArtifactExpectations.Hash(bytes);

        if (caseId == "invalid-utf8")
            Assert.ThrowsExactly<DecoderFallbackException>(() => CorpusPreparation.ReadPinned(path, hash));
        else
        {
            var error = Assert.ThrowsExactly<InvalidDataException>(() => CorpusPreparation.ReadPinned(path, hash));
            Assert.AreEqual("Row 1: label must be exactly ham or spam.", error.Message);
        }
        Assert.AreEqual(hash, ArtifactExpectations.HashFile(path));
        ArtifactExpectations.Bytes(bytes, path);
    }

    [TestMethod]
    [DataRow("valid")]
    [DataRow("uci-page.html")]
    [DataRow("uci-metadata.json")]
    [DataRow("sms-spam-collection.zip")]
    [DataRow("readme")]
    [DataRow("SMSSpamCollection")]
    public void ValidateAcquisition_LocalEvidenceIsHashPinned(string caseId)
    {
        using var fixture = new LocalAcquisitionFixture();
        byte[] originalCorpus = File.ReadAllBytes(fixture.CorpusPath);
        byte[] originalReceipt = File.ReadAllBytes(fixture.ReceiptPath);
        if (caseId != "valid")
        {
            string changedPath = fixture.Directory.FilePath(caseId);
            byte[] changed = [.. File.ReadAllBytes(changedPath), 0x21];
            File.WriteAllBytes(changedPath, changed);

            var error = Assert.ThrowsExactly<InvalidDataException>(() =>
                CorpusPreparation.ValidateAcquisition(fixture.ReceiptPath, fixture.CorpusPath));

            Assert.AreEqual("SHA-256 mismatch: " + changedPath, error.Message);
            ArtifactExpectations.Bytes(changed, changedPath);
            ArtifactExpectations.Bytes(originalReceipt, fixture.ReceiptPath);
            if (caseId != "SMSSpamCollection") ArtifactExpectations.Bytes(originalCorpus, fixture.CorpusPath);
            return;
        }

        var receipt = CorpusPreparation.ValidateAcquisition(fixture.ReceiptPath, fixture.CorpusPath);

        Assert.AreEqual(1, receipt.Version);
        Assert.AreEqual("https://archive.ics.uci.edu/dataset/228/sms+spam+collection", receipt.SourceUrl);
        Assert.AreEqual("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip", receipt.DownloadUrl);
        Assert.AreEqual("10.24432/C5CC84", receipt.Doi);
        Assert.AreEqual("Authored local test evidence", receipt.Attribution);
        Assert.AreEqual("Local CC BY 4.0 evidence", receipt.License);
        Assert.AreEqual("uci-page.html", receipt.LicenseEvidenceFile);
        Assert.AreEqual("uci-metadata.json", receipt.MetadataFile);
        Assert.AreEqual("sms-spam-collection.zip", receipt.ArchiveFile);
        Assert.AreEqual("SMSSpamCollection", receipt.CorpusFile);
        Assert.AreEqual("readme", receipt.ReadmeFile);
        Assert.AreEqual(ArtifactExpectations.HashFile(fixture.Directory.FilePath("uci-page.html")), receipt.LicenseEvidenceSha256);
        Assert.AreEqual(ArtifactExpectations.HashFile(fixture.Directory.FilePath("uci-metadata.json")), receipt.MetadataSha256);
        Assert.AreEqual(ArtifactExpectations.HashFile(fixture.Directory.FilePath("sms-spam-collection.zip")), receipt.ArchiveSha256);
        Assert.AreEqual(ArtifactExpectations.Hash(originalCorpus), receipt.CorpusSha256);
        Assert.AreEqual(ArtifactExpectations.HashFile(fixture.Directory.FilePath("readme")), receipt.ReadmeSha256);
        Assert.AreEqual(999, receipt.MetadataInstances);
        Assert.AreEqual("first-tab-strict-utf8-original-lines-v1", receipt.Parser);
        Assert.AreEqual(LocalAcquisitionFixture.Timestamp, receipt.AcquiredUtc);
        Assert.AreEqual(60, CorpusPreparation.ReadPinned(fixture.CorpusPath, receipt.CorpusSha256).Count);
        using var zip = ZipFile.OpenRead(fixture.Directory.FilePath(receipt.ArchiveFile));
        CollectionAssert.AreEqual(new[] { "SMSSpamCollection", "readme" }, zip.Entries.Select(e => e.FullName).ToArray());
        using var content = zip.GetEntry("SMSSpamCollection")!.Open();
        using var memory = new MemoryStream();
        content.CopyTo(memory);
        CollectionAssert.AreEqual(originalCorpus, memory.ToArray());
        ArtifactExpectations.Bytes(originalReceipt, fixture.ReceiptPath);
        ArtifactExpectations.Bytes(originalCorpus, fixture.CorpusPath);
    }

    [TestMethod]
    [DataRow("version")]
    [DataRow("parser")]
    [DataRow("doi")]
    [DataRow("license-traversal")]
    [DataRow("license-rooted")]
    [DataRow("license-nested")]
    [DataRow("metadata-traversal")]
    [DataRow("metadata-rooted")]
    [DataRow("metadata-nested")]
    [DataRow("archive-traversal")]
    [DataRow("archive-rooted")]
    [DataRow("archive-nested")]
    [DataRow("readme-traversal")]
    [DataRow("readme-rooted")]
    [DataRow("readme-nested")]
    public void ValidateAcquisition_RejectsUnsupportedReceiptAndNonSiblingEvidenceNames(string mutation)
    {
        using var fixture = new LocalAcquisitionFixture();
        var receipt = fixture.Receipt;
        byte[] originalCorpus = File.ReadAllBytes(fixture.CorpusPath);
        if (mutation == "version") receipt = receipt with { Version = 2 };
        else if (mutation == "parser") receipt = receipt with { Parser = "other-parser" };
        else if (mutation == "doi") receipt = receipt with { Doi = "other-doi" };
        else
        {
            string[] parts = mutation.Split('-');
            string name = parts[1] switch
            {
                "traversal" => "../readme",
                "rooted" => fixture.Directory.FilePath("readme"),
                "nested" => "nested/readme",
                _ => throw new ArgumentOutOfRangeException(nameof(mutation))
            };
            receipt = parts[0] switch
            {
                "license" => receipt with { LicenseEvidenceFile = name },
                "metadata" => receipt with { MetadataFile = name },
                "archive" => receipt with { ArchiveFile = name },
                "readme" => receipt with { ReadmeFile = name },
                _ => throw new ArgumentOutOfRangeException(nameof(mutation))
            };
        }
        fixture.WriteReceipt(receipt);
        byte[] snapshot = File.ReadAllBytes(fixture.ReceiptPath);

        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            CorpusPreparation.ValidateAcquisition(fixture.ReceiptPath, fixture.CorpusPath));

        Assert.AreEqual(mutation is "version" or "parser" or "doi"
            ? "Unsupported acquisition receipt." : "Acquisition paths must be sibling filenames.", error.Message);
        ArtifactExpectations.Bytes(originalCorpus, fixture.CorpusPath);
        ArtifactExpectations.Bytes(snapshot, fixture.ReceiptPath);
        Assert.AreEqual(5, System.IO.Directory.GetFiles(fixture.Directory.Root).Count(p => p != fixture.ReceiptPath));
    }
}
