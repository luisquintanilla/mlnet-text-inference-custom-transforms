using System.IO.Compression;
using System.Text.Json;

namespace DecisionArrowPredictor;

public sealed record CorpusRow(long RowId, bool Label, string Text);
public sealed record AcquisitionReceipt(
    int Version, string SourceUrl, string DownloadUrl, string Doi, string Attribution,
    string License, string LicenseEvidenceFile, string LicenseEvidenceSha256,
    string MetadataFile, string MetadataSha256, string ArchiveFile, string ArchiveSha256,
    string CorpusFile, string CorpusSha256, string ReadmeFile, string ReadmeSha256,
    int MetadataInstances, string Parser, DateTimeOffset AcquiredUtc);

public static class CorpusPreparation
{
    public const string SourceUrl = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection";
    public const string DownloadUrl = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip";
    public const string MetadataUrl = "https://archive.ics.uci.edu/api/dataset?id=228";
    public const string ParserVersion = "first-tab-strict-utf8-original-lines-v1";

    public static IReadOnlyList<CorpusRow> Parse(TextReader reader)
    {
        var result = new List<CorpusRow>();
        string? line;
        long rowId = 0;
        while ((line = reader.ReadLine()) is not null)
        {
            rowId++;
            int tab = line.IndexOf('\t');
            if (tab <= 0 || tab == line.Length - 1)
                throw new InvalidDataException($"Row {rowId}: expected label, first tab, and nonempty original message.");
            string label = line[..tab];
            if (label is not ("ham" or "spam"))
                throw new InvalidDataException($"Row {rowId}: label must be exactly ham or spam.");
            result.Add(new(rowId, label == "spam", line[(tab + 1)..]));
        }
        if (result.Count == 0)
            throw new InvalidDataException("Corpus has no records.");
        return result;
    }

    public static IReadOnlyList<CorpusRow> ReadPinned(string path, string expectedHash)
    {
        ArtifactFiles.RequireHash(path, expectedHash);
        using var reader = new StreamReader(path, ArtifactFiles.Utf8, detectEncodingFromByteOrderMarks: false);
        return Parse(reader);
    }

    public static async Task<AcquisitionReceipt> AcquireAsync(string output, CancellationToken token = default)
    {
        if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
            throw new IOException("Download output must be a new empty directory; acquired bytes are immutable.");
        Directory.CreateDirectory(output);
        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
        var page = await client.GetByteArrayAsync(SourceUrl, token);
        string html = ArtifactFiles.Utf8.GetString(page);
        if (!html.Contains("creativecommons.org/licenses/by/4.0", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("UCI page does not supply the expected CC BY 4.0 license evidence; stop for review.");
        ArtifactFiles.WriteBytes(Path.Combine(output, "uci-page.html"), page);
        var metadata = await client.GetByteArrayAsync(MetadataUrl, token);
        ArtifactFiles.WriteBytes(Path.Combine(output, "uci-metadata.json"), metadata);
        using var doc = JsonDocument.Parse(metadata);
        var data = doc.RootElement.GetProperty("data");
        if (data.GetProperty("dataset_doi").GetString() != "10.24432/C5CC84")
            throw new InvalidDataException("UCI metadata DOI changed.");
        var archive = await client.GetByteArrayAsync(DownloadUrl, token);
        if (archive.Length > 16 * 1024 * 1024)
            throw new InvalidDataException("Unexpected corpus archive size.");
        string archivePath = Path.Combine(output, "sms-spam-collection.zip");
        ArtifactFiles.WriteBytes(archivePath, archive);
        using var zip = new ZipArchive(new MemoryStream(archive), ZipArchiveMode.Read);
        foreach (var name in new[] { "SMSSpamCollection", "readme" })
        {
            var entry = zip.GetEntry(name) ?? throw new InvalidDataException($"Missing ZIP entry {name}.");
            if (entry.Length > 16 * 1024 * 1024)
                throw new InvalidDataException($"Unexpected entry size: {name}.");
            using var input = entry.Open();
            using var bytes = new MemoryStream();
            await input.CopyToAsync(bytes, token);
            ArtifactFiles.WriteBytes(Path.Combine(output, name), bytes.ToArray());
        }
        var receipt = new AcquisitionReceipt(1, SourceUrl, DownloadUrl, "10.24432/C5CC84",
            "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. " +
            "Introductory paper: Almeida, Hidalgo & Yamakami (2011), doi:10.1145/2034691.2034742.",
            "CC BY 4.0 (UCI page); inspect bundled readme for original research-use terms.",
            "uci-page.html", ArtifactFiles.Hash(Path.Combine(output, "uci-page.html")),
            "uci-metadata.json", ArtifactFiles.Hash(Path.Combine(output, "uci-metadata.json")),
            "sms-spam-collection.zip", ArtifactFiles.Hash(archivePath),
            "SMSSpamCollection", ArtifactFiles.Hash(Path.Combine(output, "SMSSpamCollection")),
            "readme", ArtifactFiles.Hash(Path.Combine(output, "readme")),
            data.GetProperty("num_instances").GetInt32(), ParserVersion, DateTimeOffset.UtcNow);
        // Publish provenance before the corpus can be passed to preparation.
        ArtifactFiles.Write(Path.Combine(output, "acquisition.json"), receipt);
        return receipt;
    }

    public static AcquisitionReceipt ValidateAcquisition(string receiptPath, string corpusPath)
    {
        var receipt = ArtifactFiles.Read<AcquisitionReceipt>(receiptPath);
        if (receipt.Version != 1 || receipt.Parser != ParserVersion || receipt.Doi != "10.24432/C5CC84")
            throw new InvalidDataException("Unsupported acquisition receipt.");
        string root = Path.GetDirectoryName(Path.GetFullPath(receiptPath))!;
        foreach (var (file, hash) in new[]
        {
            (receipt.LicenseEvidenceFile, receipt.LicenseEvidenceSha256),
            (receipt.MetadataFile, receipt.MetadataSha256), (receipt.ArchiveFile, receipt.ArchiveSha256),
            (receipt.ReadmeFile, receipt.ReadmeSha256)
        })
        {
            if (Path.GetFileName(file) != file)
                throw new InvalidDataException("Acquisition paths must be sibling filenames.");
            ArtifactFiles.RequireHash(Path.Combine(root, file), hash);
        }
        ArtifactFiles.RequireHash(corpusPath, receipt.CorpusSha256);
        return receipt;
    }
}
