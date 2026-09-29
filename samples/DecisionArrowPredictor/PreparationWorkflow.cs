using System.Text.Json;

namespace DecisionArrowPredictor;

public sealed record PreparationReceipt(int Version, string Status, string AcquisitionSha256,
    string CorpusSha256, string QuestionsSha256, string GroupsSha256, string SplitSha256,
    string StatesSha256, int ParsedRows, int Ham, int Spam, int MetadataInstances,
    string CountDiscrepancy, PartitionCount[] Partitions);

public static class PreparationWorkflow
{
    public static GroupDiagnostics Review(string acquisitionPath, string corpusPath, string output)
    {
        var receipt = CorpusPreparation.ValidateAcquisition(acquisitionPath, corpusPath);
        var source = CorpusPreparation.ReadPinned(corpusPath, receipt.CorpusSha256);
        var groups = DuplicateGrouping.Build(source);
        ArtifactFiles.Write(Path.Combine(output, "group-diagnostics.json"), groups);
        return groups;
    }

    public static PreparationReceipt Freeze(string acquisitionPath, string corpusPath,
        string questionsPath, string diagnosticsPath, string reviewedGroupsHash, string output)
    {
        var acquisition = CorpusPreparation.ValidateAcquisition(acquisitionPath, corpusPath);
        ArtifactFiles.RequireHash(diagnosticsPath, reviewedGroupsHash);
        var source = CorpusPreparation.ReadPinned(corpusPath, acquisition.CorpusSha256);
        var groups = ArtifactFiles.Read<GroupDiagnostics>(diagnosticsPath);
        var recomputed = DuplicateGrouping.Build(source);
        if (JsonSerializer.Serialize(groups, ArtifactFiles.Json) != JsonSerializer.Serialize(recomputed, ArtifactFiles.Json))
            throw new InvalidDataException("Reviewed grouping diagnostics do not match the pinned corpus/algorithm.");
        var questionsBytes = File.ReadAllBytes(questionsPath);
        ArtifactFiles.RequireHash(questionsPath, FeatureContract.QuestionsV1Sha256);
        FeatureContract.ValidateQuestions(questionsBytes);
        string questionsHash = ArtifactFiles.Hash(questionsPath);
        var split = SplitManifest.Create(groups, acquisition.CorpusSha256, questionsHash, reviewedGroupsHash, source);
        split.Validate(source);
        ArtifactFiles.WriteBytes(Path.Combine(output, "questions.v1.json"), questionsBytes);
        ArtifactFiles.Write(Path.Combine(output, "split.v1.json"), split);
        string statesPath = Path.Combine(output, "states.v1.jsonl");
        using (var stream = new FileStream(statesPath + ".partial", FileMode.CreateNew))
        using (var writer = new StreamWriter(stream, ArtifactFiles.Utf8))
            foreach (var row in source)
                writer.WriteLine(JsonSerializer.Serialize(new { rowId = row.RowId, state = row.Text }));
        File.Move(statesPath + ".partial", statesPath, false);
        var receipt = new PreparationReceipt(1, "complete", ArtifactFiles.Hash(acquisitionPath),
            acquisition.CorpusSha256, questionsHash, reviewedGroupsHash,
            ArtifactFiles.Hash(Path.Combine(output, "split.v1.json")), ArtifactFiles.Hash(statesPath),
            source.Count, source.Count(r => !r.Label), source.Count(r => r.Label), acquisition.MetadataInstances,
            $"Parsed download count is authoritative; metadata={acquisition.MetadataInstances}, parsed={source.Count}. " +
            "Bundled README/source composition and line endings are reviewed separately; no rows added/dropped to match metadata.",
            split.Counts);
        ArtifactFiles.Write(Path.Combine(output, "preparation.v1.json"), receipt);
        return receipt;
    }
}
