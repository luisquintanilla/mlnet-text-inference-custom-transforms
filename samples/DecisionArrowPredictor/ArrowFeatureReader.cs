using System.Text.Json;
using Apache.Arrow;
using DecisionInference.Arrow;

namespace DecisionArrowPredictor;

public sealed record FeatureObservation(long RowId, float[] Semantic, double SpamBaseline);
public sealed record ImportReceipt(int Version, string Status, string FeatureFingerprint,
    string DatasetManifestSha256, string SplitSha256, string QuestionsSha256, int Rows,
    string Conversion, string[] Projection, ExtractionCost Extraction);

public static class ArrowFeatureReader
{
    public const string LegacyLayaFingerprint = "72acebb6036bfa8fdaa1d47a1490f5ea1bfc6e08fe71e7e84e8c8922ec4de514";

    public static DecisionArrowSchema ExpectedContract(string contractPath, string fingerprint, string questionsPath)
    {
        ArtifactFiles.RequireHash(questionsPath, FeatureContract.QuestionsV1Sha256);
        FeatureContract.ValidateQuestions(File.ReadAllBytes(questionsPath));
        var contract = DecisionArrowSchema.FromCanonicalJson(File.ReadAllText(contractPath, ArtifactFiles.Utf8));
        FeatureContract.RequireIdentity(contract.FeatureFingerprint, fingerprint);
        using var canonical = JsonDocument.Parse(contract.CanonicalJson);
        using var frozen = JsonDocument.Parse(File.ReadAllBytes(questionsPath));
        if (!JsonElement.DeepEquals(canonical.RootElement.GetProperty("questions"), frozen.RootElement.GetProperty("questions")) ||
            !JsonElement.DeepEquals(canonical.RootElement.GetProperty("featureProjection"), frozen.RootElement.GetProperty("featureProjection")) ||
            canonical.RootElement.GetProperty("numericRepresentation").GetString() != "float64")
            throw new InvalidDataException("Producer expected contract differs from exact frozen question strings/projection.");
        return contract;
    }

    public static FeatureObservation[] Project(RecordBatch batch)
    {
        if (batch.Column(0) is not Int64Array ids || batch.Column(1) is not DoubleArray commercial ||
            batch.Column(2) is not DoubleArray contact || batch.Column(3) is not StructArray pressure ||
            batch.Column(4) is not StructArray purpose || batch.Column(5) is not DoubleArray spam ||
            pressure.Fields.Count != 5 || purpose.Fields.Count != 2 ||
            pressure.Fields[4] is not FixedSizeListArray pressureProbabilities ||
            purpose.Fields[1] is not FixedSizeListArray purposeProbabilities)
            throw new InvalidDataException("Expected producer-validated v1 five-question probability layout.");
        var rows = new FeatureObservation[batch.Length];
        double Number(DoubleArray column, int row) => column.GetValue(row) ??
            throw new InvalidDataException("Null probability in completed decision dataset.");
        for (int row = 0; row < batch.Length; row++)
        {
            if (pressure.IsNull(row) || purpose.IsNull(row))
                throw new InvalidDataException("Null decision struct.");
            using var time = pressureProbabilities.GetSlicedValues(row) as DoubleArray ??
                throw new InvalidDataException("Invalid time-pressure probability list.");
            using var intent = purposeProbabilities.GetSlicedValues(row) as DoubleArray ??
                throw new InvalidDataException("Invalid purpose probability list.");
            if (time.Length != 3 || intent.Length != 5)
                throw new InvalidDataException("Malformed probability vector width.");
            double[] vector =
            [
                Number(commercial, row), Number(contact, row),
                Number(time, 0), Number(time, 1), Number(time, 2),
                Number(intent, 0), Number(intent, 1), Number(intent, 2), Number(intent, 3), Number(intent, 4)
            ];
            double direct = Number(spam, row);
            if (!double.IsFinite(direct) || direct < 0 || direct > 1)
                throw new InvalidDataException("Invalid direct spam baseline probability.");
            rows[row] = new(ids.GetValue(row) ?? throw new InvalidDataException("Null row_id."),
                FeatureContract.ConvertProbabilities(vector), direct);
        }
        return rows;
    }

    public static async Task<ImportedStudy> ImportAsync(string manifestPath, string contractPath, string expectedFingerprint,
        string preparationPath, string splitPath, string statesPath, string questionsPath, CancellationToken token = default)
    {
        var preparation = ArtifactFiles.Read<PreparationReceipt>(preparationPath);
        if (preparation.Version != 1 || preparation.Status != "complete")
            throw new InvalidDataException("Preparation is not complete.");
        ArtifactFiles.RequireHash(splitPath, preparation.SplitSha256);
        ArtifactFiles.RequireHash(statesPath, preparation.StatesSha256);
        ArtifactFiles.RequireHash(questionsPath, preparation.QuestionsSha256);
        var split = ArtifactFiles.Read<SplitManifest>(splitPath);
        if (split.CorpusSha256 != preparation.CorpusSha256 || split.QuestionsSha256 != preparation.QuestionsSha256 ||
            split.GroupDiagnosticsSha256 != preparation.GroupsSha256)
            throw new InvalidDataException("Split provenance differs from frozen preparation.");
        var states = ReadStates(statesPath);
        var labels = split.Rows.ToDictionary(r => r.RowId);
        if (states.Count != preparation.ParsedRows || labels.Count != states.Count ||
            !states.Keys.Order().SequenceEqual(labels.Keys.Order()) ||
            labels.Values.Count(r => r.Label) != preparation.Spam ||
            labels.Values.Count(r => !r.Label) != preparation.Ham)
            throw new InvalidDataException("Preparation source ID/class counts differ from states/splits.");
        split.Validate(states.Select(p => new CorpusRow(p.Key, labels[p.Key].Label, p.Value)).ToArray());
        var contract = ExpectedContract(contractPath, expectedFingerprint, questionsPath);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(manifestPath, contract, states.Keys, token);
        if (reader.Manifest.Provenance.InputSha256 != preparation.StatesSha256 ||
            reader.Manifest.Provenance.QuestionsSha256 != preparation.QuestionsSha256 ||
            reader.Manifest.Provenance.ExecutionMode is not ("scalar" or "native" or "scalar-cpu" or "native-cpu"))
            throw new InvalidDataException("Expected matching completed real scalar/native export, not synthetic or changed input.");
        var measures = reader.Manifest.Provenance.Measurements ??
            throw new InvalidDataException("Real semantic extraction measurements are missing.");
        foreach (string name in new[] { "loadMilliseconds", "exportBeforeManifestPublicationMilliseconds" })
            if (!measures.TryGetValue(name, out double value) || !double.IsFinite(value) || value < 0)
                throw new InvalidDataException($"Missing/invalid real semantic extraction measurement: {name}.");
        var imported = new List<LearningRow>(states.Count);
        var seen = new HashSet<long>();
        RecordBatch? batch;
        while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
        {
            using (batch)
                foreach (var observation in Project(batch))
                {
                    if (!seen.Add(observation.RowId) || !labels.TryGetValue(observation.RowId, out var label))
                        throw new InvalidDataException("Duplicate/extra Arrow source ID.");
                    imported.Add(new LearningRow
                    {
                        RowId = observation.RowId, GroupId = label.GroupId, Label = label.Label,
                        Text = states[observation.RowId], Semantic = observation.Semantic, SpamBaseline = observation.SpamBaseline
                    });
                }
        }
        if (!seen.SetEquals(states.Keys))
            throw new InvalidDataException("Missing Arrow source IDs.");
        var extraction = new ExtractionCost(reader.Manifest.Provenance.ExecutionMode, states.Count,
            new Dictionary<string, double>(measures, StringComparer.Ordinal),
            "exportBeforeManifestPublicationMilliseconds: writer setup through generation/preparation/scoring/append/IPC flush/hash/finalize; " +
            "excludes preflight/model load/final manifest serialization. loadMilliseconds is separate. " +
            "firstResultMilliseconds and inferenceAndAppendMilliseconds overlap this scope; do not sum. " +
            "Head batch time per row is recorded separately; extraction remains necessary for new semantic/combined/direct predictions.");
        var study = new ImportedStudy(contract.FeatureFingerprint, ArtifactFiles.Hash(manifestPath), preparation.SplitSha256,
            preparation.QuestionsSha256, imported.OrderBy(r => r.RowId).ToArray(), split, extraction);
        StudyWorkflow.Validate(study);
        return study;
    }

    public static ImportReceipt Receipt(ImportedStudy study) => new(1, "complete", study.FeatureFingerprint,
        study.DatasetManifestSha256, study.SplitSha256, study.QuestionsSha256, study.Rows.Length,
        FeatureContract.Conversion, FeatureContract.Projection,
        study.Extraction ?? throw new InvalidDataException("Missing real extraction cost."));

    public static async Task<StudyData> ImportCompactAsync(string manifestPath, string contractPath,
        string expectedFingerprint, string preparationPath, string splitPath, string statesPath, string questionsPath,
        long numericCapBytes = ProbabilityStore.DefaultNumericCapBytes, CancellationToken token = default)
    {
        var preparation = ArtifactFiles.Read<PreparationReceipt>(preparationPath);
        if (preparation.Version != 1 || preparation.Status != "complete")
            throw new InvalidDataException("Preparation is not complete.");
        ArtifactFiles.RequireHash(splitPath, preparation.SplitSha256);
        ArtifactFiles.RequireHash(statesPath, preparation.StatesSha256);
        ArtifactFiles.RequireHash(questionsPath, preparation.QuestionsSha256);
        var split = ArtifactFiles.Read<SplitManifest>(splitPath);
        if (split.CorpusSha256 != preparation.CorpusSha256 || split.QuestionsSha256 != preparation.QuestionsSha256 ||
            split.GroupDiagnosticsSha256 != preparation.GroupsSha256)
            throw new InvalidDataException("Split provenance differs from frozen preparation.");
        var states = ReadStates(statesPath);
        var labels = split.Rows.ToDictionary(r => r.RowId);
        if (states.Count != preparation.ParsedRows || labels.Count != states.Count ||
            !states.Keys.Order().SequenceEqual(labels.Keys.Order()) ||
            labels.Values.Count(r => r.Label) != preparation.Spam ||
            labels.Values.Count(r => !r.Label) != preparation.Ham)
            throw new InvalidDataException("Preparation source ID/class counts differ from states/splits.");
        split.Validate(states.Select(p => new CorpusRow(p.Key, labels[p.Key].Label, p.Value)).ToArray());
        var metadata = new StudyMetadata(states.OrderBy(p => p.Key).Select(p =>
            new StudyRowMetadata(p.Key, labels[p.Key].GroupId, labels[p.Key].Label, p.Value, labels[p.Key].Split)));
        var contract = ExpectedContract(contractPath, expectedFingerprint, questionsPath);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(manifestPath, contract, states.Keys, token);
        if (reader.Manifest.Provenance.InputSha256 != preparation.StatesSha256 ||
            reader.Manifest.Provenance.QuestionsSha256 != preparation.QuestionsSha256)
            throw new InvalidDataException("Completed dataset source/questions differ from frozen preparation.");
        // Julia is enabled only after its producer-owned canonical contract/package handoff is pinned.
        if (contract.FeatureFingerprint != LegacyLayaFingerprint ||
            reader.Manifest.Provenance.ExecutionMode is not ("scalar" or "native" or "scalar-cpu" or "native-cpu"))
            throw new InvalidDataException("Compact real import currently requires the pinned historical Laya identity/mode; " +
                "Julia awaits its independent producer handoff. Synthetic/unknown/mixed real modes are not accepted.");
        var measurements = reader.Manifest.Provenance.Measurements ??
            throw new InvalidDataException("Real semantic extraction measurements are missing.");
        foreach (string name in new[] { "loadMilliseconds", "exportBeforeManifestPublicationMilliseconds" })
            if (!measurements.TryGetValue(name, out double value) || !double.IsFinite(value) || value < 0)
                throw new InvalidDataException($"Missing/invalid real semantic extraction measurement: {name}.");
        var store = new ProbabilityStore(metadata.Count, numericCapBytes);
        try
        {
            var seen = new bool[metadata.Count];
            int count = 0;
            RecordBatch? batch;
            while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
            {
                using (batch)
                {
                    var accessor = new ProbabilityBatchAccessor(batch);
                    for (int row = 0; row < accessor.Count; row++)
                    {
                        int ordinal = metadata.Ordinal(accessor.RowId(row));
                        if (seen[ordinal]) throw new InvalidDataException("Duplicate Arrow source ID.");
                        store.SetDirect(ordinal, accessor.CopyRow(row, store.WritableSemantic(ordinal)));
                        seen[ordinal] = true;
                        count = checked(count + 1);
                    }
                }
            }
            if (count != metadata.Count || seen.Any(value => !value) || reader.Manifest.RowCount != count)
                throw new InvalidDataException("Missing Arrow source IDs/count mismatch.");
            token.ThrowIfCancellationRequested();
            var extraction = new ExtractionCost(reader.Manifest.Provenance.ExecutionMode, count,
                new Dictionary<string, double>(measurements, StringComparer.Ordinal),
                "Producer exportBeforeManifestPublicationMilliseconds includes scoring/append/IPC/hash/finalize; " +
                "loadMilliseconds separate; nested producer stages overlap. Consumer import/open/projection are separate.");
            return new(metadata, store, contract.FeatureFingerprint, ArtifactFiles.Hash(manifestPath),
                preparation.SplitSha256, preparation.QuestionsSha256, extraction);
        }
        catch
        {
            store.Dispose();
            throw;
        }
    }

    public static async Task<int> SmokeAsync(string manifestPath, string contractPath, string expectedFingerprint,
        string questionsPath, CancellationToken token = default)
    {
        var expected = ExpectedContract(contractPath, expectedFingerprint, questionsPath);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(manifestPath, expected, token);
        if (reader.Manifest.Provenance.ExecutionMode != "synthetic")
            throw new InvalidDataException("Offline smoke requires a declared synthetic fixture.");
        int rows = 0;
        RecordBatch? batch;
        while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
        {
            using (batch) rows = checked(rows + Project(batch).Length);
        }
        if (rows != reader.Manifest.RowCount)
            throw new InvalidDataException("Fixture row count changed.");
        return rows;
    }

    private static Dictionary<long, string> ReadStates(string path)
    {
        var result = new Dictionary<long, string>();
        using var input = new StreamReader(path, ArtifactFiles.Utf8, false);
        string? line;
        long lineId = 0;
        while ((line = input.ReadLine()) is not null)
        {
            lineId++;
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            if (root.EnumerateObject().Count() != 2 || !root.TryGetProperty("rowId", out var id) ||
                !root.TryGetProperty("state", out var state) || state.ValueKind != JsonValueKind.String)
                throw new InvalidDataException($"State export line {lineId}: expected label-free rowId/state only.");
            if (!result.TryAdd(id.GetInt64(), state.GetString()!))
                throw new InvalidDataException($"State export line {lineId}: duplicate row ID.");
        }
        return result;
    }
}
