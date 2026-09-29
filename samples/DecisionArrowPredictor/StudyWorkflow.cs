using System.Text.Json;

namespace DecisionArrowPredictor;

public sealed record ExtractionCost(string ExecutionMode, int Rows, IReadOnlyDictionary<string, double> Measurements,
    string Scope);
public sealed record ImportedStudy(string FeatureFingerprint, string DatasetManifestSha256,
    string SplitSha256, string QuestionsSha256, LearningRow[] Rows, SplitManifest Split, ExtractionCost? Extraction = null);
public sealed record ArmFreeze(string Arm, int TargetRows, int ActualRows, long[] TrainingRowIds,
    string? ModelReceiptFile, string? ModelReceiptSha256, double? Prior, double Threshold, double BudgetThreshold);
public sealed record TrainingFreeze(int Version, string Status, string FeatureFingerprint,
    string DatasetManifestSha256, string SplitSha256, string QuestionsSha256, string Conversion,
    ArmFreeze[] Arms, string Selection, string LearningSubsetPolicy);
public sealed record ArmEvaluation(string Arm, int TargetRows, int ActualTrainingRows, Metrics Holdout,
    Metrics AtValidationFprBudget, string SavedPredictionsFile, string SavedPredictionsSha256,
    BootstrapReport Uncertainty, BootstrapReport PairedDeltaFromText);
public sealed record StudyEvaluation(int Version, string Status, string TrainingFreezeSha256,
    string FeatureFingerprint, string DatasetManifestSha256, string SplitSha256,
    ArmEvaluation[] Arms, ExtractionCost? Extraction, string MetricsDefinition, string Limitations);

public static class StudyWorkflow
{
    public static void Validate(ImportedStudy study)
    {
        if (string.IsNullOrWhiteSpace(study.FeatureFingerprint))
            throw new InvalidDataException("Missing producer feature identity.");
        if (study.Rows.Length != study.Split.Rows.Length ||
            study.Rows.Select(r => r.RowId).Distinct().Count() != study.Rows.Length)
            throw new InvalidDataException("Imported study count/IDs mismatch.");
        var labels = study.Split.Rows.ToDictionary(r => r.RowId);
        if (study.Rows.Any(r => !labels.TryGetValue(r.RowId, out var s) || r.Label != s.Label || r.GroupId != s.GroupId ||
            r.Semantic.Length != FeatureContract.Width || r.Semantic.Any(v => !float.IsFinite(v) || v < 0 || v > 1) ||
            !double.IsFinite(r.SpamBaseline) || r.SpamBaseline < 0 || r.SpamBaseline > 1))
            throw new InvalidDataException("Imported IDs/labels/groups/probabilities mismatch.");
        study.Split.Validate(study.Rows.Select(r => new CorpusRow(r.RowId, r.Label, r.Text)).ToArray());
        if (study.Split.QuestionsSha256 != study.QuestionsSha256)
            throw new InvalidDataException("Frozen question hash mismatch.");
    }

    public static TrainingFreeze Train(ImportedStudy study, string output)
    {
        Validate(study);
        if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
            throw new IOException("Training output must be empty; models/freezes are immutable.");
        Directory.CreateDirectory(output);
        var assignment = study.Split.Rows.ToDictionary(r => r.RowId, r => r.Split);
        var training = study.Rows.Where(r => assignment[r.RowId] == "train").ToArray();
        var validation = study.Rows.Where(r => assignment[r.RowId] == "validation").ToArray();
        var arms = new List<ArmFreeze>();
        foreach (int target in new[] { 100, 500, training.Length }.Distinct())
        {
            var subset = PredictorTraining.Subset(training, target);
            long[] ids = subset.Select(r => r.RowId).ToArray();
            double prior = (double)subset.Count(r => r.Label) / subset.Length;
            foreach (var baseline in new[] { "prior", "direct" })
            {
                var predictions = Baseline(validation, baseline, prior);
                arms.Add(new(baseline, target, subset.Length, ids, null, null,
                    baseline == "prior" ? prior : null,
                    PredictorEvaluation.SelectThreshold(predictions, false),
                    PredictorEvaluation.SelectThreshold(predictions, true)));
            }
            foreach (string arm in PredictorTraining.LearnedArms)
            {
                var fitted = PredictorTraining.Fit(arm, target, subset, validation, study.FeatureFingerprint,
                    study.DatasetManifestSha256, study.SplitSha256, study.QuestionsSha256, output);
                string filename = $"{arm}-{target}.receipt.json";
                arms.Add(new(arm, target, subset.Length, ids, filename,
                    ArtifactFiles.Hash(Path.Combine(output, filename)), null,
                    fitted.Receipt.Threshold, fitted.Receipt.BudgetThreshold));
            }
        }
        var freeze = new TrainingFreeze(1, "complete", study.FeatureFingerprint, study.DatasetManifestSha256,
            study.SplitSha256, study.QuestionsSha256, FeatureContract.Conversion, arms.ToArray(),
            "Train seed1, single-thread SDCA max100, L2[0.0001,0.001,0.01]; validation tie-aware AP then log loss; " +
            "F1 threshold and maximum recall at <=1% validation FPR, ties by lower FPR then higher threshold. No holdout selection.",
            "SHA256(seed1:groupId) order, whole groups until target and both classes; matched learned-arm IDs. " +
            "Prior uses each matched subset prevalence; direct baseline is stored spam probability.");
        ArtifactFiles.Write(Path.Combine(output, "training.freeze.json"), freeze);
        return freeze;
    }

    public static StudyEvaluation Evaluate(ImportedStudy study, string trainingFreezePath, string output)
    {
        Validate(study);
        var freeze = ArtifactFiles.Read<TrainingFreeze>(trainingFreezePath);
        FeatureContract.RequireIdentity(freeze.FeatureFingerprint, study.FeatureFingerprint);
        if (freeze.Version != 1 || freeze.Status != "complete" || freeze.DatasetManifestSha256 != study.DatasetManifestSha256 ||
            freeze.SplitSha256 != study.SplitSha256 || freeze.QuestionsSha256 != study.QuestionsSha256 ||
            freeze.Conversion != FeatureContract.Conversion)
            throw new InvalidDataException("Training freeze does not match imported study.");
        string root = Path.GetDirectoryName(Path.GetFullPath(trainingFreezePath))!;
        var partitions = study.Split.Rows.ToDictionary(r => r.RowId, r => r.Split);
        var holdout = study.Rows.Where(r => partitions[r.RowId] == "holdout").ToArray();
        int trainingCount = partitions.Values.Count(p => p == "train");
        int[] expectedTargets = [.. new[] { 100, 500, trainingCount }.Distinct().Order()];
        string[] expectedArms = ["combined", "direct", "prior", "semantic", "text"];
        if (freeze.Arms is null || freeze.Arms.Length == 0 ||
            !freeze.Arms.Select(a => a.TargetRows).Distinct().Order().SequenceEqual(expectedTargets) ||
            freeze.Arms.GroupBy(a => a.TargetRows)
                .Any(g => !g.Select(a => a.Arm).Order(StringComparer.Ordinal).SequenceEqual(expectedArms)))
            throw new InvalidDataException("Expected all declared learning curves with exactly five frozen arms each.");
        if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
            throw new IOException("Evaluation output must be empty; saved holdout predictions are immutable.");
        Directory.CreateDirectory(output);
        var results = new List<ArmEvaluation>();
        foreach (var curve in freeze.Arms.GroupBy(a => a.TargetRows))
        {
            if (!curve.Select(a => a.Arm).Order(StringComparer.Ordinal).SequenceEqual(expectedArms))
                throw new InvalidDataException("Expected exactly five frozen arms per subset.");
            var text = curve.Single(a => a.Arm == "text");
            var selectedTraining = text.TrainingRowIds.ToHashSet();
            if (study.Rows.Where(r => partitions[r.RowId] == "train").GroupBy(r => r.GroupId)
                .Any(g => g.Any(r => selectedTraining.Contains(r.RowId)) && g.Any(r => !selectedTraining.Contains(r.RowId))))
                throw new InvalidDataException("Frozen learning subset splits a duplicate group.");
            foreach (var arm in curve)
            {
                if (arm.ActualRows != arm.TrainingRowIds.Length ||
                    !arm.TrainingRowIds.SequenceEqual(text.TrainingRowIds) ||
                    arm.TrainingRowIds.Distinct().Count() != arm.TrainingRowIds.Length ||
                    arm.TrainingRowIds.Any(id => !partitions.TryGetValue(id, out var s) || s != "train"))
                    throw new InvalidDataException("Frozen arms do not use matching training-only source rows.");
                Prediction[] predictions;
                if (arm.Arm is "prior" or "direct")
                    predictions = Baseline(holdout, arm.Arm, arm.Arm == "prior"
                        ? arm.Prior ?? throw new InvalidDataException("Missing frozen training prevalence.") : 0);
                else
                {
                    string file = arm.ModelReceiptFile ?? throw new InvalidDataException("Missing model receipt.");
                    if (Path.GetFileName(file) != file) throw new InvalidDataException("Receipt must be a sibling filename.");
                    string receiptPath = Path.Combine(root, file);
                    ArtifactFiles.RequireHash(receiptPath, arm.ModelReceiptSha256 ?? "");
                    var receipt = ArtifactFiles.Read<ModelReceipt>(receiptPath);
                    if (receipt.Arm != arm.Arm || receipt.TargetRows != arm.TargetRows || receipt.ActualRows != arm.ActualRows ||
                        receipt.Threshold != arm.Threshold || receipt.BudgetThreshold != arm.BudgetThreshold ||
                        receipt.SplitSha256 != study.SplitSha256 || receipt.DatasetManifestSha256 != study.DatasetManifestSha256 ||
                        receipt.QuestionsSha256 != study.QuestionsSha256 || Path.GetFileName(receipt.ModelFile) != receipt.ModelFile)
                        throw new InvalidDataException("Frozen model receipt mismatch.");
                    predictions = PredictorTraining.Predict(PredictorTraining.Load(
                        Path.Combine(root, receipt.ModelFile), receipt, study.FeatureFingerprint), holdout);
                }
                string predictionsFile = $"{arm.Arm}-{arm.TargetRows}.holdout.json";
                ArtifactFiles.Write(Path.Combine(output, predictionsFile), predictions);
            }
            // Bootstrap reads the saved artifacts, never repeats model inference.
            foreach (var arm in curve)
            {
                string filename = $"{arm.Arm}-{arm.TargetRows}.holdout.json";
                var saved = ArtifactFiles.Read<Prediction[]>(Path.Combine(output, filename));
                var reference = ArtifactFiles.Read<Prediction[]>(Path.Combine(output, $"text-{arm.TargetRows}.holdout.json"));
                results.Add(new(arm.Arm, arm.TargetRows, arm.ActualRows,
                    PredictorEvaluation.Calculate(saved, arm.Threshold), PredictorEvaluation.Calculate(saved, arm.BudgetThreshold),
                    filename, ArtifactFiles.Hash(Path.Combine(output, filename)),
                    PredictorEvaluation.Bootstrap(saved, arm.Threshold),
                    PredictorEvaluation.Bootstrap(saved, arm.Threshold, reference, text.Threshold)));
            }
        }
        var report = new StudyEvaluation(1, "complete", ArtifactFiles.Hash(trainingFreezePath), study.FeatureFingerprint,
            study.DatasetManifestSha256, study.SplitSha256, results.ToArray(), study.Extraction,
            "AUPRC = tie-aware step average precision (not trapezoidal interpolation); ROC ties receive half credit. " +
            "Log loss clips evaluation-local probabilities to [1e-15,1-1e-15]; Brier/predictions use unchanged values. " +
            "Undefined class cases reported, 1000 seeded whole-group bootstrap percentile intervals and paired deltas to text.",
            "Public corpus may occur in model pretraining; within-corpus behavior is not deployment generalization or calibration. " +
            "Ten semantic coordinates include redundant exclusive distributions, not ten independent signals. " +
            "Semantic/combined and direct arms require semantic extraction for new text; head timing is not full inference cost. " +
            "No chronological split, cloud inference, raw-message publication, whole-pipeline ONNX export, or extractor distillation.");
        ArtifactFiles.Write(Path.Combine(output, "evaluation.json"), report);
        return report;
    }

    private static Prediction[] Baseline(IReadOnlyList<LearningRow> rows, string arm, double prior)
    {
        if (arm == "prior" && (!double.IsFinite(prior) || prior <= 0 || prior >= 1))
            throw new InvalidDataException("Frozen training prevalence must lie strictly between zero and one.");
        return rows.Select(r => new Prediction(r.RowId, r.GroupId, r.Label,
            arm == "prior" ? prior : r.SpamBaseline)).ToArray();
    }
}
