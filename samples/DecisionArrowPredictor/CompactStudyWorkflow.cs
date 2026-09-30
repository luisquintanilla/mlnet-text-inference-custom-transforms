namespace DecisionArrowPredictor;

public static class CompactStudyWorkflow
{
    public static TrainingFreeze Train(StudyData study, string output)
    {
        study.RequireOpen();
        if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
            throw new IOException("Training output must be empty; models/freezes are immutable.");
        Directory.CreateDirectory(output);
        var training = study.Partition("train");
        var validation = study.Partition("validation");
        var predictions = new PredictionBuffer(validation.Count);
        var arms = new List<ArmFreeze>();
        foreach (int target in new[] { 100, 500, training.Count }.Distinct())
        {
            var subset = training.Subset(target);
            long[] ids = subset.SourceIds();
            double prior = subset.Prevalence();
            foreach (string baseline in new[] { "prior", "direct" })
            {
                predictions.FillBaseline(validation, baseline, prior);
                arms.Add(new(baseline, target, subset.Count, ids, null, null, baseline == "prior" ? prior : null,
                    PredictorEvaluation.SelectThreshold(predictions, false), PredictorEvaluation.SelectThreshold(predictions, true)));
            }
            foreach (string arm in PredictorTraining.LearnedArms)
            {
                var fitted = CompactPredictorTraining.Fit(arm, target, subset, validation, output);
                string filename = $"{arm}-{target}.receipt.json";
                arms.Add(new(arm, target, subset.Count, ids, filename, ArtifactFiles.Hash(Path.Combine(output, filename)),
                    null, fitted.Receipt.Threshold, fitted.Receipt.BudgetThreshold));
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

    public static StudyEvaluation Evaluate(StudyData study, string trainingFreezePath, string output)
    {
        study.RequireOpen();
        var freeze = ArtifactFiles.Read<TrainingFreeze>(trainingFreezePath);
        FeatureContract.RequireIdentity(freeze.FeatureFingerprint, study.FeatureFingerprint);
        if (freeze.Version != 1 || freeze.Status != "complete" || freeze.DatasetManifestSha256 != study.DatasetManifestSha256 ||
            freeze.SplitSha256 != study.SplitSha256 || freeze.QuestionsSha256 != study.QuestionsSha256 ||
            freeze.Conversion != FeatureContract.Conversion)
            throw new InvalidDataException("Training freeze does not match imported study.");
        string root = Path.GetDirectoryName(Path.GetFullPath(trainingFreezePath))!;
        var train = study.Partition("train");
        var holdout = study.Partition("holdout");
        int[] targets = [.. new[] { 100, 500, train.Count }.Distinct().Order()];
        string[] names = ["combined", "direct", "prior", "semantic", "text"];
        if (freeze.Arms is null || freeze.Arms.Length == 0 ||
            !freeze.Arms.Select(a => a.TargetRows).Distinct().Order().SequenceEqual(targets) ||
            freeze.Arms.GroupBy(a => a.TargetRows).Any(g => !g.Select(a => a.Arm).Order(StringComparer.Ordinal).SequenceEqual(names)))
            throw new InvalidDataException("Expected all declared learning curves with exactly five frozen arms each.");
        foreach (var curve in freeze.Arms.GroupBy(a => a.TargetRows))
        {
            long[] ids = train.Subset(curve.Key).SourceIds();
            foreach (var arm in curve)
                if (arm.ActualRows != ids.Length || !arm.TrainingRowIds.SequenceEqual(ids))
                    throw new InvalidDataException("Frozen arms differ from the declared matched training-only grouped subset.");
        }
        if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
            throw new IOException("Evaluation output must be empty; saved holdout predictions are immutable.");
        Directory.CreateDirectory(output);
        var buffer = new PredictionBuffer(holdout.Count);
        var replay = new PredictionBuffer(holdout.Count);
        var results = new List<ArmEvaluation>();
        foreach (var curve in freeze.Arms.GroupBy(a => a.TargetRows))
        {
            var text = curve.Single(a => a.Arm == "text");
            foreach (var arm in curve)
            {
                if (arm.Arm is "prior" or "direct")
                    buffer.FillBaseline(holdout, arm.Arm, arm.Arm == "prior" ?
                        arm.Prior ?? throw new InvalidDataException("Missing frozen training prevalence.") : 0);
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
                    string modelPath = Path.Combine(root, receipt.ModelFile);
                    var view = holdout.View();
                    buffer.Fill(PredictorTraining.Load(modelPath, receipt, study.FeatureFingerprint, view.Schema), view, holdout);
                    replay.Fill(PredictorTraining.Load(modelPath, receipt, study.FeatureFingerprint, view.Schema), view, holdout);
                    replay.RequireReplay(buffer);
                }
                ArtifactFiles.Write(Path.Combine(output, $"{arm.Arm}-{arm.TargetRows}.holdout.json"), buffer.Snapshot());
            }
            // Keep the original whole-group bootstrap and summation order; no inference occurs here.
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
            "Unchanged tie-aware step AP, half-credit ROC ties, metric-local logloss clipping, Brier/confusion; " +
            "1000 seed1729 sorted whole-group bootstrap draws from saved predictions, undefined outcomes retained.",
            "Public-corpus pretraining contamination possible; one grouped nonchronological split; fixed-model bootstrap; " +
            "multiple comparisons. Direct scores are not necessarily calibrated. Laya-vs-Julia is a different-model study, " +
            "not an optimization accuracy claim. Host Arrow Double probabilities, CPU-only ML.NET, no device zero-copy.");
        ArtifactFiles.Write(Path.Combine(output, "evaluation.json"), report);
        return report;
    }
}
