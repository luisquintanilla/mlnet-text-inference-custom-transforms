using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace DecisionArrowPredictor;

public sealed class LearningRow
{
    public long RowId { get; set; }
    public long GroupId { get; set; }
    public bool Label { get; set; }
    public string Text { get; set; } = "";
    [VectorType(FeatureContract.Width)]
    public float[] Semantic { get; set; } = [];
    public double SpamBaseline { get; set; }
}

public sealed class ScoredRow
{
    public long RowId { get; set; }
    public long GroupId { get; set; }
    public bool Label { get; set; }
    public float Probability { get; set; }
}

public sealed record ModelReceipt(int Version, string Arm, int TargetRows, int ActualRows, string FeatureFingerprint,
    string Conversion, string DatasetManifestSha256, string SplitSha256, string QuestionsSha256,
    string ModelFile, string ModelSha256, long ModelBytes, double L2, int Seed, int MaxIterations,
    double Threshold, double BudgetThreshold, double TrainingMilliseconds, double TuningAndValidationMilliseconds,
    double WarmHeadMillisecondsPerRow,
    Metrics Validation, string[] Projection);
public sealed record TrainedArm(ModelReceipt Receipt, ITransformer Model);

public static class PredictorTraining
{
    public static readonly double[] L2Grid = [0.0001, 0.001, 0.01];
    public static readonly string[] LearnedArms = ["text", "semantic", "combined"];
    public const int Seed = 1;
    public const int MaxIterations = 100;

    public static LearningRow[] Subset(IReadOnlyList<LearningRow> training, int target)
    {
        if (target <= 0 || !training.Any(r => r.Label) || !training.Any(r => !r.Label) ||
            training.Select(r => r.RowId).Distinct().Count() != training.Count)
            throw new InvalidDataException("Learning subsets need a positive target, unique rows, and both classes.");
        if (target >= training.Count) return training.OrderBy(r => r.RowId).ToArray();
        var groups = training.GroupBy(r => r.GroupId)
            .OrderBy(g => SplitManifest.OrderKey(g.Key, Seed), StringComparer.Ordinal).ToArray();
        var result = new List<LearningRow>();
        foreach (var group in groups)
        {
            result.AddRange(group);
            if (result.Count >= target && result.Any(r => r.Label) && result.Any(r => !r.Label)) break;
        }
        return result.OrderBy(r => r.RowId).ToArray();
    }

    public static TrainedArm Fit(string arm, int target, LearningRow[] training, LearningRow[] validation,
        string identity, string datasetHash, string splitHash, string questionsHash, string output)
    {
        if (!LearnedArms.Contains(arm, StringComparer.Ordinal))
            throw new ArgumentException("Unknown learned arm.", nameof(arm));
        if (!training.Any(r => r.Label) || !training.Any(r => !r.Label) ||
            !validation.Any(r => r.Label) || !validation.Any(r => !r.Label) ||
            training.Select(r => r.RowId).Intersect(validation.Select(r => r.RowId)).Any() ||
            training.Select(r => r.GroupId).Intersect(validation.Select(r => r.GroupId)).Any())
            throw new InvalidDataException("Fit needs disjoint train/validation rows with both classes.");
        ITransformer? best = null;
        Metrics? bestMetrics = null;
        double bestL2 = 0;
        double bestFitMilliseconds = 0;
        var timer = Stopwatch.StartNew();
        foreach (double l2 in L2Grid)
        {
            var context = new MLContext(Seed);
            var view = context.Data.LoadFromEnumerable(training);
            IEstimator<ITransformer> estimator = arm switch
            {
                "text" => context.Transforms.Text.FeaturizeText("Features", nameof(LearningRow.Text)),
                "semantic" => context.Transforms.CopyColumns("Features", nameof(LearningRow.Semantic)),
                "combined" => context.Transforms.Text.FeaturizeText("TextFeatures", nameof(LearningRow.Text))
                    .Append(context.Transforms.Concatenate("Features", "TextFeatures", nameof(LearningRow.Semantic))),
                _ => throw new ArgumentException("Unknown arm.")
            };
            estimator = estimator.Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(
                new SdcaLogisticRegressionBinaryTrainer.Options
                {
                    LabelColumnName = nameof(LearningRow.Label), FeatureColumnName = "Features",
                    L2Regularization = (float)l2, MaximumNumberOfIterations = MaxIterations, NumberOfThreads = 1
                }));
            var fitTimer = Stopwatch.StartNew();
            var fitted = estimator.Fit(view);
            fitTimer.Stop();
            var measured = PredictorEvaluation.Calculate(Predict(fitted, validation), 0.5);
            if (bestMetrics is null || measured.Auprc > bestMetrics.Auprc ||
                (measured.Auprc == bestMetrics.Auprc && measured.LogLoss < bestMetrics.LogLoss))
            {
                best = fitted; bestMetrics = measured; bestL2 = l2; bestFitMilliseconds = fitTimer.Elapsed.TotalMilliseconds;
            }
        }
        timer.Stop();
        var model = best ?? throw new InvalidOperationException("No fitted candidate.");
        var validationPredictions = Predict(model, validation);
        double threshold = PredictorEvaluation.SelectThreshold(validationPredictions, false);
        double budgetThreshold = PredictorEvaluation.SelectThreshold(validationPredictions, true);
        string filename = $"{arm}-{target}.mlnet";
        string path = Path.Combine(output, filename);
        var ml = new MLContext(Seed);
        Directory.CreateDirectory(output);
        using (var file = new FileStream(path + ".partial", FileMode.CreateNew))
            ml.Model.Save(model, ml.Data.LoadFromEnumerable(training).Schema, file);
        File.Move(path + ".partial", path, false);
        _ = Predict(model, validation);
        var warm = Stopwatch.StartNew();
        for (int i = 0; i < 5; i++) _ = Predict(model, validation);
        warm.Stop();
        var receipt = new ModelReceipt(1, arm, target, training.Length, identity, FeatureContract.Conversion,
            datasetHash, splitHash, questionsHash, filename, ArtifactFiles.Hash(path), new FileInfo(path).Length,
            bestL2, Seed, MaxIterations, threshold, budgetThreshold, bestFitMilliseconds, timer.Elapsed.TotalMilliseconds,
            warm.Elapsed.TotalMilliseconds / (5 * validation.Length),
            PredictorEvaluation.Calculate(validationPredictions, threshold), FeatureContract.Projection);
        var loaded = Load(path, receipt, identity);
        var replayed = Predict(loaded, validation);
        if (replayed.Length != validationPredictions.Length ||
            replayed.Where((p, i) => p.RowId != validationPredictions[i].RowId ||
            p.GroupId != validationPredictions[i].GroupId || p.Label != validationPredictions[i].Label ||
            !double.IsFinite(p.Probability) ||
            Math.Abs(p.Probability - validationPredictions[i].Probability) > 1e-6).Any())
            throw new InvalidDataException("Saved model replay differs from fitted validation predictions.");
        ArtifactFiles.Write(Path.Combine(output, $"{arm}-{target}.receipt.json"), receipt);
        return new(receipt, model);
    }

    public static Prediction[] Predict(ITransformer model, IReadOnlyList<LearningRow> rows)
    {
        var ml = new MLContext(Seed);
        return ml.Data.CreateEnumerable<ScoredRow>(model.Transform(ml.Data.LoadFromEnumerable(rows)), false)
            .Select(r => new Prediction(r.RowId, r.GroupId, r.Label, r.Probability)).ToArray();
    }

    public static ITransformer Load(string modelPath, ModelReceipt receipt, string expectedIdentity)
    {
        if (receipt.Version != 1 || receipt.Conversion != FeatureContract.Conversion ||
            !receipt.Projection.SequenceEqual(FeatureContract.Projection) ||
            receipt.ModelBytes != new FileInfo(modelPath).Length)
            throw new InvalidDataException("Saved predictor feature projection/receipt mismatch.");
        FeatureContract.RequireIdentity(receipt.FeatureFingerprint, expectedIdentity);
        ArtifactFiles.RequireHash(modelPath, receipt.ModelSha256);
        using var file = File.OpenRead(modelPath);
        return new MLContext(Seed).Model.Load(file, out _);
    }
}
