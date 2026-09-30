using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace DecisionArrowPredictor;

public static class CompactPredictorTraining
{
    public static IEstimator<ITransformer> Estimator(MLContext context, string arm, double l2,
        Func<IDataView, IDataView>? observeLearnerInput = null)
    {
        IEstimator<ITransformer> estimator = arm switch
        {
            "text" => context.Transforms.Text.FeaturizeText("Features", nameof(LearningRow.Text)),
            "semantic" => context.Transforms.CopyColumns("Features", nameof(LearningRow.Semantic)),
            "combined" => context.Transforms.Text.FeaturizeText("TextFeatures", nameof(LearningRow.Text))
                .Append(context.Transforms.Concatenate("Features", "TextFeatures", nameof(LearningRow.Semantic))),
            _ => throw new ArgumentException("Unknown learned arm.", nameof(arm))
        };
        IEstimator<ITransformer> learner = context.BinaryClassification.Trainers.SdcaLogisticRegression(
            new SdcaLogisticRegressionBinaryTrainer.Options
            {
                LabelColumnName = nameof(LearningRow.Label), FeatureColumnName = "Features",
                L2Regularization = (float)l2, MaximumNumberOfIterations = PredictorTraining.MaxIterations, NumberOfThreads = 1
            });
        return estimator.Append(observeLearnerInput is null ? learner : new TracedEstimator(learner, observeLearnerInput));
    }

    public static TrainedArm Fit(string arm, int target, RowSelection training, RowSelection validation, string output)
    {
        var data = training.Owner;
        data.RequireOpen();
        if (!ReferenceEquals(data, validation.Owner) || training.Prevalence() is <= 0 or >= 1 ||
            validation.Prevalence() is <= 0 or >= 1 ||
            training.SourceIds().Intersect(validation.SourceIds()).Any() ||
            Enumerable.Range(0, training.Count).Select(i => data.Metadata[training[i]].GroupId)
                .Intersect(Enumerable.Range(0, validation.Count).Select(i => data.Metadata[validation[i]].GroupId)).Any())
            throw new InvalidDataException("Fit needs disjoint train/validation rows with both classes.");
        var validationView = validation.View();
        var predictions = new PredictionBuffer(validation.Count);
        ITransformer? best = null;
        Metrics? bestMetrics = null;
        double bestL2 = 0, bestFitMilliseconds = 0;
        var timer = Stopwatch.StartNew();
        foreach (double l2 in PredictorTraining.L2Grid)
        {
            var context = new MLContext(PredictorTraining.Seed);
            var trainView = training.View(context);
            var fitTimer = Stopwatch.StartNew();
            var model = Estimator(context, arm, l2).Fit(trainView);
            fitTimer.Stop();
            predictions.Fill(model, validationView, validation);
            var metrics = PredictorEvaluation.Calculate(predictions, .5);
            if (bestMetrics is null || metrics.Auprc > bestMetrics.Auprc ||
                (metrics.Auprc == bestMetrics.Auprc && metrics.LogLoss < bestMetrics.LogLoss))
            {
                best = model; bestMetrics = metrics; bestL2 = l2; bestFitMilliseconds = fitTimer.Elapsed.TotalMilliseconds;
            }
        }
        timer.Stop();
        var fitted = best ?? throw new InvalidOperationException("No fitted candidate.");
        predictions.Fill(fitted, validationView, validation);
        double threshold = PredictorEvaluation.SelectThreshold(predictions, false);
        double budgetThreshold = PredictorEvaluation.SelectThreshold(predictions, true);
        var metricsAtThreshold = PredictorEvaluation.Calculate(predictions, threshold);
        string filename = $"{arm}-{target}.mlnet";
        string path = Path.Combine(output, filename);
        Directory.CreateDirectory(output);
        using (var file = new FileStream(path + ".partial", FileMode.CreateNew))
            new MLContext(PredictorTraining.Seed).Model.Save(fitted, StudyDataView.SharedSchema, file);
        File.Move(path + ".partial", path, false);
        predictions.Fill(fitted, validationView, validation);
        var warm = Stopwatch.StartNew();
        for (int i = 0; i < 5; i++) predictions.Fill(fitted, validationView, validation);
        warm.Stop();
        var receipt = new ModelReceipt(1, arm, target, training.Count, data.FeatureFingerprint, FeatureContract.Conversion,
            data.DatasetManifestSha256, data.SplitSha256, data.QuestionsSha256, filename, ArtifactFiles.Hash(path),
            new FileInfo(path).Length, bestL2, PredictorTraining.Seed, PredictorTraining.MaxIterations, threshold,
            budgetThreshold, bestFitMilliseconds, timer.Elapsed.TotalMilliseconds,
            warm.Elapsed.TotalMilliseconds / (5 * validation.Count), metricsAtThreshold, FeatureContract.Projection);
        var replay = new PredictionBuffer(validation.Count);
        replay.Fill(PredictorTraining.Load(path, receipt, data.FeatureFingerprint, validationView.Schema), validationView, validation);
        replay.RequireReplay(predictions);
        ArtifactFiles.Write(Path.Combine(output, $"{arm}-{target}.receipt.json"), receipt);
        return new(receipt, fitted);
    }
}
