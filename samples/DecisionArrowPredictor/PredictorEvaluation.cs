namespace DecisionArrowPredictor;

public sealed record Prediction(long RowId, long GroupId, bool Label, double Probability);
public sealed record Metrics(int Rows, int Spam, int Ham, double? Auprc, double? RocAuc,
    double LogLoss, double Brier, double Threshold, int TruePositive, int FalsePositive,
    int TrueNegative, int FalseNegative, double? Precision, double? Recall, double? FalsePositiveRate);
public sealed record BootstrapInterval(string Metric, int Defined, int Undefined, double? Lower, double? Upper);
public sealed record BootstrapReport(int Seed, int Resamples, string Unit, BootstrapInterval[] Intervals);

public static class PredictorEvaluation
{
    public const int BootstrapSeed = 1729;
    public const int BootstrapResamples = 1000;
    public const double LogLossEpsilon = 1e-15;

    public static Metrics Calculate(IReadOnlyList<Prediction> predictions, double threshold)
    {
        if (predictions.Count == 0 || !double.IsFinite(threshold) ||
            predictions.Any(p => !double.IsFinite(p.Probability) || p.Probability < 0 || p.Probability > 1))
            throw new InvalidDataException("Metrics require nonempty finite probabilities and a finite threshold.");
        int positive = predictions.Count(p => p.Label), negative = predictions.Count - positive;
        int tp = predictions.Count(p => p.Label && p.Probability >= threshold);
        int fp = predictions.Count(p => !p.Label && p.Probability >= threshold);
        double logLoss = 0, brier = 0;
        foreach (var p in predictions)
        {
            // Clipping is metric-local, only to make log(0) finite; predictions remain unchanged.
            double q = Math.Clamp(p.Probability, LogLossEpsilon, 1 - LogLossEpsilon);
            logLoss -= p.Label ? Math.Log(q) : Math.Log(1 - q);
            brier += Math.Pow(p.Probability - (p.Label ? 1 : 0), 2);
        }
        double ap = 0, roc = 0;
        int seenPositive = 0, seenNegative = 0;
        foreach (var tie in predictions.GroupBy(p => p.Probability).OrderByDescending(g => g.Key))
        {
            int a = tie.Count(p => p.Label), b = tie.Count() - a;
            seenPositive += a; seenNegative += b;
            ap += a * (double)seenPositive / (seenPositive + seenNegative);
            roc += a * (negative - seenNegative + b * 0.5);
        }
        bool both = positive > 0 && negative > 0;
        return new(predictions.Count, positive, negative, both ? ap / positive : null,
            both ? roc / ((double)positive * negative) : null, logLoss / predictions.Count, brier / predictions.Count,
            threshold, tp, fp, negative - fp, positive - tp,
            tp + fp == 0 ? null : (double)tp / (tp + fp), positive == 0 ? null : (double)tp / positive,
            negative == 0 ? null : (double)fp / negative);
    }

    public static double SelectThreshold(IReadOnlyList<Prediction> validation, bool falsePositiveBudget)
    {
        if (!validation.Any(p => p.Label) || !validation.Any(p => !p.Label))
            throw new InvalidDataException("Threshold selection requires both validation classes.");
        var candidates = validation.Select(p => p.Probability).Append(Math.BitIncrement(1d)).Distinct();
        var metrics = candidates.Select(t => Calculate(validation, t));
        if (falsePositiveBudget)
            return metrics.Where(m => m.FalsePositiveRate <= 0.01).OrderByDescending(m => m.Recall)
                .ThenBy(m => m.FalsePositiveRate).ThenByDescending(m => m.Threshold).First().Threshold;
        double F1(Metrics m) => m.TruePositive == 0 ? 0 :
            2d * m.TruePositive / (2 * m.TruePositive + m.FalsePositive + m.FalseNegative);
        return metrics.OrderByDescending(F1).ThenBy(m => m.FalsePositive).ThenByDescending(m => m.Threshold).First().Threshold;
    }

    public static BootstrapReport Bootstrap(IReadOnlyList<Prediction> savedHoldout, double threshold,
        IReadOnlyList<Prediction>? pairedReference = null, double? referenceThreshold = null)
    {
        if (savedHoldout.Select(p => p.RowId).Distinct().Count() != savedHoldout.Count)
            throw new InvalidDataException("Saved holdout predictions have duplicate IDs.");
        var groups = savedHoldout.GroupBy(p => p.GroupId).OrderBy(g => g.Key).Select(g => g.ToArray()).ToArray();
        if (groups.Length == 0) throw new InvalidDataException("No holdout groups.");
        Dictionary<long, Prediction>? reference = null;
        if (pairedReference is not null)
        {
            reference = pairedReference.ToDictionary(p => p.RowId);
            if (reference.Count != savedHoldout.Count ||
                savedHoldout.Any(p => !reference.TryGetValue(p.RowId, out var r) || r.GroupId != p.GroupId || r.Label != p.Label))
                throw new InvalidDataException("Paired bootstrap rows/groups/labels do not match.");
        }
        string[] names = ["auprc", "rocAuc", "logLoss", "brier", "recall", "falsePositiveRate"];
        var values = names.ToDictionary(n => n, _ => new List<double>());
        var random = new Random(BootstrapSeed);
        for (int iteration = 0; iteration < BootstrapResamples; iteration++)
        {
            var sample = Enumerable.Range(0, groups.Length).SelectMany(_ => groups[random.Next(groups.Length)]).ToArray();
            var m = Calculate(sample, threshold);
            var other = reference is null ? null : Calculate(sample.Select(p => reference[p.RowId]).ToArray(),
                referenceThreshold ?? threshold);
            double?[] numbers = [m.Auprc, m.RocAuc, m.LogLoss, m.Brier, m.Recall, m.FalsePositiveRate];
            double?[] baseline = other is null ? new double?[6] :
                [other.Auprc, other.RocAuc, other.LogLoss, other.Brier, other.Recall, other.FalsePositiveRate];
            for (int i = 0; i < names.Length; i++)
                if (numbers[i] is { } number && (reference is null || baseline[i].HasValue))
                    values[names[i]].Add(reference is null ? number : number - baseline[i]!.Value);
        }
        double Quantile(List<double> sorted, double p) => sorted[(int)Math.Floor(p * (sorted.Count - 1))];
        return new(BootstrapSeed, BootstrapResamples, "non-stratified duplicate-group resampling; whole groups with replacement",
            names.Select(n =>
            {
                var v = values[n]; v.Sort();
                return new BootstrapInterval((reference is null ? "" : "pairedDelta:") + n,
                    v.Count, BootstrapResamples - v.Count, v.Count == 0 ? null : Quantile(v, 0.025),
                    v.Count == 0 ? null : Quantile(v, 0.975));
            }).ToArray());
    }
}
