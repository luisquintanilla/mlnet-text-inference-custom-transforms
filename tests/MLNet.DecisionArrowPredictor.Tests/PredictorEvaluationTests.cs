using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class PredictorEvaluationTests
{
    [TestMethod]
    public void Calculate_TiedScoreGoldenMatchesAllMetricsAndConfusionCounts()
    {
        var rows = EvaluationGoldens.Tied;
        var snapshot = rows.ToArray();

        var actual = PredictorEvaluation.Calculate(rows, .5);

        EvaluationGoldens.MetricsEqual(TiedMetrics(.5, 2, 1, 1, 0, 2d / 3, 1, .5), actual);
        CollectionAssert.AreEqual(snapshot, rows);
        Assert.AreEqual(4, actual.TruePositive + actual.FalsePositive + actual.TrueNegative + actual.FalseNegative);
    }

    [TestMethod]
    public void Calculate_TieOrderingAndAllTiedBlockAreInvariant()
    {
        var rows = EvaluationGoldens.Tied;

        var permuted = PredictorEvaluation.Calculate([rows[1], rows[3], rows[2], rows[0]], .5);
        var tied = PredictorEvaluation.Calculate([new(9, 9, true, .4), new(2, 2, false, .4)], .4);
        var reversed = PredictorEvaluation.Calculate([new(2, 2, false, .4), new(9, 9, true, .4)], .4);

        EvaluationGoldens.MetricsEqual(TiedMetrics(.5, 2, 1, 1, 0, 2d / 3, 1, .5), permuted);
        var golden = new Metrics(2, 1, 1, .5, .5, -Math.Log(.24) / 2, .26, .4, 1, 1, 0, 0, .5, 1, 1);
        EvaluationGoldens.MetricsEqual(golden, tied);
        EvaluationGoldens.MetricsEqual(golden, reversed);
        CollectionAssert.AreEqual(new long[] { 1, 2, 3, 4 }, rows.Select(r => r.RowId).ToArray());
    }

    [TestMethod]
    public void Calculate_EndpointLogLossClampsLocallyAndBrierUsesOriginalValues()
    {
        Prediction[] wrong = [new(1, 10, true, 0), new(2, 20, false, 1)];
        Prediction[] correct = [new(1, 10, true, 1), new(2, 20, false, 0)];

        var wrongMetrics = PredictorEvaluation.Calculate(wrong, .5);
        var correctMetrics = PredictorEvaluation.Calculate(correct, .5);

        EvaluationGoldens.MetricsEqual(new(2, 1, 1, .5, 0, 34.53917619362578, 1, .5, 0, 1, 0, 1, 0, 0, 1), wrongMetrics);
        EvaluationGoldens.MetricsEqual(new(2, 1, 1, 1, 1, 9.992007221626415e-16, 0, .5, 1, 0, 1, 0, 1, 1, 0), correctMetrics);
        Assert.IsTrue(double.IsFinite(wrongMetrics.LogLoss));
        Assert.AreEqual(1e-15, PredictorEvaluation.LogLossEpsilon);
        CollectionAssert.AreEqual(new double[] { 0, 1 }, wrong.Select(r => r.Probability).ToArray());
        CollectionAssert.AreEqual(new double[] { 1, 0 }, correct.Select(r => r.Probability).ToArray());
    }

    [TestMethod]
    [DataRow("spam-only")]
    [DataRow("ham-only")]
    [DataRow("predict-none")]
    public void Calculate_SingleClassAndNoPredictedPositiveMetricsHaveExpectedNulls(string caseId)
    {
        var rows = caseId == "predict-none" ? EvaluationGoldens.Tied :
            new Prediction[] { new(1, 10, caseId == "spam-only", .2), new(2, 20, caseId == "spam-only", .8) };
        double threshold = caseId == "predict-none" ? Math.BitIncrement(1d) : .5;
        var snapshot = rows.ToArray();

        var actual = PredictorEvaluation.Calculate(rows, threshold);

        var expected = caseId switch
        {
            "spam-only" => new Metrics(2, 2, 0, null, null, -Math.Log(.16) / 2, .34, .5, 1, 0, 0, 1, 1, .5, null),
            "ham-only" => new Metrics(2, 0, 2, null, null, -Math.Log(.16) / 2, .34, .5, 0, 1, 1, 0, 0, null, .5),
            "predict-none" => TiedMetrics(threshold, 0, 0, 2, 2, null, 0, 0),
            _ => throw new ArgumentOutOfRangeException(nameof(caseId))
        };
        EvaluationGoldens.MetricsEqual(expected, actual);
        CollectionAssert.AreEqual(snapshot, rows);
        Assert.AreEqual(rows.Length, actual.Ham + actual.Spam);
    }

    [TestMethod]
    public void Calculate_FiniteThresholdAboveOnePredictsNoneAndEqualityIsPositive()
    {
        Prediction[] rows = [new(1, 10, true, 1), new(2, 20, false, .5)];
        double aboveOne = Math.BitIncrement(1d);

        var equality = PredictorEvaluation.Calculate(rows, 1);
        var none = PredictorEvaluation.Calculate(rows, aboveOne);

        EvaluationGoldens.MetricsEqual(new(2, 1, 1, 1, 1, .34657359027997314, .125, 1, 1, 0, 1, 0, 1, 1, 0), equality);
        EvaluationGoldens.MetricsEqual(new(2, 1, 1, 1, 1, .34657359027997314, .125, aboveOne, 0, 0, 1, 1, null, 0, 0), none);
        CollectionAssert.AreEqual(new double[] { 1, .5 }, rows.Select(r => r.Probability).ToArray());
        Assert.IsTrue(aboveOne > 1 && double.IsFinite(aboveOne));
    }

    [TestMethod]
    [DataRow("empty")]
    [DataRow("probability-nan")]
    [DataRow("probability-positive-infinity")]
    [DataRow("probability-negative-infinity")]
    [DataRow("probability-negative")]
    [DataRow("probability-greater-than-one")]
    [DataRow("threshold-nan")]
    [DataRow("threshold-positive-infinity")]
    [DataRow("threshold-negative-infinity")]
    public void Calculate_EmptyNonfiniteAndOutOfBoundsInputsAreRejected(string caseId)
    {
        var rows = caseId == "empty" ? Array.Empty<Prediction>() : EvaluationGoldens.Tied;
        double threshold = .5;
        if (caseId.StartsWith("probability-", StringComparison.Ordinal))
            rows[1] = rows[1] with { Probability = InvalidNumber(caseId["probability-".Length..]) };
        if (caseId.StartsWith("threshold-", StringComparison.Ordinal))
            threshold = InvalidNumber(caseId["threshold-".Length..]);
        var snapshot = rows.ToArray();

        var error = Assert.ThrowsExactly<InvalidDataException>(() => PredictorEvaluation.Calculate(rows, threshold));

        Assert.AreEqual("Metrics require nonempty finite probabilities and a finite threshold.", error.Message);
        CollectionAssert.AreEqual(snapshot, rows);
        Assert.AreEqual(caseId == "empty" ? 0 : 4, rows.Length);
    }

    [TestMethod]
    public void SelectThreshold_F1UsesValidationScoresAndPrefersFewerFalsePositives()
    {
        var validation = EvaluationGoldens.Tied;
        Prediction[] tie =
        [
            new(1, 1, true, .9), new(2, 2, false, .8),
            new(3, 3, false, .7), new(4, 4, true, .6)
        ];

        double threshold = PredictorEvaluation.SelectThreshold(validation, false);
        double tieThreshold = PredictorEvaluation.SelectThreshold(tie, false);

        Assert.AreEqual(.5, threshold);
        Assert.AreEqual(.9, tieThreshold);
        var higher = PredictorEvaluation.Calculate(tie, .9);
        var lower = PredictorEvaluation.Calculate(tie, .6);
        EvaluationGoldens.Near(2d / 3, F1(higher));
        EvaluationGoldens.Near(2d / 3, F1(lower));
        Assert.AreEqual(0, higher.FalsePositive);
        Assert.AreEqual(2, lower.FalsePositive);
        Assert.AreEqual(1, higher.TruePositive);
        Assert.AreEqual(2, lower.TruePositive);
        Assert.AreEqual(.9, PredictorEvaluation.SelectThreshold(tie.Reverse().ToArray(), false));
        Assert.AreEqual(.5, PredictorEvaluation.SelectThreshold([validation[3], validation[1], validation[0], validation[2]], false));
        CollectionAssert.AreEqual(new double[] { .9, .8, .7, .6 }, tie.Select(p => p.Probability).ToArray());
    }

    [TestMethod]
    [DataRow("one-top-ham", .8, 2, 1, 99, 0, 1d)]
    [DataRow("one-lower-ham", .8, 2, 0, 100, 0, 1d)]
    [DataRow("second-ham-at-point-eight", .9, 1, 1, 99, 1, .5)]
    [DataRow("two-top-hams", 0d, 0, 0, 100, 2, 0d)]
    public void SelectThreshold_BudgetIncludesOnePercentAndNeverSplitsTies(
        string caseId, double expectedThreshold, int tp, int fp, int tn, int fn, double recall)
    {
        var rows = Enumerable.Range(0, 100).Select(i => new Prediction(i + 3, i + 3, false,
            i == 0 ? caseId == "one-lower-ham" ? .7 : .9 : i == 1 && caseId == "second-ham-at-point-eight" ? .8 :
                i == 1 && caseId == "two-top-hams" ? .9 : .1))
            .Prepend(new(2, 2, true, .8)).Prepend(new(1, 1, true, .9)).ToArray();
        if (caseId == "two-top-hams") expectedThreshold = Math.BitIncrement(1d);
        var snapshot = rows.ToArray();

        double threshold = PredictorEvaluation.SelectThreshold(rows, true);
        var metrics = PredictorEvaluation.Calculate(rows, threshold);

        Assert.AreEqual(expectedThreshold, threshold);
        Assert.AreEqual(tp, metrics.TruePositive);
        Assert.AreEqual(fp, metrics.FalsePositive);
        Assert.AreEqual(tn, metrics.TrueNegative);
        Assert.AreEqual(fn, metrics.FalseNegative);
        EvaluationGoldens.NullableNear(recall, metrics.Recall);
        EvaluationGoldens.NullableNear(fp / 100d, metrics.FalsePositiveRate);
        Assert.IsTrue(metrics.FalsePositiveRate <= .01);
        Assert.AreEqual(102, metrics.Rows);
        Assert.AreEqual(threshold, PredictorEvaluation.SelectThreshold(rows.Reverse().ToArray(), true));
        Assert.AreEqual(Math.BitIncrement(1d), PredictorEvaluation.SelectThreshold(EvaluationGoldens.Tied, true));
        CollectionAssert.AreEqual(snapshot, rows);
    }

    [TestMethod]
    [DataRow("empty", false)]
    [DataRow("empty", true)]
    [DataRow("ham-only", false)]
    [DataRow("ham-only", true)]
    [DataRow("spam-only", false)]
    [DataRow("spam-only", true)]
    public void SelectThreshold_EmptyOrSingleClassInputIsRejected(string caseId, bool budget)
    {
        Prediction[] rows = caseId == "empty" ? [] :
            [new(1, 10, caseId == "spam-only", .2), new(2, 20, caseId == "spam-only", .8)];
        var snapshot = rows.ToArray();

        var error = Assert.ThrowsExactly<InvalidDataException>(() => PredictorEvaluation.SelectThreshold(rows, budget));

        Assert.AreEqual("Threshold selection requires both validation classes.", error.Message);
        CollectionAssert.AreEqual(snapshot, rows);
        Assert.AreEqual(caseId == "empty" ? 0 : 2, rows.Length);
    }

    [TestMethod]
    public void Bootstrap_GroupedDrawsMatchIndependentSeededCountsAndWeightedBounds()
    {
        var rows = EvaluationGoldens.Grouped;
        var snapshot = rows.ToArray();
        CollectionAssert.AreEqual(new[] { 233, 505, 262 }, EvaluationGoldens.DrawOutcomes());

        var report = PredictorEvaluation.Bootstrap(rows, .5);

        double lowLoss = -Math.Log(.8), highLoss = (-Math.Log(.8) - Math.Log(.6)) / 2;
        EvaluationGoldens.Report(report, EvaluationGoldens.Defined,
            [1, 1, lowLoss, .04, 1, 0], [1, 1, highLoss, .10, 1, 0]);
        var brier = EvaluationGoldens.FloorBounds(.10, (.04 + .16 + .04) / 3, .04);
        EvaluationGoldens.NullableNear(brier[0], report.Intervals[3].Lower);
        EvaluationGoldens.NullableNear(brier[1], report.Intervals[3].Upper);
        CollectionAssert.AreEqual(snapshot, rows);
        Assert.AreEqual(1729, PredictorEvaluation.BootstrapSeed);
        Assert.AreEqual(1000, PredictorEvaluation.BootstrapResamples);

        // In F6 the mixed outcome is hidden between both returned quantiles.
        // Four groups expose a 3-ham-group/1-spam-group replicate at the upper
        // quantile: seven real rows, not an unweighted mean of four group means.
        Prediction[] weighted =
        [
            .. rows, new(4, 30, true, .8), new(5, 40, true, .8)
        ];
        var weightedReport = PredictorEvaluation.Bootstrap(weighted, .5);
        EvaluationGoldens.NullableNear(.04, weightedReport.Intervals[3].Lower);
        EvaluationGoldens.NullableNear(.64 / 7, weightedReport.Intervals[3].Upper);
        Assert.AreEqual(1000, weightedReport.Intervals[3].Defined);
        Assert.AreEqual(0, weightedReport.Intervals[3].Undefined);
        Assert.AreNotEqual(.085, weightedReport.Intervals[3].Upper); // Wrong equal-group weighting.
        EvaluationGoldens.NullableNear((-3 * Math.Log(.8) - 3 * Math.Log(.6) - Math.Log(.8)) / 7,
            weightedReport.Intervals[2].Upper);
    }

    [TestMethod]
    public void Bootstrap_FloorQuantilesUseNonInterpolatedOrderStatistics()
    {
        double[] scores = [.02, .12, .27, .36, .48, .65, .79, .95];
        Prediction[] rows = scores.Select((p, i) => new Prediction(i + 1, i + 1, i % 2 == 1, p)).ToArray();
        var snapshot = rows.ToArray();

        var report = PredictorEvaluation.Bootstrap(rows, .5);

        // Independently computed 1000 Random(1729) draws of eight whole groups:
        // adjacent order statistics differ, so interpolation/ceiling cannot pass.
        var brier = report.Intervals.Single(i => i.Metric == "brier");
        Assert.AreEqual(1000, brier.Defined);
        Assert.AreEqual(0, brier.Undefined);
        EvaluationGoldens.NullableNear(.1117625, brier.Lower); // floor(.025 * 999) = 24
        EvaluationGoldens.NullableNear(.4693250000000001, brier.Upper); // floor(.975 * 999) = 974
        Assert.AreNotEqual(.112275, brier.Lower); // index 25
        Assert.AreNotEqual(.4714375, brier.Upper); // index 975
        foreach (var interval in report.Intervals)
            Assert.AreEqual(1000, interval.Defined + interval.Undefined);
        Assert.AreEqual(996, report.Intervals.Single(i => i.Metric == "auprc").Defined);
        Assert.AreEqual(4, report.Intervals.Single(i => i.Metric == "rocAuc").Undefined);
        CollectionAssert.AreEqual(snapshot, rows);
    }

    [TestMethod]
    public void Bootstrap_PermutationAndRepeatAreDeterministic()
    {
        var rows = EvaluationGoldens.Grouped;
        var first = PredictorEvaluation.Bootstrap(rows, .5);

        var repeat = PredictorEvaluation.Bootstrap(rows, .5);
        var permuted = PredictorEvaluation.Bootstrap([rows[2], rows[1], rows[0]], .5);

        Assert.AreEqual(first.Seed, repeat.Seed);
        Assert.AreEqual(first.Resamples, repeat.Resamples);
        Assert.AreEqual(first.Unit, repeat.Unit);
        CollectionAssert.AreEqual(first.Intervals, repeat.Intervals);
        EvaluationGoldens.Report(permuted, EvaluationGoldens.Defined,
            [1, 1, -Math.Log(.8), .04, 1, 0],
            [1, 1, (-Math.Log(.8) - Math.Log(.6)) / 2, .10, 1, 0]);
        Assert.AreEqual(first.Unit, permuted.Unit);
        CollectionAssert.AreEqual(new long[] { 1, 2, 3 }, rows.Select(r => r.RowId).ToArray());
    }

    [TestMethod]
    public void Bootstrap_IdenticalReorderedReferenceProducesKnownZeroDeltas()
    {
        var rows = EvaluationGoldens.Grouped;
        var reference = rows.Reverse().ToArray();
        var snapshot = reference.ToArray();

        var report = PredictorEvaluation.Bootstrap(rows, .5, reference);

        EvaluationGoldens.Report(report, EvaluationGoldens.Defined,
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], paired: true);
        foreach (var interval in report.Intervals)
        {
            Assert.AreEqual(0d, interval.Lower!.Value);
            Assert.AreEqual(0d, interval.Upper!.Value);
        }
        CollectionAssert.AreEqual(snapshot, reference);
        CollectionAssert.AreEqual(new double[] { .2, .4, .8 }, rows.Select(r => r.Probability).ToArray());
    }

    [TestMethod]
    public void Bootstrap_ChangedReferenceProducesKnownNonzeroPairedDeltas()
    {
        var rows = EvaluationGoldens.Grouped;
        Prediction[] reference = [new(3, 20, true, .7), new(2, 10, false, .5), new(1, 10, false, .3)];
        var snapshot = reference.ToArray();
        double hamLossDelta = (-Math.Log(.8) - Math.Log(.6) + Math.Log(.7) + Math.Log(.5)) / 2;
        double spamLossDelta = -Math.Log(.8) + Math.Log(.7);
        double mixedLossDelta = (-2 * Math.Log(.8) - Math.Log(.6) + 2 * Math.Log(.7) + Math.Log(.5)) / 3;
        var lossBounds = EvaluationGoldens.FloorBounds(hamLossDelta, mixedLossDelta, spamLossDelta);
        var brierBounds = EvaluationGoldens.FloorBounds(-.07, (.24 - .43) / 3, -.05);

        var report = PredictorEvaluation.Bootstrap(rows, .5, reference);

        EvaluationGoldens.Report(report, EvaluationGoldens.Defined,
            [0, 0, lossBounds[0], -.07, 0, -.5], [0, 0, lossBounds[1], -.05, 0, -.5], paired: true);
        EvaluationGoldens.NullableNear(brierBounds[0], report.Intervals[3].Lower);
        EvaluationGoldens.NullableNear(brierBounds[1], report.Intervals[3].Upper);
        Assert.IsTrue(report.Intervals[2].Upper < 0);
        CollectionAssert.AreEqual(snapshot, reference);
        CollectionAssert.AreEqual(new double[] { .2, .4, .8 }, rows.Select(r => r.Probability).ToArray());
    }

    [TestMethod]
    public void Bootstrap_SeparateReferenceThresholdControlsPairedDecisionMetrics()
    {
        // The current source has a fourth optional argument not present in research.
        // Each arm's saved validation threshold must drive its decision metrics.
        var rows = EvaluationGoldens.Grouped;
        var reference = rows.Reverse().ToArray();

        var report = PredictorEvaluation.Bootstrap(rows, .5, reference, referenceThreshold: .9);

        EvaluationGoldens.Report(report, EvaluationGoldens.Defined,
            [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], paired: true);
        CollectionAssert.AreEqual(new double[] { .8, .4, .2 }, reference.Select(p => p.Probability).ToArray());
        Assert.AreEqual(233, report.Intervals[4].Undefined);
    }

    [TestMethod]
    [DataRow("ham-only")]
    [DataRow("spam-only")]
    public void Bootstrap_SingleClassUndefinedMetricsHaveNullBounds(string caseId)
    {
        bool spam = caseId == "spam-only";
        Prediction[] rows = [new(1, 10, spam, spam ? .8 : .2), new(2, 20, spam, spam ? .8 : .2)];

        var report = PredictorEvaluation.Bootstrap(rows, .5);

        EvaluationGoldens.Report(report, [0, 0, 1000, 1000, spam ? 1000 : 0, spam ? 0 : 1000],
            [null, null, -Math.Log(.8), .04, spam ? 1 : null, spam ? null : 0],
            [null, null, -Math.Log(.8), .04, spam ? 1 : null, spam ? null : 0]);
        Assert.AreEqual(2, rows.Length);
        Assert.AreEqual(spam ? 2 : 0, rows.Count(p => p.Label));
    }

    [TestMethod]
    [DataRow("empty")]
    [DataRow("duplicate-saved")]
    [DataRow("missing-reference")]
    [DataRow("extra-reference")]
    [DataRow("reference-label")]
    [DataRow("reference-group")]
    [DataRow("probability-nan")]
    [DataRow("probability-positive-infinity")]
    [DataRow("probability-negative-infinity")]
    [DataRow("probability-negative")]
    [DataRow("probability-greater-than-one")]
    [DataRow("threshold-nan")]
    [DataRow("threshold-positive-infinity")]
    [DataRow("threshold-negative-infinity")]
    public void Bootstrap_InvalidSavedRowsOrPairedReferenceIsRejected(string caseId)
    {
        var rows = caseId == "empty" ? Array.Empty<Prediction>() : EvaluationGoldens.Grouped;
        Prediction[]? reference = null;
        double threshold = .5;
        string message;
        if (caseId == "empty") message = "No holdout groups.";
        else if (caseId == "duplicate-saved")
        {
            rows[1] = rows[1] with { RowId = 1 };
            message = "Saved holdout predictions have duplicate IDs.";
        }
        else if (caseId.Contains("reference", StringComparison.Ordinal))
        {
            reference = rows.ToArray();
            if (caseId == "missing-reference") reference = reference[..^1];
            if (caseId == "extra-reference") reference = [.. reference, new(999, 999, false, .4)];
            if (caseId == "reference-label") reference[1] = reference[1] with { Label = true };
            if (caseId == "reference-group") reference[1] = reference[1] with { GroupId = 999 };
            message = "Paired bootstrap rows/groups/labels do not match.";
        }
        else
        {
            if (caseId.StartsWith("probability-", StringComparison.Ordinal))
                rows[1] = rows[1] with { Probability = InvalidNumber(caseId["probability-".Length..]) };
            else threshold = InvalidNumber(caseId["threshold-".Length..]);
            message = "Metrics require nonempty finite probabilities and a finite threshold.";
        }
        var snapshot = rows.ToArray();
        var referenceSnapshot = reference?.ToArray();

        var error = Assert.ThrowsExactly<InvalidDataException>(() => PredictorEvaluation.Bootstrap(rows, threshold, reference));

        Assert.AreEqual(message, error.Message);
        CollectionAssert.AreEqual(snapshot, rows);
        if (reference is not null) CollectionAssert.AreEqual(referenceSnapshot!, reference);
        Assert.AreEqual(caseId == "empty" ? 0 : 3, rows.Length);
    }

    [TestMethod]
    public void Bootstrap_DuplicateReferenceIdsThrowArgumentException()
    {
        var rows = EvaluationGoldens.Grouped;
        var reference = rows.ToArray();
        reference[1] = reference[1] with { RowId = 1 };
        var snapshot = reference.ToArray();

        Assert.ThrowsExactly<ArgumentException>(() => PredictorEvaluation.Bootstrap(rows, .5, reference));

        CollectionAssert.AreEqual(snapshot, reference);
        CollectionAssert.AreEqual(new long[] { 1, 2, 3 }, rows.Select(r => r.RowId).ToArray());
        Assert.AreEqual(3, reference.Length);
    }

    private static Metrics TiedMetrics(double threshold, int tp, int fp, int tn, int fn,
        double? precision, double recall, double fpr) =>
        new(4, 2, 2, 7d / 12, 5d / 8, -Math.Log(.0405) / 4, .27, threshold, tp, fp, tn, fn, precision, recall, fpr);

    private static double InvalidNumber(string caseId) => caseId switch
    {
        "nan" => double.NaN,
        "positive-infinity" => double.PositiveInfinity,
        "negative-infinity" => double.NegativeInfinity,
        "negative" => -double.Epsilon,
        "greater-than-one" => Math.BitIncrement(1d),
        _ => throw new ArgumentOutOfRangeException(nameof(caseId))
    };

    private static double F1(Metrics m) => 2d * m.TruePositive /
        (2 * m.TruePositive + m.FalsePositive + m.FalseNegative);
}
