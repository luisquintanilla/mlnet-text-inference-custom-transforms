using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class FeatureContractTests
{
    [TestMethod]
    public void Questions_MinimalFrozenDocumentMatchesProjectionAndQuestionContract()
    {
        byte[] bytes = QuestionFixtures.Bytes();
        byte[] snapshot = bytes.ToArray();

        FeatureContract.ValidateQuestions(bytes);

        Assert.AreEqual(10, FeatureContract.Width);
        CollectionAssert.AreEqual(QuestionFixtures.Projection, FeatureContract.Projection);
        Assert.AreEqual("Arrow float64 probabilities -> ML.NET float32, checked finite [0,1], original Arrow unchanged",
            FeatureContract.Conversion);
        Assert.IsFalse(FeatureContract.Projection.Contains("spam_baseline"));
        using var document = JsonDocument.Parse(bytes);
        var questions = document.RootElement.GetProperty("questions").EnumerateArray().ToArray();
        CollectionAssert.AreEqual(new[] { "commercial_solicitation", "requested_contact_action", "time_pressure", "message_purpose", "spam_baseline" },
            questions.Select(q => q.GetProperty("id").GetString()).ToArray());
        CollectionAssert.AreEqual(new[] { "binary", "binary", "score", "choice", "binary" },
            questions.Select(q => q.GetProperty("kind").GetString()).ToArray());
        CollectionAssert.AreEqual(snapshot, bytes);
    }

    [TestMethod]
    [DataRow("schema-version")]
    [DataRow("question-count")]
    [DataRow("projection-count")]
    [DataRow("projection-order")]
    [DataRow("baseline-intrusion")]
    [DataRow("question-id")]
    [DataRow("question-kind")]
    [DataRow("question-order")]
    [DataRow("rubric-order")]
    [DataRow("candidate-order")]
    public void Questions_IndependentSupportedShapeMutationsAreRejected(string mutation)
    {
        byte[] baseline = QuestionFixtures.Bytes();
        byte[] changed = QuestionFixtures.Bytes(QuestionFixtures.Mutate(mutation));
        byte[] snapshot = changed.ToArray();
        Assert.IsFalse(baseline.SequenceEqual(changed));

        var error = Assert.ThrowsExactly<InvalidDataException>(() => FeatureContract.ValidateQuestions(changed));

        Assert.IsTrue(error.Message.Contains(mutation is "rubric-order" or "candidate-order"
            ? "candidate/rubric" : mutation.StartsWith("question-", StringComparison.Ordinal) && mutation != "question-count"
                ? "identity/order/kind" : "projection", StringComparison.Ordinal));
        CollectionAssert.AreEqual(snapshot, changed);
        CollectionAssert.AreEqual(QuestionFixtures.Projection, FeatureContract.Projection);
        FeatureContract.ValidateQuestions(baseline);
    }

    [TestMethod]
    public void ConvertProbabilities_CastsInclusiveEndpointsWithoutMutatingInput()
    {
        double[] source = [0, 1, .1, .2, .3, .4, .5, .6, .7, .8];
        double[] snapshot = source.ToArray();

        float[] first = FeatureContract.ConvertProbabilities(source);
        float[] second = FeatureContract.ConvertProbabilities(source);

        CollectionAssert.AreEqual(new float[] { 0, 1, .1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f }, first);
        CollectionAssert.AreEqual(snapshot, source);
        Assert.AreNotSame(first, second);
        first[2] = .9f;
        Assert.AreEqual(.1f, second[2]);
        Assert.AreEqual(.1, source[2]);
        Assert.AreEqual(10, second.Length);
    }

    [TestMethod]
    [DataRow("width-nine")]
    [DataRow("width-eleven")]
    [DataRow("nan")]
    [DataRow("positive-infinity")]
    [DataRow("negative-infinity")]
    [DataRow("negative")]
    [DataRow("greater-than-one")]
    public void ConvertProbabilities_InvalidWidthOrValueIsRejected(string caseId)
    {
        double[] values = Enumerable.Repeat(.25, caseId == "width-nine" ? 9 : caseId == "width-eleven" ? 11 : 10).ToArray();
        if (!caseId.StartsWith("width-", StringComparison.Ordinal))
            values[4] = caseId switch
            {
                "nan" => double.NaN,
                "positive-infinity" => double.PositiveInfinity,
                "negative-infinity" => double.NegativeInfinity,
                "negative" => -double.Epsilon,
                "greater-than-one" => Math.BitIncrement(1d),
                _ => throw new ArgumentOutOfRangeException(nameof(caseId))
            };
        var snapshot = values.ToArray();

        var error = Assert.ThrowsExactly<InvalidDataException>(() => FeatureContract.ConvertProbabilities(values));

        Assert.AreEqual("Expected ten finite probabilities in [0,1].", error.Message);
        CollectionAssert.AreEqual(snapshot, values);
        Assert.AreEqual(.25, values[0]);
    }

    [TestMethod]
    [DataRow("exact", "fixture-contract-v1", "fixture-contract-v1", true)]
    [DataRow("different", "fixture-contract-v2", "fixture-contract-v1", false)]
    [DataRow("case", "Fixture-contract-v1", "fixture-contract-v1", false)]
    [DataRow("padded", " fixture-contract-v1 ", "fixture-contract-v1", false)]
    [DataRow("blank-expected", "", "", false)]
    [DataRow("whitespace-expected", " \t", " \t", false)]
    public void RequireIdentity_ExactNonblankOrdinalIdentityRequired(string caseId, string actual, string expected, bool accepted)
    {
        string snapshot = actual;
        if (accepted)
        {
            FeatureContract.RequireIdentity(actual, expected);
            Assert.AreEqual("fixture-contract-v1", actual, caseId);
        }
        else
        {
            var error = Assert.ThrowsExactly<InvalidDataException>(() => FeatureContract.RequireIdentity(actual, expected));
            Assert.AreEqual("External producer feature-contract identity mismatch.", error.Message, caseId);
        }
        Assert.AreEqual(snapshot, actual);
        CollectionAssert.AreEqual(QuestionFixtures.Projection, FeatureContract.Projection);
    }

    [TestMethod]
    public void RequireIdentity_UnicodeNormalizationIsNotApplied()
    {
        const string composed = "caf\u00e9-contract";
        const string decomposed = "cafe\u0301-contract";
        Assert.AreEqual(composed, decomposed.Normalize());

        var error = Assert.ThrowsExactly<InvalidDataException>(() => FeatureContract.RequireIdentity(decomposed, composed));

        Assert.AreEqual("External producer feature-contract identity mismatch.", error.Message);
        Assert.AreEqual("cafe\u0301-contract", decomposed);
        Assert.AreNotEqual(composed.Length, decomposed.Length);
        FeatureContract.RequireIdentity(composed, composed);
    }
}
