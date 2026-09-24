using MLNet.TextInference.Onnx;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MLNet.TextInference.TypedDecisions.Tests;

[TestClass]
public sealed class StableSoftmaxTests
{
    [TestMethod]
    public void ExtremeFiniteLogitsRemainFiniteAndNormalized()
    {
        var probabilities = StableSoftmax.Create([10000f, 9999f, -10000f], 3);

        Assert.IsTrue(probabilities.All(float.IsFinite));
        Assert.AreEqual(1f, probabilities.Sum(), 1e-6f);
        Assert.IsTrue(probabilities[0] > probabilities[1]);
        Assert.AreEqual(0f, probabilities[2]);
    }

    [TestMethod]
    public void ValidSliceClearsPaddedValues()
    {
        var probabilities = StableSoftmax.Create([2f, 0f, 10000f, -10000f], 2);

        Assert.AreEqual(1f, probabilities[0] + probabilities[1], 1e-6f);
        Assert.AreEqual(0f, probabilities[2]);
        Assert.AreEqual(0f, probabilities[3]);
    }

    [TestMethod]
    public void TiesPreserveEqualProbabilitiesAndFirstMaximum()
    {
        var probabilities = StableSoftmax.Create([3f, 3f, 0f], 3);

        Assert.AreEqual(probabilities[0], probabilities[1], 1e-6f);
        Assert.AreEqual(0, StableSoftmax.IndexOfMax(probabilities));
    }

    [TestMethod]
    public void NonFiniteLogitIsRejected()
    {
        Assert.ThrowsExactly<ArgumentException>(() =>
            StableSoftmax.Create([0f, float.PositiveInfinity], 2));
    }

    [TestMethod]
    public void InvalidTemperatureIsRejected()
    {
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() =>
            StableSoftmax.Create([0f, 1f], 2, 0f));
    }
}
