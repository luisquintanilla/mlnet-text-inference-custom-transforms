using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class ConsumerFitControlsTests
{
    [TestMethod]
    [TestCategory("ControlledFit")]
    public void BoundedAuthoredFits_PreserveActualLearnerTracesGridAndStandardReplayWithoutHoldout()
    {
        using var temp = new TempDirectory();
        var legacy = StudyFixture.Authored();
        using var compact = StudyData.FromLegacy(legacy);
        string output = temp.FilePath("bounded-fits");
        ConsumerFitControls.Verify(legacy, compact, output);
        using var receipt = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(output, "fit-control.receipt.json")));
        var root = receipt.RootElement;
        Assert.AreEqual("BOUNDED_INDEPENDENT_FIT_PARITY_PASS", root.GetProperty("status").GetString());
        Assert.AreEqual(33, root.GetProperty("trainingRows").GetInt32());
        Assert.AreEqual(4, root.GetProperty("validationRows").GetInt32());
        Assert.IsFalse(root.GetProperty("holdoutOpened").GetBoolean());
        Assert.IsFalse(root.GetProperty("fullStudyReady").GetBoolean());
        Assert.AreEqual(9, root.GetProperty("controls").GetArrayLength());
        foreach (var candidate in root.GetProperty("controls").EnumerateArray())
        {
            Assert.IsTrue(candidate.GetProperty("independentFitReplayPass").GetBoolean());
            Assert.IsTrue(candidate.GetProperty("instrumentationDidNotChangePredictions").GetBoolean());
            var source = candidate.GetProperty("tracedCompact").GetProperty("sourceTrace");
            var learner = candidate.GetProperty("tracedCompact").GetProperty("learnerTrace");
            Assert.IsTrue(source.GetArrayLength() > 0);
            Assert.IsTrue(learner.GetArrayLength() > 0);
        }
        Assert.AreEqual(6, System.IO.Directory.GetFiles(output, "*.mlnet").Length);
        Assert.AreEqual(6, System.IO.Directory.GetFiles(output, "*.mlnet.replay.json").Length);
        Assert.IsFalse(File.Exists(Path.Combine(output, "fit-control.failure.json")));
    }
}
