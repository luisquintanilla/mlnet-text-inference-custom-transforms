using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLNet.TextInference.Onnx;
using MLNet.TextInference.TypedDecisions;

namespace MLNet.TextInference.TypedDecisions.Tests;

[TestClass]
public sealed class TypedDecisionAdapterTests
{
    [TestMethod]
    public void FacadeEstimator_GetOutputSchema_AdvertisesTypedDecisionColumns()
    {
        var options = new OnnxTypedDecisionsOptions
        {
            BundlePath = "not-opened-by-schema-validation",
            Questions =
            [
                DecisionQuestion.Choice("team", "Which team?", ["billing", "support"])
            ]
        };
        var estimator = new OnnxTypedDecisionsEstimator(new MLContext(), options);
        var input = new SchemaShape(
        [
            CreateColumn("State", TextDataViewType.Instance)
        ]);

        var output = estimator.GetOutputSchema(input);

        CollectionAssert.IsSubsetOf(
            new[]
            {
                "State",
                options.ResultsColumnName,
                options.ChoiceColumnName,
                options.ScoreColumnName,
                options.ProbabilityTrueColumnName,
                options.ConfidenceColumnName,
                options.ActionProbabilityColumnName
            },
            output.Select(column => column.Name).ToArray());
    }

    [TestMethod]
    public void FacadeEstimator_Fit_ValidatesInputSchemaBeforeOpeningBundle()
    {
        var options = new OnnxTypedDecisionsOptions
        {
            BundlePath = "missing-bundle",
            Questions =
            [
                DecisionQuestion.Noul("risk", "Will the customer churn?")
            ]
        };
        var estimator = new OnnxTypedDecisionsEstimator(new MLContext(), options);
        var data = new MLContext().Data.LoadFromEnumerable(
            new[] { new MissingStateRow { Value = "state" } });

        Assert.ThrowsException<ArgumentException>(() => estimator.Fit(data));
    }

    [TestMethod]
    public void StagedEstimators_PropagateIntermediateTextColumns()
    {
        var ml = new MLContext();
        var prepare = new PrepareDecisionInputsEstimator(
            ml,
            new PrepareDecisionInputsOptions
            {
                BundlePath = "not-opened-by-schema-validation",
                Questions = [DecisionQuestion.Noul("risk", "Will the customer churn?")]
            });
        var score = new ScoreOnnxDecisionModelEstimator(
            ml,
            new ScoreOnnxDecisionModelOptions
            {
                BundlePath = "not-opened-by-schema-validation"
            });
        var input = new SchemaShape(
        [
            CreateColumn("State", TextDataViewType.Instance)
        ]);

        var prepared = prepare.GetOutputSchema(input);
        var scored = score.GetOutputSchema(prepared);

        Assert.IsTrue(prepared.Any(column => column.Name == "PreparedDecisionInputs"));
        Assert.IsTrue(scored.Any(column => column.Name == "ScoredDecisionOutputs"));
    }

    private static SchemaShape.Column CreateColumn(string name, DataViewType type)
    {
        var constructor = typeof(SchemaShape.Column).GetConstructors(
            BindingFlags.NonPublic | BindingFlags.Instance)[0];
        return (SchemaShape.Column)constructor.Invoke([
            name,
            SchemaShape.Column.VectorKind.Scalar,
            type,
            false,
            (SchemaShape?)null
        ]);
    }

    private sealed class MissingStateRow
    {
        public string Value { get; init; } = string.Empty;
    }
}
