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
    public void FacadeEstimator_GetOutputSchema_AdvertisesPerQuestionNativeOutputs()
    {
        var options = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = "not-opened-by-schema-validation",
            Questions =
            [
                DecisionQuestion.Choice("team", "Which team?", ["billing", "support"]),
                DecisionQuestion.Noul("risk", "Will the customer churn?")
            ]
        };
        var estimator = new OnnxTypedDecisionsEstimator(new MLContext(), options);
        var input = new SchemaShape([CreateColumn("State", TextDataViewType.Instance)]);

        var output = estimator.GetOutputSchema(input);
        var names = output.Select(column => column.Name).ToHashSet(StringComparer.Ordinal);

        CollectionAssert.IsSubsetOf(
            new[]
            {
                "State",
                "DecisionResults",
                "Decision_team_PredictedLabel",
                "Decision_team_Probabilities",
                "Decision_team_Confidence",
                "Decision_team_ActionProbability",
                "Decision_risk_PredictedLabel",
                "Decision_risk_Probability",
                "Decision_risk_Confidence",
                "Decision_risk_ActionProbability"
            },
            names.ToArray());
    }

    [TestMethod]
    public void FacadeEstimator_Fit_ValidatesInputSchemaBeforeOpeningBundle()
    {
        var options = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = "missing-bundle",
            Questions = [DecisionQuestion.Noul("risk", "Will the customer churn?")]
        };
        var estimator = new OnnxTypedDecisionsEstimator(new MLContext(), options);
        var data = new MLContext().Data.LoadFromEnumerable(
            new[] { new MissingStateRow { Value = "state" } });

        Assert.ThrowsException<ArgumentException>(() => estimator.Fit(data));
    }

    [TestMethod]
    public void StagedEstimators_AdvertiseNativeTensorColumns()
    {
        var ml = new MLContext();
        var prepare = new DecisionInputPreparationEstimator(
            ml,
            new DecisionInputPreparationOptions
            {
                ModelAssetsPath = "not-opened-by-schema-validation",
                Questions = [DecisionQuestion.Noul("risk", "Will the customer churn?")]
            });
        var score = new OnnxDecisionModelScorerEstimator(
            ml,
            new OnnxDecisionModelScorerOptions
            {
                ModelAssetsPath = "not-opened-by-schema-validation"
            });
        var input = new SchemaShape([CreateColumn("State", TextDataViewType.Instance)]);

        var prepared = prepare.GetOutputSchema(input);
        var scored = score.GetOutputSchema(prepared);
        var preparedNames = prepared.Select(column => column.Name).ToHashSet(StringComparer.Ordinal);
        var scoredNames = scored.Select(column => column.Name).ToHashSet(StringComparer.Ordinal);

        CollectionAssert.IsSubsetOf(
            new[]
            {
                "DecisionInputIds",
                "DecisionAttentionMask",
                "DecisionMarkerPositions",
                "DecisionMarkerMask",
                "DecisionQuestionTypes",
                "DecisionBatchSize",
                "DecisionSequenceLength",
                "DecisionMarkerWidth"
            },
            preparedNames.ToArray());
        CollectionAssert.IsSubsetOf(
            new[] { "DecisionLogits", "DecisionActionProbabilities" },
            scoredNames.ToArray());
    }

    [TestMethod]
    public void DecodeEstimator_AdvertisesNativeScoredInputsAndPerQuestionOutputs()
    {
        var ml = new MLContext();
        var options = new DecisionDecodingOptions
        {
            ModelAssetsPath = "not-opened-by-schema-validation",
            Questions = [DecisionQuestion.Score("urgency", "How urgent?", ["low", "high"])]
        };
        var input = new SchemaShape(
        [
            CreateColumn("DecisionInputIds", new VectorDataViewType(NumberDataViewType.Int64)),
            CreateColumn("DecisionAttentionMask", new VectorDataViewType(NumberDataViewType.Int64)),
            CreateColumn("DecisionMarkerPositions", new VectorDataViewType(NumberDataViewType.Int64)),
            CreateColumn("DecisionMarkerMask", new VectorDataViewType(BooleanDataViewType.Instance)),
            CreateColumn("DecisionQuestionTypes", new VectorDataViewType(NumberDataViewType.Int64)),
            CreateColumn("DecisionBatchSize", NumberDataViewType.Int32),
            CreateColumn("DecisionSequenceLength", NumberDataViewType.Int32),
            CreateColumn("DecisionMarkerWidth", NumberDataViewType.Int32),
            CreateColumn("DecisionLogits", new VectorDataViewType(NumberDataViewType.Single)),
            CreateColumn("DecisionActionProbabilities", new VectorDataViewType(NumberDataViewType.Single))
        ]);

        var output = new DecisionDecodingEstimator(ml, options).GetOutputSchema(input);
        var names = output.Select(column => column.Name).ToHashSet(StringComparer.Ordinal);

        CollectionAssert.IsSubsetOf(
            new[]
            {
                "DecisionResults",
                "Decision_urgency_Score",
                "Decision_urgency_Probabilities",
                "Decision_urgency_Confidence",
                "Decision_urgency_ActionProbability"
            },
            names.ToArray());
    }

    [TestMethod]
    public void Options_RejectDuplicateQuestionIds()
    {
        var options = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = "fixture",
            Questions =
            [
                DecisionQuestion.Choice("same", "First", ["a", "b"]),
                DecisionQuestion.Noul("same", "Second")
            ]
        };

        Assert.ThrowsException<ArgumentException>(() =>
            new OnnxTypedDecisionsEstimator(new MLContext(), options));
    }

    [TestMethod]
    public void FacadeEstimator_MultipleQuestionsOfTheSameTypeGetDistinctColumns()
    {
        var options = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = "not-opened-by-schema-validation",
            Questions =
            [
                DecisionQuestion.Choice("primary", "Which primary team?", ["billing", "support"]),
                DecisionQuestion.Choice("secondary", "Which backup team?", ["billing", "support"]),
                DecisionQuestion.Score("initial", "What is the initial score?", ["low", "high"]),
                DecisionQuestion.Score("final", "What is the final score?", ["low", "high"])
            ]
        };

        var schema = new OnnxTypedDecisionsEstimator(new MLContext(), options)
            .GetOutputSchema(new SchemaShape([CreateColumn("State", TextDataViewType.Instance)]));
        var names = schema.Select(column => column.Name).ToHashSet(StringComparer.Ordinal);

        CollectionAssert.IsSubsetOf(
            new[]
            {
                "Decision_primary_PredictedLabel",
                "Decision_secondary_PredictedLabel",
                "Decision_initial_Score",
                "Decision_final_Score",
                "Decision_primary_Probabilities",
                "Decision_secondary_Probabilities",
                "Decision_initial_Probabilities",
                "Decision_final_Probabilities"
            },
            names.ToArray());
    }

    private static SchemaShape.Column CreateColumn(string name, DataViewType type)
    {
        var constructor = typeof(SchemaShape.Column).GetConstructors(
            BindingFlags.NonPublic | BindingFlags.Instance)[0];
        return (SchemaShape.Column)constructor.Invoke([
            name,
            type is VectorDataViewType
                ? SchemaShape.Column.VectorKind.Vector
                : SchemaShape.Column.VectorKind.Scalar,
            type is VectorDataViewType vector ? vector.ItemType : type,
            false,
            (SchemaShape?)null
        ]);
    }

    private sealed class MissingStateRow
    {
        public string Value { get; init; } = string.Empty;
    }

}
