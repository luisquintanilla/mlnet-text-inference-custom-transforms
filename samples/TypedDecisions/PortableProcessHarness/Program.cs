// Tracked .NET 10 file-based writer/reader harness for portable typed-decision
// acceptance. It intentionally receives a caller-owned fixture path and never
// deletes or modifies that path.
#:project ..\..\..\src\MLNet.TextInference.Onnx\MLNet.TextInference.Onnx.csproj
#:package Microsoft.ML@5.0.0
#:package Microsoft.ML.OnnxRuntime@1.24.2
#:property PublishAot=false

using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.TextInference.Onnx;
using MLNet.TextInference.TypedDecisions;
using static MLNet.TextInference.TypedDecisions.DecisionQuestion;

var mode = GetOption("--mode") ?? "reader";
var kind = GetOption("--kind") ?? "facade";
var assets = GetOption("--model-assets");
var archive = GetOption("--portable-path")
    ?? throw new ArgumentException("--portable-path is required.");
var ml = new MLContext(seed: 1);
var states = new[]
{
    "portable process state",
    "portable process state with additional context"
};
var data = ml.Data.LoadFromEnumerable(
    states.Select(static state => new StateRow { Evidence = state }));

if (mode.Equals("writer", StringComparison.OrdinalIgnoreCase))
{
    if (string.IsNullOrWhiteSpace(assets))
        throw new ArgumentException("--model-assets is required for writer mode.");
    var result = WriteArtifact(ml, data, kind, assets, archive);
    Console.WriteLine(JsonSerializer.Serialize(result, JsonOptions()));
}
else if (mode.Equals("reader", StringComparison.OrdinalIgnoreCase))
{
    var result = ReadArtifact(ml, data, states, kind, archive);
    Console.WriteLine(JsonSerializer.Serialize(result, JsonOptions()));
}
else
{
    throw new ArgumentException($"Unknown mode '{mode}'.");
}

static object WriteArtifact(
    MLContext ml,
    IDataView data,
    string kind,
    string assets,
    string archive)
{
    switch (kind.ToLowerInvariant())
    {
        case "facade":
        {
            using var transformer = ml.Transforms.OnnxTypedDecisions(
                FacadeOptions(assets, Questions()))
                .Fit(data);
            transformer.Save(archive);
            return SnapshotFacade(ml, data, transformer);
        }
        case "stages":
        {
            var stages = ml.Transforms.PrepareDecisionInputs(
                    PreparationOptions(assets, Questions()))
                .Append(ml.Transforms.ScoreOnnxDecisionModel(
                    ScoringOptions(assets)))
                .Append(ml.Transforms.DecodeDecisions(
                    DecodingOptions(assets, Questions())));
            var transformer = stages.Fit(data);
            try
            {
                var typed = (TransformerChain<DecisionDecodingTransformer>)transformer;
                TypedDecisionPortableModel.SavePipeline(typed, archive);
                return SnapshotStages(ml, data, typed);
            }
            finally
            {
                (transformer as IDisposable)?.Dispose();
            }
        }
        case "composed":
        {
            var first = FacadeOptions(assets, Questions());
            var second = new OnnxTypedDecisionsOptions
            {
                ModelAssetsPath = assets,
                Questions = BinaryQuestions(),
                StateColumnName = "Evidence",
                OutputPrefix = "Appended_",
                ResultsColumnName = "AppendedDecisionResults",
                BatchSize = 2
            };
            var transformer = ml.Transforms.OnnxTypedDecisions(first)
                .AppendOnnxTypedDecisions(ml, second)
                .Fit(data);
            try
            {
                var typed = (TransformerChain<OnnxTypedDecisionsTransformer>)transformer;
                TypedDecisionPortableModel.SavePipeline(typed, archive);
                return SnapshotComposed(ml, data, null, typed);
            }
            finally
            {
                (transformer as IDisposable)?.Dispose();
            }
        }
        default:
            throw new ArgumentException($"Unknown artifact kind '{kind}'.");
    }
}

static object ReadArtifact(
    MLContext ml,
    IDataView data,
    IReadOnlyList<string> states,
    string kind,
    string archive)
{
    switch (kind.ToLowerInvariant())
    {
        case "facade":
        {
            using var transformer = OnnxTypedDecisionsTransformer.Load(ml, archive);
            return SnapshotFacade(ml, data, transformer);
        }
        case "stages":
        {
            using var transformer = TypedDecisionPortableModel.LoadPipeline(ml, archive);
            return SnapshotStages(ml, data, transformer);
        }
        case "composed":
        {
            using var transformer = TypedDecisionPortableModel.LoadPipeline(ml, archive);
            return SnapshotComposed(ml, data, states, transformer);
        }
        default:
            throw new ArgumentException($"Unknown artifact kind '{kind}'.");
    }
}

static object SnapshotFacade(
    MLContext ml,
    IDataView data,
    OnnxTypedDecisionsTransformer transformer)
{
    var states = data.GetColumn<string>("Evidence").ToArray();
    var direct = states.Select(transformer.Infer).Select(SnapshotResponse).ToArray();
    using var engine = ml.Model.CreatePredictionEngine<StateRow, FacadeRow>(
        transformer,
        new PredictionEngineOptions { OwnsTransformer = false });
    var predictionEngine = states
        .Select(state => SnapshotFacadeRow(engine.Predict(new StateRow { Evidence = state })))
        .ToArray();
    return new
    {
        kind = "facade",
        direct,
        prediction_engine = predictionEngine,
        dataview = SnapshotDataView(transformer.Transform(data))
    };
}

static object SnapshotStages(
    MLContext ml,
    IDataView data,
    ITransformer transformer)
{
    using var engine = ml.Model.CreatePredictionEngine<StateRow, StageRow>(
        transformer,
        new PredictionEngineOptions { OwnsTransformer = false });
    var states = data.GetColumn<string>("Evidence").ToArray();
    return new
    {
        kind = "stages",
        prediction_engine = states
            .Select(state => SnapshotStageRow(engine.Predict(new StateRow { Evidence = state })))
            .ToArray(),
        dataview = SnapshotDataView(transformer.Transform(data))
    };
}

static object SnapshotComposed(
    MLContext ml,
    IDataView data,
    IReadOnlyList<string>? states,
    ITransformer transformer)
{
    states ??= data.GetColumn<string>("Evidence").ToArray();
    using var engine = ml.Model.CreatePredictionEngine<StateRow, ComposedRow>(
        transformer,
        new PredictionEngineOptions { OwnsTransformer = false });
    return new
    {
        kind = "composed",
        prediction_engine = states
            .Select(state => SnapshotComposedRow(
                engine.Predict(new StateRow { Evidence = state })))
            .ToArray(),
        dataview = SnapshotDataView(transformer.Transform(data))
    };
}

static object SnapshotResponse(DecisionResponse response)
    => new
    {
        input_tokens = response.InputTokenCount,
        results = response.Results.Select(SnapshotDecisionResult).ToArray()
    };

static object SnapshotDecisionResult(DecisionResult result)
    => result switch
    {
        ChoiceDecisionResult choice => (object)new
        {
            id = choice.Id,
            type = "choice",
            confidence = choice.Confidence,
            action_probability = choice.ActionProbability,
            labels = choice.Distribution.Labels.ToArray(),
            probabilities = choice.Distribution.Probabilities.ToArray(),
            choice = choice.Choice
        },
        ScoreDecisionResult score => new
        {
            id = score.Id,
            type = "score",
            confidence = score.Confidence,
            action_probability = score.ActionProbability,
            labels = score.Distribution.Labels.ToArray(),
            probabilities = score.Distribution.Probabilities.ToArray(),
            score = score.Score,
            legend = score.Legend
        },
        NoulDecisionResult noul => new
        {
            id = noul.Id,
            type = "noul",
            confidence = noul.Confidence,
            action_probability = noul.ActionProbability,
            labels = noul.Distribution.Labels.ToArray(),
            probabilities = noul.Distribution.Probabilities.ToArray(),
            value = noul.Value,
            probability_true = noul.ProbabilityTrue
        },
        _ => throw new InvalidDataException(
            $"Unsupported decision result '{result.GetType().FullName}'.")
    };

static object SnapshotFacadeRow(FacadeRow row)
    => new
    {
        row.Portable_priority_PredictedLabel,
        priority_probabilities = row.Portable_priority_Probabilities.DenseValues().ToArray(),
        row.Portable_priority_Confidence,
        row.Portable_priority_ActionProbability,
        row.Portable_quality_Score,
        quality_probabilities = row.Portable_quality_Probabilities.DenseValues().ToArray(),
        row.Portable_quality_Confidence,
        row.Portable_quality_ActionProbability,
        row.Portable_actionable_PredictedLabel,
        row.Portable_actionable_Probability,
        row.Portable_actionable_Confidence,
        row.Portable_actionable_ActionProbability,
        row.PortableResults
    };

static object SnapshotStageRow(StageRow row)
    => new
    {
        input_ids = row.PortableInputIds.DenseValues().ToArray(),
        attention_mask = row.PortableAttentionMask.DenseValues().ToArray(),
        marker_positions = row.PortableMarkerPositions.DenseValues().ToArray(),
        marker_mask = row.PortableMarkerMask.DenseValues().ToArray(),
        question_types = row.PortableQuestionTypes.DenseValues().ToArray(),
        row.PortableBatchSize,
        row.PortableSequenceLength,
        row.PortableMarkerWidth,
        logits = row.PortableLogits.DenseValues().ToArray(),
        action_probabilities = row.PortableActionProbabilities.DenseValues().ToArray(),
        row.Portable_priority_PredictedLabel,
        priority_probabilities = row.Portable_priority_Probabilities.DenseValues().ToArray(),
        row.Portable_priority_Confidence,
        row.Portable_priority_ActionProbability,
        row.Portable_quality_Score,
        quality_probabilities = row.Portable_quality_Probabilities.DenseValues().ToArray(),
        row.Portable_quality_Confidence,
        row.Portable_quality_ActionProbability,
        row.Portable_actionable_PredictedLabel,
        row.Portable_actionable_Probability,
        row.Portable_actionable_Confidence,
        row.Portable_actionable_ActionProbability,
        row.PortableResults
    };

static object SnapshotComposedRow(ComposedRow row)
    => new
    {
        first = SnapshotFacadeRow(row),
        appended = new
        {
            row.Appended_binary_PredictedLabel,
            probabilities = row.Appended_binary_Probabilities.DenseValues().ToArray(),
            row.Appended_binary_Confidence,
            row.Appended_binary_ActionProbability,
            row.AppendedDecisionResults
        }
    };

static object SnapshotDataView(IDataView view)
{
    var columns = view.Schema.Select(static column => new
    {
        name = column.Name,
        type = column.Type.ToString(),
        hidden = column.IsHidden,
        dimensions = (column.Type as VectorDataViewType)?.Dimensions.ToArray(),
        slot_names = ReadSlotNames(column)
    }).ToArray();
    var rows = new List<IReadOnlyDictionary<string, object?>>();
    using var cursor = view.GetRowCursor(view.Schema);
    var getters = view.Schema
        .Select(column => (column.Name, Getter: CreateGetter(cursor, column)))
        .ToArray();
    while (cursor.MoveNext())
    {
        var row = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var getter in getters)
            row[getter.Name] = getter.Getter();
        rows.Add(row);
    }
    return new { columns, rows };
}

static Func<object?> CreateGetter(DataViewRowCursor cursor, DataViewSchema.Column column)
{
    if (column.Type.RawType == typeof(string))
    {
        var getter = cursor.GetGetter<string>(column);
        return () =>
        {
            string value = string.Empty;
            getter(ref value);
            return value;
        };
    }
    if (column.Type.RawType == typeof(ReadOnlyMemory<char>))
    {
        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(column);
        return () =>
        {
            ReadOnlyMemory<char> value = default;
            getter(ref value);
            return value.ToString();
        };
    }
    if (column.Type.RawType == typeof(bool))
    {
        var getter = cursor.GetGetter<bool>(column);
        return () =>
        {
            var value = false;
            getter(ref value);
            return value;
        };
    }
    if (column.Type.RawType == typeof(int))
    {
        var getter = cursor.GetGetter<int>(column);
        return () =>
        {
            var value = 0;
            getter(ref value);
            return value;
        };
    }
    if (column.Type.RawType == typeof(float))
    {
        var getter = cursor.GetGetter<float>(column);
        return () =>
        {
            var value = 0f;
            getter(ref value);
            return value;
        };
    }
    if (column.Type.RawType == typeof(VBuffer<float>))
    {
        var getter = cursor.GetGetter<VBuffer<float>>(column);
        return () =>
        {
            VBuffer<float> value = default;
            getter(ref value);
            return Normalize(value);
        };
    }
    if (column.Type.RawType == typeof(VBuffer<long>))
    {
        var getter = cursor.GetGetter<VBuffer<long>>(column);
        return () =>
        {
            VBuffer<long> value = default;
            getter(ref value);
            return Normalize(value);
        };
    }
    if (column.Type.RawType == typeof(VBuffer<bool>))
    {
        var getter = cursor.GetGetter<VBuffer<bool>>(column);
        return () =>
        {
            VBuffer<bool> value = default;
            getter(ref value);
            return Normalize(value);
        };
    }
    throw new NotSupportedException(
        $"Portable process snapshot does not support '{column.Type.RawType}'.");
}

static object? Normalize<T>(T value)
{
    return value switch
    {
        VBuffer<float> floats => new
        {
            dimensions = floats.Length,
            values = floats.DenseValues().ToArray()
        },
        VBuffer<long> longs => new
        {
            dimensions = longs.Length,
            values = longs.DenseValues().ToArray()
        },
        VBuffer<bool> booleans => new
        {
            dimensions = booleans.Length,
            values = booleans.DenseValues().ToArray()
        },
        VBuffer<ReadOnlyMemory<char>> text => new
        {
            dimensions = text.Length,
            values = text.DenseValues().Select(static item => item.ToString()).ToArray()
        },
        _ => value
    };
}

static string[]? ReadSlotNames(DataViewSchema.Column column)
{
    if (!column.Annotations.Schema.Any(annotation => annotation.Name == "SlotNames"))
        return null;
    var annotation = column.Annotations.Schema["SlotNames"];
    var getter = column.Annotations.GetGetter<VBuffer<ReadOnlyMemory<char>>>(annotation);
    VBuffer<ReadOnlyMemory<char>> values = default;
    getter(ref values);
    return values.DenseValues().Select(static value => value.ToString()).ToArray();
}

static IReadOnlyList<DecisionQuestion> Questions() =>
[
    Choice("priority", "How urgent?", new Dictionary<string, string?>
    {
        ["zebra"] = null, ["!"] = string.Empty, ["alpha"] = "explicit"
    }),
    Score("quality", "How strong?", ["weak", "moderate", "strong"]),
    Noul("actionable", "Can it be acted on?")
];

static IReadOnlyList<DecisionQuestion> BinaryQuestions() =>
[
    Choice("binary", "Is it binary?", new Dictionary<string, string?>
    {
        ["no"] = "no", ["yes"] = "yes"
    })
];

static OnnxTypedDecisionsOptions FacadeOptions(
    string assets,
    IReadOnlyList<DecisionQuestion> questions)
    => new()
    {
        ModelAssetsPath = assets,
        Questions = questions,
        StateColumnName = "Evidence",
        OutputPrefix = "Portable_",
        ResultsColumnName = "PortableResults",
        BatchSize = 1
    };

static DecisionInputPreparationOptions PreparationOptions(
    string assets,
    IReadOnlyList<DecisionQuestion> questions)
    => new()
    {
        ModelAssetsPath = assets,
        Questions = questions,
        StateColumnName = "Evidence",
        InputIdsColumnName = "PortableInputIds",
        AttentionMaskColumnName = "PortableAttentionMask",
        MarkerPositionsColumnName = "PortableMarkerPositions",
        MarkerMaskColumnName = "PortableMarkerMask",
        QuestionTypesColumnName = "PortableQuestionTypes",
        BatchSizeColumnName = "PortableBatchSize",
        SequenceLengthColumnName = "PortableSequenceLength",
        MarkerWidthColumnName = "PortableMarkerWidth"
    };

static OnnxDecisionModelScorerOptions ScoringOptions(string assets)
    => new()
    {
        ModelAssetsPath = assets,
        InputIdsColumnName = "PortableInputIds",
        AttentionMaskColumnName = "PortableAttentionMask",
        MarkerPositionsColumnName = "PortableMarkerPositions",
        MarkerMaskColumnName = "PortableMarkerMask",
        QuestionTypesColumnName = "PortableQuestionTypes",
        BatchSizeColumnName = "PortableBatchSize",
        SequenceLengthColumnName = "PortableSequenceLength",
        MarkerWidthColumnName = "PortableMarkerWidth",
        LogitsColumnName = "PortableLogits",
        ActionProbabilitiesColumnName = "PortableActionProbabilities",
        BatchSize = 2
    };

static DecisionDecodingOptions DecodingOptions(
    string assets,
    IReadOnlyList<DecisionQuestion> questions)
    => new()
    {
        ModelAssetsPath = assets,
        Questions = questions,
        InputIdsColumnName = "PortableInputIds",
        AttentionMaskColumnName = "PortableAttentionMask",
        MarkerPositionsColumnName = "PortableMarkerPositions",
        MarkerMaskColumnName = "PortableMarkerMask",
        QuestionTypesColumnName = "PortableQuestionTypes",
        BatchSizeColumnName = "PortableBatchSize",
        SequenceLengthColumnName = "PortableSequenceLength",
        MarkerWidthColumnName = "PortableMarkerWidth",
        LogitsColumnName = "PortableLogits",
        ActionProbabilitiesColumnName = "PortableActionProbabilities",
        OutputPrefix = "Portable_",
        ResultsColumnName = "PortableResults"
    };

string? GetOption(string name)
{
    var index = Array.IndexOf(args, name);
    return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
}

static JsonSerializerOptions JsonOptions()
    => new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false
    };

sealed class StateRow
{
    public string Evidence { get; set; } = string.Empty;
}

class FacadeRow
{
    public string PortableResults { get; set; } = string.Empty;
    public string Portable_priority_PredictedLabel { get; set; } = string.Empty;
    public VBuffer<float> Portable_priority_Probabilities { get; set; }
    public float Portable_priority_Confidence { get; set; }
    public float Portable_priority_ActionProbability { get; set; }
    public float Portable_quality_Score { get; set; }
    public VBuffer<float> Portable_quality_Probabilities { get; set; }
    public float Portable_quality_Confidence { get; set; }
    public float Portable_quality_ActionProbability { get; set; }
    public bool Portable_actionable_PredictedLabel { get; set; }
    public float Portable_actionable_Probability { get; set; }
    public float Portable_actionable_Confidence { get; set; }
    public float Portable_actionable_ActionProbability { get; set; }
}

sealed class StageRow : FacadeRow
{
    public VBuffer<long> PortableInputIds { get; set; }
    public VBuffer<long> PortableAttentionMask { get; set; }
    public VBuffer<long> PortableMarkerPositions { get; set; }
    public VBuffer<bool> PortableMarkerMask { get; set; }
    public VBuffer<long> PortableQuestionTypes { get; set; }
    public int PortableBatchSize { get; set; }
    public int PortableSequenceLength { get; set; }
    public int PortableMarkerWidth { get; set; }
    public VBuffer<float> PortableLogits { get; set; }
    public VBuffer<float> PortableActionProbabilities { get; set; }
}

sealed class ComposedRow : FacadeRow
{
    public string AppendedDecisionResults { get; set; } = string.Empty;
    public string Appended_binary_PredictedLabel { get; set; } = string.Empty;
    public VBuffer<float> Appended_binary_Probabilities { get; set; }
    public float Appended_binary_Confidence { get; set; }
    public float Appended_binary_ActionProbability { get; set; }
}
