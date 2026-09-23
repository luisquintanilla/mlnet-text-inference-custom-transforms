using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLNet.TextInference.TypedDecisions;

/// <summary>Stable JSON envelopes used by the ML.NET stage adapters.</summary>
public static class DecisionJsonCodec
{
    public static string SerializeInputs(DecisionInputBatch batch)
    {
        ArgumentNullException.ThrowIfNull(batch);
        batch.Validate();
        return JsonSerializer.Serialize(new InputEnvelope
        {
            BatchSize = batch.BatchSize,
            SequenceLength = batch.SequenceLength,
            MarkerWidth = batch.MarkerWidth,
            InputIds = batch.InputIds,
            AttentionMask = batch.AttentionMask,
            MarkerPositions = batch.MarkerPositions,
            MarkerMask = batch.MarkerMask,
            QuestionTypes = batch.QuestionTypes,
            Items = batch.Items.Select(static item => new InputItemEnvelope
            {
                RequestIndex = item.RequestIndex,
                Question = item.Question,
                MarkerPositions = item.MarkerPositions,
                OptionLabels = item.OptionLabels,
                QuestionType = item.QuestionType
            }).ToArray()
        }, JsonOptions);
    }

    public static DecisionInputBatch DeserializeInputs(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        var envelope = JsonSerializer.Deserialize<InputEnvelope>(json, JsonOptions)
            ?? throw new InvalidDataException("Prepared decision inputs are not valid JSON.");
        var items = envelope.Items.Select(static item => new DecisionInputItem(
            item.RequestIndex,
            item.Question,
            item.MarkerPositions,
            item.OptionLabels,
            item.QuestionType)).ToArray();
        var batch = new DecisionInputBatch
        {
            BatchSize = envelope.BatchSize,
            SequenceLength = envelope.SequenceLength,
            MarkerWidth = envelope.MarkerWidth,
            InputIds = envelope.InputIds,
            AttentionMask = envelope.AttentionMask,
            MarkerPositions = envelope.MarkerPositions,
            MarkerMask = envelope.MarkerMask,
            QuestionTypes = envelope.QuestionTypes,
            Items = items
        };
        batch.Validate();
        return batch;
    }

    public static string SerializeScored(
        DecisionInputBatch inputs,
        DecisionModelOutputs outputs)
    {
        ArgumentNullException.ThrowIfNull(outputs);
        outputs.Validate();
        return JsonSerializer.Serialize(new ScoredEnvelope
        {
            Inputs = JsonSerializer.Deserialize<InputEnvelope>(
                SerializeInputs(inputs), JsonOptions)!,
            Outputs = outputs
        }, JsonOptions);
    }

    public static (DecisionInputBatch Inputs, DecisionModelOutputs Outputs) DeserializeScored(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        var envelope = JsonSerializer.Deserialize<ScoredEnvelope>(json, JsonOptions)
            ?? throw new InvalidDataException("Scored decision inputs are not valid JSON.");
        var inputs = DeserializeInputs(JsonSerializer.Serialize(envelope.Inputs, JsonOptions));
        envelope.Outputs.Validate();
        return (inputs, envelope.Outputs);
    }

    public static string SerializeResponse(DecisionResponse response)
    {
        ArgumentNullException.ThrowIfNull(response);
        return JsonSerializer.Serialize(new
        {
            input_tokens = response.InputTokenCount,
            results = response.Results.Select(SerializeResult)
        }, JsonOptions);
    }

    private static object SerializeResult(DecisionResult result)
    {
        var common = new
        {
            id = result.Id,
            type = result.Type.ToString().ToLowerInvariant(),
            confidence = result.Confidence,
            action_probability = result.ActionProbability,
            labels = result.Distribution.Labels,
            probabilities = result.Distribution.Probabilities
        };

        return result switch
        {
            ChoiceDecisionResult choice => new
            {
                common.id,
                common.type,
                common.confidence,
                common.action_probability,
                common.labels,
                common.probabilities,
                choice = choice.Choice
            },
            ScoreDecisionResult score => new
            {
                common.id,
                common.type,
                common.confidence,
                common.action_probability,
                common.labels,
                common.probabilities,
                score = score.Score,
                legend = score.Legend
            },
            NoulDecisionResult noul => new
            {
                common.id,
                common.type,
                common.confidence,
                common.action_probability,
                common.labels,
                common.probabilities,
                noul = noul.Value,
                probability_true = noul.ProbabilityTrue
            },
            _ => throw new ArgumentOutOfRangeException(nameof(result))
        };
    }

    private sealed class InputEnvelope
    {
        public int BatchSize { get; set; }
        public int SequenceLength { get; set; }
        public int MarkerWidth { get; set; }
        public long[] InputIds { get; set; } = [];
        public long[] AttentionMask { get; set; } = [];
        public long[] MarkerPositions { get; set; } = [];
        public bool[] MarkerMask { get; set; } = [];
        public long[] QuestionTypes { get; set; } = [];
        public InputItemEnvelope[] Items { get; set; } = [];
    }

    private sealed class InputItemEnvelope
    {
        public int RequestIndex { get; set; }
        public DecisionQuestion Question { get; set; } = null!;
        public int[] MarkerPositions { get; set; } = [];
        public string[] OptionLabels { get; set; } = [];
        public long QuestionType { get; set; }
    }

    private sealed class ScoredEnvelope
    {
        public InputEnvelope Inputs { get; set; } = null!;
        public DecisionModelOutputs Outputs { get; set; } = null!;
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter() }
    };
}
