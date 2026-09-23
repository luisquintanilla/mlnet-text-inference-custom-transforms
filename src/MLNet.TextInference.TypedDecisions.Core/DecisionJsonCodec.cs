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
        return JsonSerializer.Serialize(batch, JsonOptions);
    }

    public static DecisionInputBatch DeserializeInputs(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        var batch = JsonSerializer.Deserialize<DecisionInputBatch>(json, JsonOptions)
            ?? throw new InvalidDataException("Prepared decision inputs are not valid JSON.");
        batch.Validate();
        return batch;
    }

    public static string SerializeScored(
        DecisionInputBatch inputs,
        DecisionModelOutputs outputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputs);
        inputs.Validate();
        outputs.Validate();
        return JsonSerializer.Serialize(new ScoredEnvelope
        {
            Inputs = inputs,
            Outputs = outputs
        }, JsonOptions);
    }

    public static (DecisionInputBatch Inputs, DecisionModelOutputs Outputs) DeserializeScored(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        var envelope = JsonSerializer.Deserialize<ScoredEnvelope>(json, JsonOptions)
            ?? throw new InvalidDataException("Scored decision inputs are not valid JSON.");
        ArgumentNullException.ThrowIfNull(envelope.Inputs);
        envelope.Inputs.Validate();
        ArgumentNullException.ThrowIfNull(envelope.Outputs);
        envelope.Outputs.Validate();
        return (envelope.Inputs, envelope.Outputs);
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

    private sealed class ScoredEnvelope
    {
        public DecisionInputBatch Inputs { get; set; } = null!;
        public DecisionModelOutputs Outputs { get; set; } = null!;
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter() }
    };
}
