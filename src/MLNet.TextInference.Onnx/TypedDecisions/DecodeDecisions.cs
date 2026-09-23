using System.Numerics.Tensors;

namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// Decodes Laya logits and action probabilities in C# using stable tensor primitives.
/// </summary>
internal sealed class DecodeDecisions
{
    private readonly DecisionTemperaturePolicy _temperaturePolicy;
    private readonly TypedDecisionDecoderMetadata _metadata;

    public DecodeDecisions(
        DecisionTemperaturePolicy temperaturePolicy,
        TypedDecisionDecoderMetadata? metadata = null)
    {
        _temperaturePolicy = temperaturePolicy ?? throw new ArgumentNullException(nameof(temperaturePolicy));
        _metadata = metadata ?? new TypedDecisionDecoderMetadata(
            temperaturePolicy.Minimum,
            temperaturePolicy.Maximum);
        if (_metadata.ActionProbabilityIndex is < 0 or > 1)
            throw new ArgumentOutOfRangeException(
                nameof(metadata),
                "ActionProbabilityIndex must be 0 or 1.");
    }

    public IReadOnlyList<TypedDecisionDiagnostic> Diagnostics =>
        _temperaturePolicy.Diagnostics;

    public DecisionResponse Decode(
        DecisionInputBatch inputs,
        DecisionModelOutputs outputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputs);
        inputs.Validate();
        outputs.Validate();
        if (inputs.BatchSize != outputs.BatchSize || inputs.MarkerWidth != outputs.MarkerWidth)
            throw new ArgumentException("Inputs and outputs have different shapes.", nameof(outputs));

        var results = new DecisionResult[inputs.BatchSize];
        var inputTokenCount = 0;
        for (int row = 0; row < inputs.BatchSize; row++)
        {
            var attention = inputs.AttentionMask.AsSpan(
                row * inputs.SequenceLength,
                inputs.SequenceLength);
            for (int token = 0; token < attention.Length; token++)
            {
                if (attention[token] != 0)
                    inputTokenCount++;
            }

            var item = inputs.Items[row];
            item.Question.Validate();
            var optionCount = item.OptionLabels.Length;
            if (optionCount < 2 && item.Question.Type is DecisionQuestionType.Choice or DecisionQuestionType.Score)
                throw new InvalidOperationException(
                    $"Question '{item.Question.Id}' has one option; single-option {item.Question.Type} " +
                    "questions are explicitly unsupported in v1.");

            var logits = outputs.Logits.AsSpan(row * outputs.MarkerWidth, optionCount);
            var probabilities = new float[optionCount];
            var temperature = _temperaturePolicy.For(item.Question.Type, optionCount);
            var scaled = new float[optionCount];
            for (int i = 0; i < optionCount; i++)
                scaled[i] = logits[i] / temperature;
            TensorPrimitives.SoftMax(scaled, probabilities);

            var labels = item.OptionLabels;
            var distribution = new DecisionDistribution(labels, probabilities);
            var confidence = Confidence(probabilities);
            var actionProbability = outputs.ActionProbabilities[
                row * 2 + _metadata.ActionProbabilityIndex];

            results[row] = item.Question.Type switch
            {
                DecisionQuestionType.Choice => DecodeChoice(
                    item.Question.Id, labels, probabilities, distribution, confidence, actionProbability),
                DecisionQuestionType.Score => DecodeScore(
                    item.Question.Id, item.Question.ScoreLevels!, probabilities, distribution, confidence, actionProbability),
                DecisionQuestionType.Noul => new NoulDecisionResult(
                    item.Question.Id,
                    probabilities[1] >= probabilities[0],
                    probabilities[1],
                    distribution,
                    confidence,
                    actionProbability),
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        return new DecisionResponse
        {
            Results = results,
            InputTokenCount = inputTokenCount
        };
    }

    private static ChoiceDecisionResult DecodeChoice(
        string id,
        IReadOnlyList<string> labels,
        IReadOnlyList<float> probabilities,
        DecisionDistribution distribution,
        float confidence,
        float actionProbability)
    {
        var best = TensorPrimitives.IndexOfMax(probabilities.ToArray());
        return new ChoiceDecisionResult(
            id,
            labels[best],
            distribution,
            confidence,
            actionProbability);
    }

    private static ScoreDecisionResult DecodeScore(
        string id,
        IReadOnlyList<string> levels,
        IReadOnlyList<float> probabilities,
        DecisionDistribution distribution,
        float confidence,
        float actionProbability)
    {
        var legend = levels
            .Select((text, index) => new KeyValuePair<string, string>(index.ToString(), text))
            .ToDictionary(static pair => pair.Key, static pair => pair.Value, StringComparer.Ordinal);
        var score = 0f;
        for (int index = 0; index < probabilities.Count; index++)
            score += index * probabilities[index];
        return new ScoreDecisionResult(
            id,
            score,
            legend,
            distribution,
            confidence,
            actionProbability);
    }

    private static float Confidence(IReadOnlyList<float> probabilities)
    {
        if (probabilities.Count < 2)
            return 1;
        var entropy = 0f;
        foreach (var probability in probabilities)
            entropy -= probability * MathF.Log(MathF.Max(probability, 1e-12f));
        return 1 - entropy / MathF.Log(probabilities.Count);
    }
}
