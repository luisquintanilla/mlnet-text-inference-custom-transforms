using Microsoft.ML.OnnxRuntime;

namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// Runs the Laya decision graph with the exact five-input contract.
/// </summary>
internal sealed class ScoreOnnxDecisionModel : IDisposable
{
    private readonly InferenceSession _session;
    private bool _disposed;
    internal int PadTokenId { get; }

    public ScoreOnnxDecisionModel(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("Decision ONNX model was not found.", modelPath);

        _session = new InferenceSession(modelPath);
        ValidateModelContract(_session);
    }

    public ScoreOnnxDecisionModel(TypedDecisionBundle bundle)
        : this(bundle?.ModelPath ?? throw new ArgumentNullException(nameof(bundle)))
    {
        PadTokenId = bundle.Tokenizer.Metadata.PadTokenId;
    }

    public DecisionModelOutputs Score(DecisionInputBatch batch)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(batch);
        batch.Validate();

        var inputs = new Dictionary<string, OrtValue>
        {
            ["input_ids"] = OrtValue.CreateTensorValueFromMemory(
                batch.InputIds, [batch.BatchSize, batch.SequenceLength]),
            ["attention_mask"] = OrtValue.CreateTensorValueFromMemory(
                batch.AttentionMask, [batch.BatchSize, batch.SequenceLength]),
            ["marker_pos"] = OrtValue.CreateTensorValueFromMemory(
                batch.MarkerPositions, [batch.BatchSize, batch.MarkerWidth]),
            ["marker_mask"] = OrtValue.CreateTensorValueFromMemory(
                batch.MarkerMask, [batch.BatchSize, batch.MarkerWidth]),
            ["qtype"] = OrtValue.CreateTensorValueFromMemory(
                batch.QuestionTypes, [batch.BatchSize])
        };

        try
        {
            using var results = _session.Run(
                new RunOptions(),
                inputs,
                ["logits", "act_probs"]);

            if (results.Count != 2)
                throw new InvalidDataException("The decision graph must return logits and act_probs.");

            var logits = results[0].GetTensorDataAsSpan<float>().ToArray();
            var actionProbabilities = results[1].GetTensorDataAsSpan<float>().ToArray();
            if (logits.Length != batch.BatchSize * batch.MarkerWidth)
            {
                throw new InvalidDataException(
                    $"The logits output has {logits.Length} values; expected " +
                    $"[{batch.BatchSize},{batch.MarkerWidth}].");
            }

            if (actionProbabilities.Length != batch.BatchSize * 2)
            {
                throw new InvalidDataException(
                    $"The act_probs output has {actionProbabilities.Length} values; expected " +
                    $"[{batch.BatchSize},2].");
            }

            var output = new DecisionModelOutputs
            {
                BatchSize = batch.BatchSize,
                MarkerWidth = batch.MarkerWidth,
                Logits = logits,
                ActionProbabilities = actionProbabilities
            };
            output.Validate();
            return output;
        }
        finally
        {
            foreach (var input in inputs.Values)
                input.Dispose();
        }
    }

    internal DecisionModelOutputs Score(
        long[] inputIds,
        long[] attentionMask,
        long[] markerPositions,
        bool[] markerMask,
        long[] questionTypes,
        int batchSize,
        int sequenceLength,
        int markerWidth)
    {
        ArgumentNullException.ThrowIfNull(inputIds);
        ArgumentNullException.ThrowIfNull(attentionMask);
        ArgumentNullException.ThrowIfNull(markerPositions);
        ArgumentNullException.ThrowIfNull(markerMask);
        ArgumentNullException.ThrowIfNull(questionTypes);
        if (batchSize <= 0 || sequenceLength <= 0 || markerWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));

        var items = Enumerable.Range(0, batchSize)
            .Select(static _ => new DecisionInputItem(
                0,
                DecisionQuestion.Noul("internal", "internal"),
                Enumerable.Range(0, 2).ToArray(),
                Enumerable.Range(0, 2).Select(static value => value.ToString()).ToArray(),
                (long)DecisionQuestionType.Noul))
            .ToArray();
        return Score(new DecisionInputBatch
        {
            BatchSize = batchSize,
            SequenceLength = sequenceLength,
            MarkerWidth = markerWidth,
            InputIds = inputIds,
            AttentionMask = attentionMask,
            MarkerPositions = markerPositions,
            MarkerMask = markerMask,
            QuestionTypes = questionTypes,
            Items = items
        });
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _session.Dispose();
    }

    private static void ValidateModelContract(InferenceSession session)
    {
        var requiredInputs = new[] { "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype" };
        foreach (var input in requiredInputs)
        {
            if (!session.InputMetadata.ContainsKey(input))
                throw new InvalidDataException(
                    $"The decision graph is missing required input '{input}'. " +
                    $"Available inputs: {string.Join(", ", session.InputMetadata.Keys)}");
        }

        foreach (var input in requiredInputs)
        {
            var metadata = session.InputMetadata[input];
            var dimensions = metadata.Dimensions;
            var expectedRank = input == "qtype" ? 1 : input is "marker_pos" or "marker_mask" ? 2 : 2;
            if (dimensions.Length != expectedRank)
                throw new InvalidDataException(
                    $"Input '{input}' must have rank {expectedRank}; actual rank is {dimensions.Length}.");

            var expectedType = input == "marker_mask" ? typeof(bool) : typeof(long);
            if (metadata.ElementType != expectedType)
                throw new InvalidDataException(
                    $"Input '{input}' must use {expectedType.Name}; actual type is " +
                    $"{metadata.ElementType?.Name ?? "unknown"}.");
        }

        foreach (var output in new[] { "logits", "act_probs" })
        {
            if (!session.OutputMetadata.TryGetValue(output, out var metadata))
                throw new InvalidDataException(
                    $"The decision graph is missing required output '{output}'. " +
                    $"Available outputs: {string.Join(", ", session.OutputMetadata.Keys)}");
            if (metadata.Dimensions.Length != 2)
                throw new InvalidDataException($"Output '{output}' must have rank 2.");
            if (metadata.ElementType != typeof(float))
                throw new InvalidDataException(
                    $"Output '{output}' must use Single; actual type is " +
                    $"{metadata.ElementType?.Name ?? "unknown"}.");
        }
    }
}
