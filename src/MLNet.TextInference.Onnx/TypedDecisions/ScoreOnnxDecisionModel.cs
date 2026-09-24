using Microsoft.ML.OnnxRuntime;
using MLNet.TextInference.Onnx;

namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// Runs the Laya decision graph with the exact five-input contract.
/// </summary>
internal sealed class ScoreOnnxDecisionModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Func<int>? _padTokenIdProvider;
    private bool _disposed;
    internal int PadTokenId => _padTokenIdProvider?.Invoke() ?? 0;

    public ScoreOnnxDecisionModel(
        string modelPath,
        OnnxExecutionOptions? executionOptions = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("Decision ONNX model was not found.", modelPath);

        var session = OnnxSessionFactory.Create(
            modelPath,
            executionOptions);
        try
        {
            ValidateModelContract(session);
            _session = session;
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    public ScoreOnnxDecisionModel(
        TypedDecisionBundle bundle,
        OnnxExecutionOptions? executionOptions = null)
        : this(
            bundle?.ModelPath ?? throw new ArgumentNullException(nameof(bundle)),
            executionOptions)
    {
        _padTokenIdProvider = () => bundle.TokenizerMetadata.PadTokenId;
    }

    public DecisionModelOutputs Score(DecisionInputBatch batch)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(batch);
        batch.Validate();
        return ScoreTensors(
            batch.InputIds,
            batch.AttentionMask,
            batch.MarkerPositions,
            batch.MarkerMask,
            batch.QuestionTypes,
            batch.BatchSize,
            batch.SequenceLength,
            batch.MarkerWidth);
    }

    private DecisionModelOutputs ScoreTensors(
        long[] inputIds,
        long[] attentionMask,
        long[] markerPositions,
        bool[] markerMask,
        long[] questionTypes,
        int batchSize,
        int sequenceLength,
        int markerWidth)
    {
        ValidateTensorLengths(
            inputIds,
            attentionMask,
            markerPositions,
            markerMask,
            questionTypes,
            batchSize,
            sequenceLength,
            markerWidth);

        var inputs = new Dictionary<string, OrtValue>(StringComparer.Ordinal);

        try
        {
            inputs["input_ids"] = OrtValue.CreateTensorValueFromMemory(
                inputIds, [batchSize, sequenceLength]);
            inputs["attention_mask"] = OrtValue.CreateTensorValueFromMemory(
                attentionMask, [batchSize, sequenceLength]);
            inputs["marker_pos"] = OrtValue.CreateTensorValueFromMemory(
                markerPositions, [batchSize, markerWidth]);
            inputs["marker_mask"] = OrtValue.CreateTensorValueFromMemory(
                markerMask, [batchSize, markerWidth]);
            inputs["qtype"] = OrtValue.CreateTensorValueFromMemory(
                questionTypes, [batchSize]);

            using var runOptions = new RunOptions();
            using var results = _session.Run(
                runOptions,
                inputs,
                ["logits", "act_probs"]);

            if (results.Count != 2)
                throw new InvalidDataException("The decision graph must return logits and act_probs.");

            var logits = ReadOutput(
                results[0], "logits", batchSize, markerWidth);
            var actionProbabilities = ReadOutput(
                results[1], "act_probs", batchSize, 2);

            var output = new DecisionModelOutputs
            {
                BatchSize = batchSize,
                MarkerWidth = markerWidth,
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
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(inputIds);
        ArgumentNullException.ThrowIfNull(attentionMask);
        ArgumentNullException.ThrowIfNull(markerPositions);
        ArgumentNullException.ThrowIfNull(markerMask);
        ArgumentNullException.ThrowIfNull(questionTypes);
        if (batchSize <= 0 || sequenceLength <= 0 || markerWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));
        return ScoreTensors(
            inputIds,
            attentionMask,
            markerPositions,
            markerMask,
            questionTypes,
            batchSize,
            sequenceLength,
            markerWidth);
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

    private static void ValidateTensorLengths(
        long[] inputIds,
        long[] attentionMask,
        long[] markerPositions,
        bool[] markerMask,
        long[] questionTypes,
        int batchSize,
        int sequenceLength,
        int markerWidth)
    {
        if (batchSize <= 0 || sequenceLength <= 0 || markerWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));
        int sequenceElements;
        int markerElements;
        try
        {
            sequenceElements = checked(batchSize * sequenceLength);
            markerElements = checked(batchSize * markerWidth);
        }
        catch (OverflowException exception)
        {
            throw new ArgumentException(
                "Decision tensor dimensions exceed the supported managed array size.",
                nameof(batchSize),
                exception);
        }

        if (inputIds.Length != sequenceElements ||
            attentionMask.Length != sequenceElements)
        {
            throw new ArgumentException(
                "input_ids and attention_mask lengths must equal batch_size * sequence_length.");
        }
        if (markerPositions.Length != markerElements ||
            markerMask.Length != markerElements)
        {
            throw new ArgumentException(
                "marker_pos and marker_mask lengths must equal batch_size * marker_width.");
        }
        if (questionTypes.Length != batchSize)
            throw new ArgumentException(
                "qtype length must equal batch_size.",
                nameof(questionTypes));
    }

        private static float[] ReadOutput(
            OrtValue output,
            string name,
            int expectedBatchSize,
            int expectedWidth)
        {
            var shape = output.GetTensorTypeAndShape();
            var dimensions = shape.Shape;
            if (dimensions.Length != 2 ||
                dimensions[0] != expectedBatchSize ||
                dimensions[1] != expectedWidth)
            {
                throw new InvalidDataException(
                    $"The {name} output has shape [{string.Join(",", dimensions)}]; " +
                    $"expected [{expectedBatchSize},{expectedWidth}].");
            }

            return output.GetTensorDataAsSpan<float>().ToArray();
        }
    }
