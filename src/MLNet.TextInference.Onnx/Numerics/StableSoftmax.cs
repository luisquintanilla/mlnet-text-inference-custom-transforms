using System.Numerics.Tensors;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Applies a finite-logit softmax after subtracting the maximum value.
/// The helper owns validation and valid-slice semantics so task decoders do not
/// accidentally normalize padding or rely on the runtime's overflow behavior.
/// </summary>
internal static class StableSoftmax
{
    public static int IndexOfMax(ReadOnlySpan<float> values)
    {
        if (values.Length == 0)
            throw new ArgumentException("At least one value is required.", nameof(values));

        int index = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[index])
                index = i;
        }

        return index;
    }

    public static float[] Create(
        ReadOnlySpan<float> logits,
        int validCount,
        float temperature = 1f)
    {
        var probabilities = new float[logits.Length];
        Apply(logits, probabilities, validCount, temperature);
        return probabilities;
    }

    public static void Apply(
        ReadOnlySpan<float> logits,
        Span<float> probabilities,
        int validCount,
        float temperature = 1f)
    {
        if (validCount <= 0 || validCount > logits.Length)
            throw new ArgumentOutOfRangeException(nameof(validCount));
        if (probabilities.Length < logits.Length)
            throw new ArgumentException(
                "The destination must be at least as long as the logits.", nameof(probabilities));
        if (!float.IsFinite(temperature) || temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature));

        var validLogits = logits[..validCount];
        for (int i = 0; i < validLogits.Length; i++)
        {
            if (!float.IsFinite(validLogits[i]))
                throw new ArgumentException(
                    "Softmax logits must be finite.", nameof(logits));
        }

        float maximum = validLogits[0];
        for (int i = 1; i < validLogits.Length; i++)
            maximum = MathF.Max(maximum, validLogits[i]);

        var validProbabilities = probabilities[..validCount];
        for (int i = 0; i < validProbabilities.Length; i++)
            validProbabilities[i] = MathF.Exp((validLogits[i] - maximum) / temperature);

        float sum = TensorPrimitives.Sum(validProbabilities);
        if (!float.IsFinite(sum) || sum <= 0)
            throw new InvalidOperationException("Softmax normalization produced an invalid sum.");

        TensorPrimitives.Divide(validProbabilities, sum, validProbabilities);
        probabilities[validCount..].Clear();
    }
}
