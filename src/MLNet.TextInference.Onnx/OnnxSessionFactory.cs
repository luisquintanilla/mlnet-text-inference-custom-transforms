using Microsoft.ML.OnnxRuntime;
using MLNet.TextInference.TypedDecisions;

namespace MLNet.TextInference.Onnx;

internal static class OnnxSessionFactory
{
    internal static InferenceSession Create(
        string modelPath,
        OnnxExecutionOptions? executionOptions = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("ONNX model was not found.", modelPath);

        var deviceId = executionOptions?.GpuDeviceId;
        var fallbackToCpu = executionOptions?.FallbackToCpu ?? false;
        var warning = executionOptions?.Warning;

        using var options = new SessionOptions();
        if (deviceId is not null)
        {
            try
            {
                options.AppendExecutionProvider_CUDA(deviceId.Value);
            }
            catch (Exception exception) when (
                fallbackToCpu &&
                IsProviderInitializationFailure(exception))
            {
                warning?.Invoke(
                    $"CUDA execution provider setup failed for device {deviceId}; " +
                    "retrying CPU. " + exception.Message);
                return CreateCpuSession(modelPath);
            }
        }

        // Provider initialization can also be deferred until the session is
        // constructed. Retry CPU only for an explicitly requested provider;
        // if CPU construction fails, its exception still surfaces to the
        // caller rather than disguising a malformed graph or sidecar.
        try
        {
            return new InferenceSession(modelPath, options);
        }
        catch (Exception exception) when (
            deviceId is not null &&
            fallbackToCpu &&
            IsProviderInitializationFailure(exception))
        {
            warning?.Invoke(
                $"CUDA session construction failed for device {deviceId}; " +
                "retrying CPU. " + exception.Message);
            return CreateCpuSession(modelPath);
        }
    }

    private static bool IsProviderInitializationFailure(Exception exception)
    {
        if (exception is DllNotFoundException or EntryPointNotFoundException)
            return true;
        if (exception is not OnnxRuntimeException)
            return false;

        var message = exception.ToString();
        return message.Contains("CUDA", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("cuDNN", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("execution provider", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("provider library", StringComparison.OrdinalIgnoreCase);
    }

    private static InferenceSession CreateCpuSession(string modelPath)
    {
        using var options = new SessionOptions();
        return new InferenceSession(modelPath, options);
    }
}
