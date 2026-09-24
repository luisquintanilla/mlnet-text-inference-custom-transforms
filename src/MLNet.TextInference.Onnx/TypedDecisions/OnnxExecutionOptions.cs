namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// Provider settings passed from an adapter to the ML.NET-independent ONNX
/// session factory.
/// </summary>
internal sealed record OnnxExecutionOptions(
    int? GpuDeviceId = null,
    bool FallbackToCpu = false,
    Action<string>? Warning = null);
