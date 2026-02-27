namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the end-to-end ONNX QA pipeline.
/// Chains text-pair tokenization → multi-output ONNX inference → QA span extraction.
/// </summary>
public class OnnxQaOptions
{
    /// <summary>Path to the ONNX QA model file.</summary>
    public required string ModelPath { get; set; }

    /// <summary>Path to the tokenizer artifacts.</summary>
    public required string TokenizerPath { get; set; }

    /// <summary>Name of the question input column. Default: "Question".</summary>
    public string QuestionColumnName { get; set; } = "Question";

    /// <summary>Name of the context input column. Default: "Context".</summary>
    public string ContextColumnName { get; set; } = "Context";

    /// <summary>Name of the output answer text column. Default: "Answer".</summary>
    public string OutputColumnName { get; set; } = "Answer";

    /// <summary>Name of the output answer score column. Default: "AnswerScore".</summary>
    public string ScoreColumnName { get; set; } = "AnswerScore";

    /// <summary>Maximum token sequence length. Default: 384.</summary>
    public int MaxTokenLength { get; set; } = 384;

    /// <summary>Maximum answer span length in tokens. Default: 30.</summary>
    public int MaxAnswerLength { get; set; } = 30;

    /// <summary>Batch size for ONNX inference. Default: 32.</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>GPU device ID. Null = CPU only.</summary>
    public int? GpuDeviceId { get; set; }

    /// <summary>If true, fall back to CPU when GPU initialization fails.</summary>
    public bool FallbackToCpu { get; set; }
}
