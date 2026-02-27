namespace MLNet.TextGeneration.OnnxGenAI;

/// <summary>
/// Configuration for the ONNX Runtime GenAI text generation transform.
/// </summary>
public class OnnxTextGenerationOptions
{
    /// <summary>Path to the ONNX GenAI model directory.</summary>
    public required string ModelPath { get; set; }

    /// <summary>Name of the input text column. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output generated text column. Default: "GeneratedText".</summary>
    public string OutputColumnName { get; set; } = "GeneratedText";

    /// <summary>Maximum generation length in tokens. Default: 512.</summary>
    public int MaxLength { get; set; } = 512;

    /// <summary>Sampling temperature. Default: 0.7.</summary>
    public float Temperature { get; set; } = 0.7f;

    /// <summary>Top-p (nucleus) sampling threshold. Default: 0.9.</summary>
    public float TopP { get; set; } = 0.9f;

    /// <summary>Optional system prompt prepended to each request.</summary>
    public string? SystemPrompt { get; set; }

    /// <summary>
    /// Optional prompt template. Use {0} as placeholder for the user prompt.
    /// If set, overrides SystemPrompt formatting.
    /// </summary>
    public string? PromptTemplate { get; set; }
}
