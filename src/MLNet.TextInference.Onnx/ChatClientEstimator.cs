using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the provider-agnostic text generation transform.
/// </summary>
public class TextGenerationOptions
{
    /// <summary>Name of the input text column. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output generated text column. Default: "GeneratedText".</summary>
    public string OutputColumnName { get; set; } = "GeneratedText";

    /// <summary>Optional system prompt prepended to each request.</summary>
    public string? SystemPrompt { get; set; }

    /// <summary>Maximum number of tokens to generate.</summary>
    public int? MaxOutputTokens { get; set; }

    /// <summary>Sampling temperature.</summary>
    public float? Temperature { get; set; }

    /// <summary>Top-p (nucleus) sampling threshold.</summary>
    public float? TopP { get; set; }
}

/// <summary>
/// ML.NET IEstimator that wraps any IChatClient to produce text generation within a pipeline.
/// Provider-agnostic — works with ONNX Runtime GenAI, OpenAI, Azure OpenAI, Ollama, or any MEAI implementation.
/// </summary>
public sealed class ChatClientEstimator : IEstimator<ChatClientTransformer>
{
    private readonly MLContext _mlContext;
    private readonly IChatClient _chatClient;
    private readonly TextGenerationOptions _options;

    public ChatClientEstimator(
        MLContext mlContext,
        IChatClient chatClient,
        TextGenerationOptions? options = null)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _chatClient = chatClient ?? throw new ArgumentNullException(nameof(chatClient));
        _options = options ?? new TextGenerationOptions();
    }

    public ChatClientTransformer Fit(IDataView input)
    {
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        return new ChatClientTransformer(_mlContext, _chatClient, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var inputCol = inputSchema.FirstOrDefault(c => c.Name == _options.InputColumnName);
        if (inputCol.Name == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        var result = inputSchema.ToDictionary(x => x.Name);

        var colCtor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];
        var outputCol = (SchemaShape.Column)colCtor.Invoke([
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)TextDataViewType.Instance,
            false,
            (SchemaShape?)null
        ]);
        result[_options.OutputColumnName] = outputCol;

        return new SchemaShape(result.Values);
    }
}
