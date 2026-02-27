using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace MLNet.TextGeneration.OnnxGenAI;

/// <summary>
/// ML.NET ITransformer that generates text using ONNX Runtime GenAI.
/// Uses eager evaluation with autoregressive token generation.
/// Owns the Model and Tokenizer lifecycle.
/// </summary>
public sealed class OnnxTextGenerationTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextGenerationOptions _options;
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;

    public bool IsRowToRowMapper => true;

    internal OnnxTextGenerationTransformer(
        MLContext mlContext,
        OnnxTextGenerationOptions options,
        Model model,
        Tokenizer tokenizer)
    {
        _mlContext = mlContext;
        _options = options;
        _model = model;
        _tokenizer = tokenizer;
    }

    public IDataView Transform(IDataView input)
    {
        var prompts = ReadTextColumn(input);
        if (prompts.Count == 0)
            return BuildOutputDataView(prompts, []);

        var responses = new List<string>(prompts.Count);
        foreach (var prompt in prompts)
        {
            var formatted = FormatPrompt(prompt);
            responses.Add(RunGeneration(formatted));
        }

        return BuildOutputDataView(prompts, responses);
    }

    /// <summary>
    /// Generates text responses for the given prompts directly, without ML.NET IDataView overhead.
    /// </summary>
    public string[] Generate(IReadOnlyList<string> prompts)
    {
        var results = new string[prompts.Count];
        for (int i = 0; i < prompts.Count; i++)
        {
            var formatted = FormatPrompt(prompts[i]);
            results[i] = RunGeneration(formatted);
        }

        return results;
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.OutputColumnName, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "Cannot save an ONNX GenAI text generation transform. " +
            "The model must be loaded from disk at runtime.");

    public void Dispose()
    {
        _tokenizer.Dispose();
        _model.Dispose();
    }

    private string FormatPrompt(string prompt)
    {
        if (!string.IsNullOrEmpty(_options.PromptTemplate))
            return string.Format(_options.PromptTemplate, prompt);

        if (!string.IsNullOrEmpty(_options.SystemPrompt))
            return $"<|system|>\n{_options.SystemPrompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n";

        return prompt;
    }

    private string RunGeneration(string formattedPrompt)
    {
        var sequences = _tokenizer.Encode(formattedPrompt);

        using var generatorParams = new GeneratorParams(_model);
        generatorParams.SetSearchOption("max_length", _options.MaxLength);
        generatorParams.SetSearchOption("temperature", _options.Temperature);
        generatorParams.SetSearchOption("top_p", _options.TopP);

        using var generator = new Generator(_model, generatorParams);
        generator.AppendTokenSequences(sequences);
        using var tokenizerStream = _tokenizer.CreateStream();

        var result = new StringBuilder();
        while (!generator.IsDone())
        {
            generator.GenerateNextToken();
            var newToken = generator.GetSequence(0UL);
            var tokenId = newToken[^1];
            result.Append(tokenizerStream.Decode(tokenId));
        }

        return result.ToString();
    }

    private List<string> ReadTextColumn(IDataView dataView)
    {
        var texts = new List<string>();
        var col = dataView.Schema[_options.InputColumnName];
        using var cursor = dataView.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(col);

        ReadOnlyMemory<char> value = default;
        while (cursor.MoveNext())
        {
            getter(ref value);
            texts.Add(value.ToString());
        }

        return texts;
    }

    private IDataView BuildOutputDataView(List<string> prompts, List<string> responses)
    {
        var rows = new List<TextGenerationRow>();

        for (int i = 0; i < prompts.Count; i++)
        {
            rows.Add(new TextGenerationRow
            {
                Text = prompts[i],
                GeneratedText = i < responses.Count ? responses[i] : ""
            });
        }

        return _mlContext.Data.LoadFromEnumerable(rows);
    }

    private sealed class TextGenerationRow
    {
        public string Text { get; set; } = "";
        public string GeneratedText { get; set; } = "";
    }
}
