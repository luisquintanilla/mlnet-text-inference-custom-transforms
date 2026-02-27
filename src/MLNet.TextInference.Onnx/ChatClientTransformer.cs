using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that generates text using any IChatClient.
/// Uses eager evaluation — each prompt is sent individually to the chat client.
/// </summary>
public sealed class ChatClientTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly IChatClient _chatClient;
    private readonly TextGenerationOptions _options;

    public bool IsRowToRowMapper => true;

    internal IChatClient ChatClient => _chatClient;

    internal ChatClientTransformer(
        MLContext mlContext,
        IChatClient chatClient,
        TextGenerationOptions options)
    {
        _mlContext = mlContext;
        _chatClient = chatClient;
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        var prompts = ReadTextColumn(input);
        if (prompts.Count == 0)
            return BuildOutputDataView(prompts, []);

        var responses = new List<string>(prompts.Count);
        foreach (var prompt in prompts)
        {
            var messages = new List<ChatMessage>();
            if (!string.IsNullOrEmpty(_options.SystemPrompt))
                messages.Add(new ChatMessage(ChatRole.System, _options.SystemPrompt));
            messages.Add(new ChatMessage(ChatRole.User, prompt));

            var chatOptions = new ChatOptions();
            if (_options.MaxOutputTokens.HasValue)
                chatOptions.MaxOutputTokens = _options.MaxOutputTokens.Value;
            if (_options.Temperature.HasValue)
                chatOptions.Temperature = _options.Temperature.Value;
            if (_options.TopP.HasValue)
                chatOptions.TopP = _options.TopP.Value;

            var response = _chatClient.GetResponseAsync(messages, chatOptions).GetAwaiter().GetResult();
            responses.Add(response.Text ?? "");
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
            var messages = new List<ChatMessage>();
            if (!string.IsNullOrEmpty(_options.SystemPrompt))
                messages.Add(new ChatMessage(ChatRole.System, _options.SystemPrompt));
            messages.Add(new ChatMessage(ChatRole.User, prompts[i]));

            var chatOptions = new ChatOptions();
            if (_options.MaxOutputTokens.HasValue)
                chatOptions.MaxOutputTokens = _options.MaxOutputTokens.Value;
            if (_options.Temperature.HasValue)
                chatOptions.Temperature = _options.Temperature.Value;
            if (_options.TopP.HasValue)
                chatOptions.TopP = _options.TopP.Value;

            var response = _chatClient.GetResponseAsync(messages, chatOptions).GetAwaiter().GetResult();
            results[i] = response.Text ?? "";
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
            "Cannot save a provider-agnostic text generation transform. " +
            "The IChatClient instance cannot be serialized.");

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
