using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.TextInference.Onnx;

Console.WriteLine("=== Provider-Agnostic Text Generation Transform ===\n");
Console.WriteLine("Use ChatClientEstimator to wrap any IChatClient as an ML.NET transform.");
Console.WriteLine("Swap DemoChatClient for OpenAI, Ollama, Azure OpenAI, ORT GenAI, etc.\n");

var mlContext = new MLContext();

// --- Create an IChatClient (demo/placeholder in this case) ---
// This is the ONLY part that changes per provider.
// For OpenAI:   new OpenAIClient(apiKey).GetChatClient("gpt-4o").AsIChatClient()
// For Azure:    new AzureOpenAIClient(endpoint, credential).GetChatClient("gpt-4o").AsIChatClient()
// For Ollama:   new OllamaChatClient(new Uri("http://localhost:11434"), "phi3")

IChatClient chatClient = new DemoChatClient();

// --- Use the chat client as an ML.NET transform ---
Console.WriteLine("1. ML.NET Pipeline with ChatClientEstimator");
Console.WriteLine(new string('-', 50));

var estimator = mlContext.Transforms.TextGeneration(chatClient, new TextGenerationOptions
{
    SystemPrompt = "You are a helpful assistant. Be concise.",
    MaxOutputTokens = 100,
    Temperature = 0.7f
});

var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "Explain .NET in one sentence." },
    new TextData { Text = "What is the capital of France?" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);
var transformer = estimator.Fit(dataView);
var transformed = transformer.Transform(dataView);

var results = mlContext.Data.CreateEnumerable<TextGenerationResult>(transformed, reuseRowObject: false).ToList();

foreach (var (item, idx) in results.Select((r, i) => (r, i)))
{
    Console.WriteLine($"  Prompt:   \"{sampleData[idx].Text}\"");
    Console.WriteLine($"  Response: \"{item.GeneratedText}\"");
    Console.WriteLine();
}

// --- Direct Generate() API ---
Console.WriteLine("2. Direct Generate() API");
Console.WriteLine(new string('-', 50));

var prompts = new[] { "What is 2+2?", "Name three colors." };
var responses = transformer.Generate(prompts);

for (int i = 0; i < prompts.Length; i++)
{
    Console.WriteLine($"  Prompt:   \"{prompts[i]}\"");
    Console.WriteLine($"  Response: \"{responses[i]}\"");
    Console.WriteLine();
}

Console.WriteLine("Note: Swap DemoChatClient for any IChatClient — pipeline code stays identical.");
Console.WriteLine("\nDone!");

// --- Domain types ---
public class TextData
{
    public string Text { get; set; } = "";
}

public class TextGenerationResult
{
    public string Text { get; set; } = "";
    public string GeneratedText { get; set; } = "";
}

/// <summary>
/// A simple demo IChatClient that returns placeholder responses.
/// Replace with a real provider (OpenAI, Ollama, Azure, ORT GenAI) for actual generation.
/// </summary>
public class DemoChatClient : IChatClient
{
    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var userMessage = messages.LastOrDefault(m => m.Role == ChatRole.User)?.Text ?? "";
        var reply = $"[Demo response to: {userMessage}]";
        return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, reply)));
    }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        throw new NotSupportedException("Streaming not supported in demo client.");
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose() { }
}
