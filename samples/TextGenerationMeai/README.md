# TextGenerationMeai Sample

Use `ChatClientEstimator` to wrap **any** `IChatClient` as an ML.NET transform — demonstrating **provider-agnostic** text generation.

## What This Sample Shows

1. **ML.NET Pipeline** — `ChatClientEstimator` wraps any `IChatClient` as a standard ML.NET transform
2. **Direct API** — Use `ChatClientTransformer.Generate()` directly without IDataView overhead
3. **Provider Swap** — Only the `IChatClient` construction changes per provider

### The Provider-Agnostic Pattern

```csharp
// --- OpenAI ---
IChatClient chatClient = new OpenAIClient(apiKey).GetChatClient("gpt-4o").AsIChatClient();

// --- Ollama ---
// IChatClient chatClient = new OllamaChatClient(new Uri("http://localhost:11434"), "phi3");

// --- Azure OpenAI ---
// IChatClient chatClient = new AzureOpenAIClient(endpoint, credential).GetChatClient("gpt-4o").AsIChatClient();

// Pipeline code is IDENTICAL regardless of provider
var estimator = mlContext.Transforms.TextGeneration(chatClient, new TextGenerationOptions
{
    SystemPrompt = "You are a helpful assistant.",
    MaxOutputTokens = 100
});
var transformer = estimator.Fit(dataView);
var results = transformer.Transform(dataView);
```

## Run

```bash
dotnet run
```

> **Note:** This sample uses a `DemoChatClient` that returns placeholder responses. Replace it with a real `IChatClient` provider for actual text generation.

## Key Code Pattern

```csharp
// Create any IChatClient (provider-specific)
IChatClient chatClient = ...;

// Wrap as ML.NET transform (provider-agnostic)
var estimator = mlContext.Transforms.TextGeneration(chatClient);
var transformer = estimator.Fit(dataView);
var results = transformer.Transform(dataView);

// Or use directly
var responses = transformer.Generate(new[] { "What is ML.NET?" });
```
