using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Tokenizers;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the text tokenizer transform.
/// Provide either <see cref="Tokenizer"/> (a pre-constructed instance) or
/// <see cref="TokenizerPath"/> (a file/directory to auto-load). If both are set,
/// <see cref="Tokenizer"/> takes precedence.
/// </summary>
public class TextTokenizerOptions
{
    /// <summary>
    /// A pre-constructed tokenizer instance. Use this when working with
    /// tokenizer formats that LoadTokenizer doesn't support, or when
    /// sharing a tokenizer across multiple estimators.
    /// Takes precedence over <see cref="TokenizerPath"/> if both are set.
    /// </summary>
    public Tokenizer? Tokenizer { get; set; }

    /// <summary>
    /// Path to tokenizer artifacts. Can be:
    /// <list type="bullet">
    ///   <item>A directory containing <c>tokenizer_config.json</c> — auto-detects tokenizer type from HuggingFace config</item>
    ///   <item>A <c>tokenizer_config.json</c> file directly — reads <c>tokenizer_class</c> and loads sibling files</item>
    ///   <item>A vocab file: <c>.txt</c> (BERT/WordPiece), <c>.model</c> (SentencePiece)</item>
    /// </list>
    /// Used only when <see cref="Tokenizer"/> is not set.
    /// </summary>
    public string? TokenizerPath { get; set; }

    /// <summary>Name of the input text column. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output token IDs column. Default: "TokenIds".</summary>
    public string TokenIdsColumnName { get; set; } = "TokenIds";

    /// <summary>Name of the output attention mask column. Default: "AttentionMask".</summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>Name of the output token type IDs column. Default: "TokenTypeIds".</summary>
    public string TokenTypeIdsColumnName { get; set; } = "TokenTypeIds";

    /// <summary>
    /// Maximum number of tokens per input text.
    /// Texts are truncated to this length; shorter texts are zero-padded.
    /// Default: 128.
    /// </summary>
    public int MaxTokenLength { get; set; } = 128;

    /// <summary>
    /// Whether to output the token type IDs column.
    /// Set to false for models that don't use segment embeddings.
    /// Default: true.
    /// </summary>
    public bool OutputTokenTypeIds { get; set; } = true;
}

/// <summary>
/// ML.NET IEstimator that creates a TextTokenizerTransformer.
/// Trivial estimator — nothing to learn from training data.
/// Fit() validates the input schema and loads the tokenizer.
/// </summary>
public sealed class TextTokenizerEstimator : IEstimator<TextTokenizerTransformer>
{
    private readonly MLContext _mlContext;
    private readonly TextTokenizerOptions _options;

    public TextTokenizerEstimator(MLContext mlContext, TextTokenizerOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (options.Tokenizer == null && options.TokenizerPath == null)
            throw new ArgumentException(
                "Either Tokenizer or TokenizerPath must be provided.", nameof(options));

        if (options.Tokenizer == null)
        {
            var path = options.TokenizerPath!;
            if (!File.Exists(path) && !Directory.Exists(path))
                throw new FileNotFoundException(
                    $"Tokenizer path not found: {path}");
        }
    }

    public TextTokenizerTransformer Fit(IDataView input)
    {
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        var tokenizer = _options.Tokenizer ?? LoadTokenizer(_options.TokenizerPath!);
        return new TextTokenizerTransformer(_mlContext, _options, tokenizer);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var inputCol = inputSchema.FirstOrDefault(c => c.Name == _options.InputColumnName);
        if (inputCol.Name == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        if (inputCol.ItemType != TextDataViewType.Instance)
            throw new ArgumentException(
                $"Column '{_options.InputColumnName}' must be of type Text.");

        var result = inputSchema.ToDictionary(x => x.Name);

        AddVectorColumn(result, _options.TokenIdsColumnName, NumberDataViewType.Int64);
        AddVectorColumn(result, _options.AttentionMaskColumnName, NumberDataViewType.Int64);
        if (_options.OutputTokenTypeIds)
            AddVectorColumn(result, _options.TokenTypeIdsColumnName, NumberDataViewType.Int64);

        return new SchemaShape(result.Values);
    }

    /// <summary>
    /// Resolves a tokenizer from a path. Supports:
    /// <list type="bullet">
    ///   <item>Directory with <c>tokenizer_config.json</c> → reads <c>tokenizer_class</c>, loads sibling files</item>
    ///   <item>Directory without config → scans for known vocab files</item>
    ///   <item><c>tokenizer_config.json</c> file → reads config, loads sibling files</item>
    ///   <item>Vocab file (<c>.txt</c>, <c>.model</c>) → infers type from extension</item>
    /// </list>
    /// </summary>
    internal static Tokenizer LoadTokenizer(string path)
    {
        // Directory: look for config or known files inside
        if (Directory.Exists(path))
            return LoadFromDirectory(path);

        // File: config or direct vocab file
        var fileName = Path.GetFileName(path).ToLowerInvariant();
        if (fileName == "tokenizer_config.json")
            return LoadFromConfig(path);

        return LoadFromVocabFile(path);
    }

    private static Tokenizer LoadFromDirectory(string directory)
    {
        var configPath = Path.Combine(directory, "tokenizer_config.json");
        if (File.Exists(configPath))
            return LoadFromConfig(configPath);

        // No config — scan for known vocab files
        var vocabTxt = Path.Combine(directory, "vocab.txt");
        if (File.Exists(vocabTxt))
            return LoadFromVocabFile(vocabTxt);

        var spModel = Path.Combine(directory, "tokenizer.model");
        if (File.Exists(spModel))
            return LoadFromVocabFile(spModel);

        var spBpeModel = Path.Combine(directory, "sentencepiece.bpe.model");
        if (File.Exists(spBpeModel))
            return LoadFromVocabFile(spBpeModel);

        throw new FileNotFoundException(
            $"No tokenizer_config.json or known vocab file found in '{directory}'. " +
            $"Expected one of: tokenizer_config.json, vocab.txt, tokenizer.model, sentencepiece.bpe.model.");
    }

    private static Tokenizer LoadFromConfig(string configPath)
    {
        var directory = Path.GetDirectoryName(configPath)
            ?? throw new ArgumentException($"Cannot determine directory for config: {configPath}");

        var json = File.ReadAllText(configPath);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var tokenizerClass = root.TryGetProperty("tokenizer_class", out var cls)
            ? cls.GetString() ?? ""
            : "";

        // Normalize: strip "Fast" suffix (BertTokenizerFast → BertTokenizer)
        if (tokenizerClass.EndsWith("Fast", StringComparison.Ordinal))
            tokenizerClass = tokenizerClass[..^4];

        return tokenizerClass switch
        {
            "BertTokenizer" => LoadBertFromConfig(directory, root),
            "DistilBertTokenizer" => LoadBertFromConfig(directory, root),
            "XLMRobertaTokenizer" => LoadSentencePieceFromDirectory(directory),
            "LlamaTokenizer" => LoadSentencePieceFromDirectory(directory),
            "CamembertTokenizer" => LoadSentencePieceFromDirectory(directory),
            "T5Tokenizer" => LoadSentencePieceFromDirectory(directory),
            "AlbertTokenizer" => LoadSentencePieceFromDirectory(directory),
            "GPT2Tokenizer" => LoadBpeFromDirectory(directory),
            "RobertaTokenizer" => LoadBpeFromDirectory(directory),
            _ when !string.IsNullOrEmpty(tokenizerClass) => throw new NotSupportedException(
                $"Unsupported tokenizer_class '{tokenizerClass}' in {configPath}. " +
                $"Supported: BertTokenizer, XLMRobertaTokenizer, LlamaTokenizer, GPT2Tokenizer, RobertaTokenizer. " +
                $"Use the Tokenizer property to provide a pre-constructed instance for unsupported types."),
            _ => throw new InvalidOperationException(
                $"No tokenizer_class found in {configPath}. Cannot auto-detect tokenizer type.")
        };
    }

    private static Tokenizer LoadBertFromConfig(string directory, JsonElement config)
    {
        var vocabPath = Path.Combine(directory, "vocab.txt");
        if (!File.Exists(vocabPath))
            throw new FileNotFoundException(
                $"BERT tokenizer requires vocab.txt in '{directory}'.");

        var lowerCase = config.TryGetProperty("do_lower_case", out var lc) && lc.GetBoolean();

        using var stream = File.OpenRead(vocabPath);
        return BertTokenizer.Create(stream, new BertOptions { LowerCaseBeforeTokenization = lowerCase });
    }

    private static Tokenizer LoadSentencePieceFromDirectory(string directory)
    {
        // Try common SentencePiece file names
        var candidates = new[] { "sentencepiece.bpe.model", "tokenizer.model", "spiece.model" };
        foreach (var candidate in candidates)
        {
            var spPath = Path.Combine(directory, candidate);
            if (File.Exists(spPath))
            {
                using var stream = File.OpenRead(spPath);
                return LlamaTokenizer.Create(stream);
            }
        }

        throw new FileNotFoundException(
            $"SentencePiece tokenizer requires one of [{string.Join(", ", candidates)}] in '{directory}'.");
    }

    private static Tokenizer LoadBpeFromDirectory(string directory)
    {
        var vocabJson = Path.Combine(directory, "vocab.json");
        var mergesPath = Path.Combine(directory, "merges.txt");

        if (!File.Exists(vocabJson))
            throw new FileNotFoundException(
                $"BPE tokenizer requires vocab.json in '{directory}'.");

        using var vocabStream = File.OpenRead(vocabJson);
        using var mergesStream = File.Exists(mergesPath) ? File.OpenRead(mergesPath) : null;
        return BpeTokenizer.Create(vocabStream, mergesStream);
    }

    private static Tokenizer LoadFromVocabFile(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        using var stream = File.OpenRead(path);

        return ext switch
        {
            ".txt" => BertTokenizer.Create(stream),
            ".model" => LlamaTokenizer.Create(stream),
            _ => throw new NotSupportedException(
                $"Unsupported tokenizer file extension '{ext}'. " +
                $"Use .txt for BERT/WordPiece, .model for SentencePiece, " +
                $"or point at a directory with tokenizer_config.json for auto-detection.")
        };
    }

    private static void AddVectorColumn(
        Dictionary<string, SchemaShape.Column> schema,
        string name,
        DataViewType itemType)
    {
        var colCtor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];
        var col = (SchemaShape.Column)colCtor.Invoke([
            name,
            SchemaShape.Column.VectorKind.Vector,
            itemType,
            false,
            (SchemaShape?)null
        ]);
        schema[name] = col;
    }
}
