using System.Text;
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
    ///   <item>A <c>tokenizer.json</c> file (HuggingFace fast tokenizer) — supports BPE and WordPiece models</item>
    ///   <item>A vocab file: <c>.txt</c> (BERT/WordPiece), <c>.model</c> (SentencePiece)</item>
    ///   <item>A directory containing <c>tokenizer.json</c> (used as last-resort fallback when no other tokenizer files are present)</item>
    /// </list>
    /// Used only when <see cref="Tokenizer"/> is not set.
    /// </summary>
    public string? TokenizerPath { get; set; }

    /// <summary>Name of the input text column. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>
    /// Optional second text input column for text-pair models (cross-encoders, QA).
    /// When set, tokens from InputColumnName get token_type_ids=0 and tokens from
    /// SecondInputColumnName get token_type_ids=1, separated by [SEP].
    /// </summary>
    public string? SecondInputColumnName { get; set; }

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

    /// <summary>
    /// If true, output token character offsets. Required for NER.
    /// Default: false.
    /// </summary>
    public bool OutputOffsets { get; set; }

    /// <summary>Name of the output column for token start character offsets. Default: "TokenStartOffsets".</summary>
    public string TokenStartOffsetsColumnName { get; set; } = "TokenStartOffsets";

    /// <summary>Name of the output column for token end character offsets. Default: "TokenEndOffsets".</summary>
    public string TokenEndOffsetsColumnName { get; set; } = "TokenEndOffsets";

    // Special token IDs populated during tokenizer loading from tokenizer_config.json.
    // Used by text-pair tokenization to manually inject [CLS]/[SEP] tokens.
    internal int? BosTokenId { get; set; }
    internal int? SepTokenId { get; set; }
    internal bool DoubleSeparator { get; set; }
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

        if (_options.SecondInputColumnName != null)
        {
            var col2 = input.Schema.GetColumnOrNull(_options.SecondInputColumnName);
            if (col2 == null)
                throw new ArgumentException(
                    $"Input schema does not contain column '{_options.SecondInputColumnName}'.");
        }

        var tokenizer = _options.Tokenizer ?? LoadTokenizer(_options.TokenizerPath!);

        if ((_options.BosTokenId == null || _options.SepTokenId == null) && _options.TokenizerPath != null)
            PopulateSpecialTokens(_options.TokenizerPath, _options);

        if (_options.SecondInputColumnName != null
            && (_options.BosTokenId == null || _options.SepTokenId == null))
        {
            throw new InvalidOperationException(
                "Text-pair tokenization requires special token IDs (BOS/CLS and SEP) but they could not be resolved. " +
                "Ensure TokenizerPath points to a directory containing tokenizer_config.json with " +
                "cls_token/sep_token definitions or added_tokens_decoder.");
        }

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
        if (_options.OutputOffsets)
        {
            AddVectorColumn(result, _options.TokenStartOffsetsColumnName, NumberDataViewType.Int64);
            AddVectorColumn(result, _options.TokenEndOffsetsColumnName, NumberDataViewType.Int64);
        }

        return new SchemaShape(result.Values);
    }

    /// <summary>
    /// Resolves a tokenizer from a path. Supports:
    /// <list type="bullet">
    ///   <item>Directory with <c>tokenizer_config.json</c> → reads <c>tokenizer_class</c>, loads sibling files</item>
    ///   <item>Directory without config → scans for known vocab files, falls back to <c>tokenizer.json</c></item>
    ///   <item><c>tokenizer_config.json</c> file → reads config, loads sibling files</item>
    ///   <item><c>tokenizer.json</c> file (HuggingFace fast tokenizer) → supports BPE and WordPiece models</item>
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

        if (fileName == "tokenizer.json")
            return LoadFromHuggingFaceTokenizerJson(path);

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

        var spmModel = Path.Combine(directory, "spm.model");
        if (File.Exists(spmModel))
            return LoadFromVocabFile(spmModel);

        // Last resort: check for HuggingFace fast tokenizer file
        var tokenizerJson = Path.Combine(directory, "tokenizer.json");
        if (File.Exists(tokenizerJson))
            return LoadFromHuggingFaceTokenizerJson(tokenizerJson);

        throw new FileNotFoundException(
            $"No tokenizer_config.json or known vocab file found in '{directory}'. " +
            $"Expected one of: tokenizer_config.json, vocab.txt, tokenizer.model, sentencepiece.bpe.model, spm.model, tokenizer.json.");
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
            "DebertaTokenizer" => LoadSentencePieceFromDirectory(directory),
            "DebertaV2Tokenizer" => LoadSentencePieceFromDirectory(directory),
            "GPT2Tokenizer" => LoadBpeFromDirectory(directory),
            "RobertaTokenizer" => LoadBpeFromDirectory(directory),
            _ when !string.IsNullOrEmpty(tokenizerClass) => throw new NotSupportedException(
                $"Unsupported tokenizer_class '{tokenizerClass}' in {configPath}. " +
                $"Supported: BertTokenizer, DebertaV2Tokenizer, XLMRobertaTokenizer, LlamaTokenizer, GPT2Tokenizer, RobertaTokenizer. " +
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
        var candidates = new[] { "sentencepiece.bpe.model", "tokenizer.model", "spiece.model", "spm.model" };
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

    /// <summary>
    /// Loads a tokenizer from a HuggingFace <c>tokenizer.json</c> fast tokenizer file.
    /// Supports BPE and WordPiece model types. Throws <see cref="NotSupportedException"/>
    /// for Unigram models (use the <c>.model</c> protobuf file instead).
    /// </summary>
    private static Tokenizer LoadFromHuggingFaceTokenizerJson(string path)
    {
        var json = File.ReadAllText(path);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (!root.TryGetProperty("model", out var model))
            throw new InvalidOperationException(
                $"tokenizer.json at '{path}' has no 'model' property. " +
                $"Ensure this is a valid HuggingFace fast tokenizer file.");

        if (!model.TryGetProperty("type", out var typeElement))
            throw new InvalidOperationException(
                $"tokenizer.json at '{path}' has no 'model.type' property.");

        var modelType = typeElement.GetString()
            ?? throw new InvalidOperationException(
                $"tokenizer.json at '{path}' has null 'model.type' property.");

        return modelType switch
        {
            "BPE" => LoadBpeFromTokenizerJson(model, path),
            "WordPiece" => LoadWordPieceFromTokenizerJson(model, root, path),
            "Unigram" => throw new NotSupportedException(
                $"Unigram tokenizer.json is not directly supported. " +
                $"Point TokenizerPath at the directory containing the .model file instead, " +
                $"or set the Tokenizer property directly."),
            _ => throw new NotSupportedException(
                $"Unsupported tokenizer model type '{modelType}' in '{path}'. " +
                $"Supported types: BPE, WordPiece.")
        };
    }

    private static Tokenizer LoadBpeFromTokenizerJson(JsonElement model, string path)
    {
        if (!model.TryGetProperty("vocab", out var vocabElement))
            throw new InvalidOperationException(
                $"BPE model in '{path}' has no 'vocab' property.");

        // model.vocab is a JSON dict (same format as vocab.json) — pass raw JSON directly
        var vocabJson = vocabElement.GetRawText();
        using var vocabStream = new MemoryStream(Encoding.UTF8.GetBytes(vocabJson));

        MemoryStream? mergesStream = null;
        if (model.TryGetProperty("merges", out var mergesElement)
            && mergesElement.ValueKind == JsonValueKind.Array
            && mergesElement.GetArrayLength() > 0)
        {
            var sb = new StringBuilder();
            foreach (var merge in mergesElement.EnumerateArray())
            {
                var mergeString = merge.GetString()
                    ?? throw new InvalidOperationException(
                        $"BPE model in '{path}' has a null entry in 'merges', which is not allowed.");
                sb.AppendLine(mergeString);
            }
            mergesStream = new MemoryStream(Encoding.UTF8.GetBytes(sb.ToString()));
        }

        try
        {
            return BpeTokenizer.Create(vocabStream, mergesStream);
        }
        finally
        {
            mergesStream?.Dispose();
        }
    }

    private static Tokenizer LoadWordPieceFromTokenizerJson(JsonElement model, JsonElement root, string path)
    {
        if (!model.TryGetProperty("vocab", out var vocabElement))
            throw new InvalidOperationException(
                $"WordPiece model in '{path}' has no 'vocab' property.");

        // Check normalizer for lowercase setting (e.g. bert-base-uncased)
        var lowerCase = false;
        if (root.TryGetProperty("normalizer", out var normalizer)
            && normalizer.ValueKind == JsonValueKind.Object
            && normalizer.TryGetProperty("lowercase", out var lc)
            && lc.ValueKind == JsonValueKind.True)
        {
            lowerCase = true;
        }

        // model.vocab is {"token": id, ...} — sort by ID and write as vocab.txt format (one token per line)
        var vocabPairs = new SortedDictionary<int, string>();
        foreach (var prop in vocabElement.EnumerateObject())
            vocabPairs[prop.Value.GetInt32()] = prop.Name;

        var sb = new StringBuilder();
        foreach (var kvp in vocabPairs)
            sb.AppendLine(kvp.Value);

        using var vocabStream = new MemoryStream(Encoding.UTF8.GetBytes(sb.ToString()));
        return BertTokenizer.Create(vocabStream, new BertOptions { LowerCaseBeforeTokenization = lowerCase });
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

    /// <summary>
    /// Extracts special token IDs (BOS/CLS and SEP) from tokenizer_config.json.
    /// These are needed for manual special token injection in text-pair tokenization.
    /// </summary>
    private static void PopulateSpecialTokens(string tokenizerPath, TextTokenizerOptions options)
    {
        string? configPath = null;
        if (Directory.Exists(tokenizerPath))
            configPath = Path.Combine(tokenizerPath, "tokenizer_config.json");
        else if (Path.GetFileName(tokenizerPath).Equals("tokenizer_config.json", StringComparison.OrdinalIgnoreCase))
            configPath = tokenizerPath;

        if (configPath == null || !File.Exists(configPath))
            return;

        var json = File.ReadAllText(configPath);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var clsTokenStr = root.TryGetProperty("cls_token", out var cls) ? cls.GetString() : null;
        var sepTokenStr = root.TryGetProperty("sep_token", out var sep) ? sep.GetString() : null;

        // Resolve token string → ID via added_tokens_decoder
        if (clsTokenStr != null && sepTokenStr != null
            && root.TryGetProperty("added_tokens_decoder", out var decoder))
        {
            foreach (var entry in decoder.EnumerateObject())
            {
                if (!int.TryParse(entry.Name, out int tokenId)) continue;
                var content = entry.Value.TryGetProperty("content", out var c) ? c.GetString() : null;
                if (content == null) continue;

                if (content == clsTokenStr && options.BosTokenId == null)
                    options.BosTokenId = tokenId;
                if (content == sepTokenStr && options.SepTokenId == null)
                    options.SepTokenId = tokenId;
            }
        }

        // Fallback: well-known defaults by tokenizer class
        if (options.BosTokenId == null || options.SepTokenId == null)
        {
            var tokenizerClass = root.TryGetProperty("tokenizer_class", out var tc) ? tc.GetString() ?? "" : "";
            if (tokenizerClass.EndsWith("Fast", StringComparison.Ordinal))
                tokenizerClass = tokenizerClass[..^4];

            (int? defaultBos, int? defaultSep) = tokenizerClass switch
            {
                "BertTokenizer" or "DistilBertTokenizer" => ((int?)101, (int?)102),
                "RobertaTokenizer" or "GPT2Tokenizer" => ((int?)0, (int?)2),
                "DebertaTokenizer" or "DebertaV2Tokenizer" => ((int?)1, (int?)2),
                "XLMRobertaTokenizer" => ((int?)0, (int?)2),
                _ => ((int?)null, (int?)null)
            };
            options.BosTokenId ??= defaultBos;
            options.SepTokenId ??= defaultSep;
        }

        // RoBERTa-family uses double separator between segments: <s> A </s></s> B </s>
        var tokClass = root.TryGetProperty("tokenizer_class", out var t) ? t.GetString() ?? "" : "";
        if (tokClass.EndsWith("Fast", StringComparison.Ordinal))
            tokClass = tokClass[..^4];
        options.DoubleSeparator = tokClass is "RobertaTokenizer" or "GPT2Tokenizer" or "XLMRobertaTokenizer";
    }
}
