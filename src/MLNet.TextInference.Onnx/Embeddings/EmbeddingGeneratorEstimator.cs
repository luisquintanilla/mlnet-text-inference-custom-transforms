using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the provider-agnostic embedding generator transform.
/// </summary>
public class EmbeddingGeneratorOptions
{
    /// <summary>Name of the input text column. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output embedding column. Default: "Embedding".</summary>
    public string OutputColumnName { get; set; } = "Embedding";

    /// <summary>
    /// Batch size for embedding generation. Default: 32.
    /// </summary>
    public int BatchSize { get; set; } = 32;
}

/// <summary>
/// ML.NET IEstimator that wraps any IEmbeddingGenerator to produce embeddings within a pipeline.
/// Provider-agnostic — works with ONNX, OpenAI, Azure OpenAI, Ollama, or any MEAI implementation.
/// </summary>
public sealed class EmbeddingGeneratorEstimator : IEstimator<EmbeddingGeneratorTransformer>
{
    private readonly MLContext _mlContext;
    private readonly IEmbeddingGenerator<string, Embedding<float>> _generator;
    private readonly EmbeddingGeneratorOptions _options;

    public EmbeddingGeneratorEstimator(
        MLContext mlContext,
        IEmbeddingGenerator<string, Embedding<float>> generator,
        EmbeddingGeneratorOptions? options = null)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _options = options ?? new EmbeddingGeneratorOptions();
    }

    public EmbeddingGeneratorTransformer Fit(IDataView input)
    {
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        return new EmbeddingGeneratorTransformer(_mlContext, _generator, _options);
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
            SchemaShape.Column.VectorKind.Vector,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);
        result[_options.OutputColumnName] = outputCol;

        return new SchemaShape(result.Values);
    }
}

/// <summary>
/// ML.NET ITransformer that generates embeddings using any IEmbeddingGenerator.
/// Uses eager evaluation — IEmbeddingGenerator is inherently batch-oriented.
/// </summary>
public sealed class EmbeddingGeneratorTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly IEmbeddingGenerator<string, Embedding<float>> _generator;
    private readonly EmbeddingGeneratorOptions _options;

    public bool IsRowToRowMapper => true;

    internal IEmbeddingGenerator<string, Embedding<float>> Generator => _generator;

    internal EmbeddingGeneratorTransformer(
        MLContext mlContext,
        IEmbeddingGenerator<string, Embedding<float>> generator,
        EmbeddingGeneratorOptions options)
    {
        _mlContext = mlContext;
        _generator = generator;
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        var texts = ReadTextColumn(input);
        if (texts.Count == 0)
            return BuildOutputDataView(texts, []);

        var allEmbeddings = new List<float[]>(texts.Count);

        for (int start = 0; start < texts.Count; start += _options.BatchSize)
        {
            int count = Math.Min(_options.BatchSize, texts.Count - start);
            var batchTexts = texts.GetRange(start, count);

            var result = _generator.GenerateAsync(batchTexts).GetAwaiter().GetResult();

            foreach (var embedding in result)
            {
                allEmbeddings.Add(embedding.Vector.ToArray());
            }
        }

        return BuildOutputDataView(texts, allEmbeddings);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        // Dimension may vary by provider; use unknown vector size
        builder.AddColumn(_options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "Cannot save a provider-agnostic embedding transform. " +
            "Use OnnxTextEmbeddingTransformer.Save() for ONNX-backed transforms.");

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

    private IDataView BuildOutputDataView(List<string> texts, List<float[]> embeddings)
    {
        var rows = new List<EmbeddingRow>();

        for (int i = 0; i < texts.Count; i++)
        {
            rows.Add(new EmbeddingRow
            {
                Text = texts[i],
                Embedding = i < embeddings.Count ? embeddings[i] : []
            });
        }

        return _mlContext.Data.LoadFromEnumerable(rows);
    }

    private sealed class EmbeddingRow
    {
        public string Text { get; set; } = "";

        [VectorType]
        public float[] Embedding { get; set; } = [];
    }
}
