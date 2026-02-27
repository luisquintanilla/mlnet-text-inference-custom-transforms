using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the ONNX text model scorer transform.
/// Runs inference on a transformer-architecture ONNX model (BERT, MiniLM, etc.).
/// </summary>
public class OnnxTextModelScorerOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; set; }

    /// <summary>Name of the input token IDs column. Default: "TokenIds".</summary>
    public string TokenIdsColumnName { get; set; } = "TokenIds";

    /// <summary>Name of the input attention mask column. Default: "AttentionMask".</summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>
    /// Name of the input token type IDs column. Default: "TokenTypeIds".
    /// Set to null if the model doesn't use token type IDs.
    /// </summary>
    public string? TokenTypeIdsColumnName { get; set; } = "TokenTypeIds";

    /// <summary>Name of the output column for raw model output. Default: "RawOutput".</summary>
    public string OutputColumnName { get; set; } = "RawOutput";

    /// <summary>
    /// Maximum sequence length. Must match the tokenizer's MaxTokenLength.
    /// Default: 128.
    /// </summary>
    public int MaxTokenLength { get; set; } = 128;

    /// <summary>Batch size for ONNX inference. Default: 32.</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>ONNX input tensor name for token IDs. Null = auto-detect ("input_ids").</summary>
    public string? InputIdsTensorName { get; set; }

    /// <summary>ONNX input tensor name for attention mask. Null = auto-detect ("attention_mask").</summary>
    public string? AttentionMaskTensorName { get; set; }

    /// <summary>ONNX input tensor name for token type IDs. Null = auto-detect ("token_type_ids" if present).</summary>
    public string? TokenTypeIdsTensorName { get; set; }

    /// <summary>
    /// ONNX output tensor name. Null = auto-detect.
    /// Auto-detection prefers "sentence_embedding" / "pooler_output" (pre-pooled),
    /// falls back to "last_hidden_state" / "output" (unpooled).
    /// </summary>
    public string? OutputTensorName { get; set; }

    /// <summary>
    /// Optional GPU device ID to run execution on. Null = use MLContext.GpuDeviceId.
    /// Set to a non-negative integer (e.g. 0) to target a specific CUDA device.
    /// Requires the consuming application to reference Microsoft.ML.OnnxRuntime.Gpu.
    /// </summary>
    public int? GpuDeviceId { get; set; }

    /// <summary>
    /// If true and GPU initialization fails, fall back to CPU instead of throwing.
    /// Default: false.
    /// </summary>
    public bool FallbackToCpu { get; set; }

    /// <summary>
    /// Preferred ONNX output tensor names, searched in order.
    /// Used by task facades to hint the appropriate output for their task.
    /// Null = use generic auto-discovery.
    /// </summary>
    public string[]? PreferredOutputNames { get; set; }

    /// <summary>
    /// Additional ONNX output tensor names to include as separate columns.
    /// Used by QA models that produce start_logits and end_logits as separate outputs.
    /// </summary>
    public string[]? AdditionalOutputTensorNames { get; set; }

    /// <summary>
    /// Column names for additional output tensors. Must match AdditionalOutputTensorNames length.
    /// </summary>
    public string[]? AdditionalOutputColumnNames { get; set; }
}

/// <summary>
/// Discovered ONNX model tensor metadata. Immutable record.
/// </summary>
internal sealed record OnnxModelMetadata(
    string InputIdsName,
    string AttentionMaskName,
    string? TokenTypeIdsName,
    string OutputTensorName,
    int HiddenDim,
    bool HasPooledOutput,
    int OutputRank,
    string[]? AdditionalOutputNames = null,
    int[]? AdditionalOutputDims = null);

/// <summary>
/// ML.NET IEstimator that creates an OnnxTextModelScorerTransformer.
/// Fit() validates the input schema, loads the ONNX model, and auto-discovers tensor metadata.
/// </summary>
public sealed class OnnxTextModelScorerEstimator : IEstimator<OnnxTextModelScorerTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextModelScorerOptions _options;

    public OnnxTextModelScorerEstimator(MLContext mlContext, OnnxTextModelScorerOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");
    }

    public OnnxTextModelScorerTransformer Fit(IDataView input)
    {
        ValidateColumn(input.Schema, _options.TokenIdsColumnName);
        ValidateColumn(input.Schema, _options.AttentionMaskColumnName);
        if (_options.TokenTypeIdsColumnName != null)
            ValidateColumn(input.Schema, _options.TokenTypeIdsColumnName);

        var session = CreateInferenceSession();
        var metadata = DiscoverModelMetadata(session);

        return new OnnxTextModelScorerTransformer(_mlContext, _options, session, metadata);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
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

        if (_options.AdditionalOutputColumnNames != null)
        {
            foreach (var additionalColName in _options.AdditionalOutputColumnNames)
            {
                var additionalCol = (SchemaShape.Column)colCtor.Invoke([
                    additionalColName,
                    SchemaShape.Column.VectorKind.Vector,
                    (DataViewType)NumberDataViewType.Single,
                    false,
                    (SchemaShape?)null
                ]);
                result[additionalColName] = additionalCol;
            }
        }

        return new SchemaShape(result.Values);
    }

    internal OnnxModelMetadata DiscoverModelMetadata(InferenceSession session)
    {
        var inputMeta = session.InputMetadata;
        var outputMeta = session.OutputMetadata;

        string inputIdsName = _options.InputIdsTensorName
            ?? FindTensorName(inputMeta, ["input_ids"], "input_ids");
        string attentionMaskName = _options.AttentionMaskTensorName
            ?? FindTensorName(inputMeta, ["attention_mask"], "attention_mask");
        string? tokenTypeIdsName = _options.TokenTypeIdsTensorName
            ?? TryFindTensorName(inputMeta, ["token_type_ids"]);

        string outputName;
        bool hasPooledOutput;
        int hiddenDim;
        int outputRank;

        if (_options.OutputTensorName != null)
        {
            outputName = _options.OutputTensorName;
            var dims = outputMeta[outputName].Dimensions;
            hasPooledOutput = !dims.Contains(-1) && dims.Length == 2;
            hiddenDim = (int)dims.Last();
            outputRank = dims.Length;
        }
        else if (_options.PreferredOutputNames != null)
        {
            var preferred = TryFindTensorName(outputMeta, _options.PreferredOutputNames);
            if (preferred != null)
            {
                outputName = preferred;
                var dims = outputMeta[preferred].Dimensions;
                hasPooledOutput = !dims.Contains(-1) && dims.Length == 2;
                hiddenDim = (int)dims.Last();
                outputRank = dims.Length;
            }
            else
            {
                outputName = outputMeta.Keys.First();
                var dims = outputMeta[outputName].Dimensions;
                hasPooledOutput = !dims.Contains(-1) && dims.Length == 2;
                hiddenDim = (int)dims.Last();
                outputRank = dims.Length;
            }
        }
        else
        {
            var pooledName = TryFindTensorName(outputMeta, ["sentence_embedding", "pooler_output"]);
            if (pooledName != null)
            {
                outputName = pooledName;
                hasPooledOutput = true;
                hiddenDim = (int)outputMeta[pooledName].Dimensions.Last();
                outputRank = 2;
            }
            else
            {
                outputName = FindTensorName(outputMeta,
                    ["last_hidden_state", "output", "hidden_states"],
                    outputMeta.Keys.First());
                hasPooledOutput = false;
                hiddenDim = (int)outputMeta[outputName].Dimensions.Last();
                outputRank = 3;
            }
        }

        if (hiddenDim <= 0)
            throw new InvalidOperationException(
                $"Could not determine hidden dimension from ONNX output '{outputName}'.");

        // Discover additional output tensors if configured
        string[]? additionalOutputNames = null;
        int[]? additionalOutputDims = null;

        if (_options.AdditionalOutputTensorNames != null)
        {
            if (_options.AdditionalOutputColumnNames == null ||
                _options.AdditionalOutputColumnNames.Length != _options.AdditionalOutputTensorNames.Length)
                throw new ArgumentException(
                    "AdditionalOutputColumnNames must be set with the same length as AdditionalOutputTensorNames.");

            additionalOutputNames = new string[_options.AdditionalOutputTensorNames.Length];
            additionalOutputDims = new int[_options.AdditionalOutputTensorNames.Length];

            for (int i = 0; i < _options.AdditionalOutputTensorNames.Length; i++)
            {
                var name = _options.AdditionalOutputTensorNames[i];
                if (!outputMeta.ContainsKey(name))
                    throw new InvalidOperationException(
                        $"Additional output tensor '{name}' not found in ONNX model outputs. " +
                        $"Available: [{string.Join(", ", outputMeta.Keys)}]");

                additionalOutputNames[i] = name;
                var dims = outputMeta[name].Dimensions;
                int lastDim = (int)dims.Last();
                if (lastDim <= 0)
                    additionalOutputDims[i] = _options.MaxTokenLength;
                else if (dims.Length == 2)
                    additionalOutputDims[i] = lastDim;
                else
                    additionalOutputDims[i] = _options.MaxTokenLength * lastDim;
            }
        }

        return new OnnxModelMetadata(
            inputIdsName, attentionMaskName, tokenTypeIdsName,
            outputName, hiddenDim, hasPooledOutput, outputRank,
            additionalOutputNames, additionalOutputDims);
    }

    /// <summary>
    /// Overload that creates a temporary session using the configured SessionOptions
    /// to discover model metadata. Used by GetOutputSchema() scenarios.
    /// </summary>
    internal OnnxModelMetadata DiscoverModelMetadata()
    {
        using var session = CreateInferenceSession();
        return DiscoverModelMetadata(session);
    }

    /// <summary>
    /// Creates an InferenceSession with GPU support if configured.
    /// If FallbackToCpu is true, catches CUDA failures and retries with CPU-only options.
    /// </summary>
    private InferenceSession CreateInferenceSession()
    {
        var (sessionOptions, fallbackToCpu) = CreateSessionOptions();

        try
        {
            return new InferenceSession(_options.ModelPath, sessionOptions);
        }
        catch (OnnxRuntimeException) when (fallbackToCpu)
        {
            // CUDA initialization failed (invalid device, driver mismatch, etc.)
            // Fall back to CPU-only session.
            return new InferenceSession(_options.ModelPath, new SessionOptions());
        }
    }

    private (SessionOptions options, bool fallbackToCpu) CreateSessionOptions()
    {
        // Resolve GPU device: per-estimator option → MLContext.GpuDeviceId → null (CPU)
        int? deviceId = _options.GpuDeviceId ?? _mlContext.GpuDeviceId;
        bool fallbackToCpu = _options.FallbackToCpu;

        // If MLContext provides FallbackToCpu and no per-estimator override was set,
        // inherit the context-level setting.
        if (_options.GpuDeviceId == null && _mlContext.GpuDeviceId != null)
            fallbackToCpu = _mlContext.FallbackToCpu;

        var options = new SessionOptions();

        if (deviceId.HasValue)
        {
            try
            {
                options.AppendExecutionProvider_CUDA(deviceId.Value);
            }
            catch (Exception) when (fallbackToCpu)
            {
                // CUDA libraries not available — fall back to CPU
            }
        }

        return (options, fallbackToCpu);
    }

    private static string FindTensorName(
        IReadOnlyDictionary<string, NodeMetadata> metadata,
        string[] candidates,
        string fallback)
    {
        return TryFindTensorName(metadata, candidates) ?? fallback;
    }

    private static string? TryFindTensorName(
        IReadOnlyDictionary<string, NodeMetadata> metadata,
        string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            if (metadata.ContainsKey(candidate))
                return candidate;
        }
        return null;
    }

    private static void ValidateColumn(DataViewSchema schema, string columnName)
    {
        if (schema.GetColumnOrNull(columnName) == null)
            throw new ArgumentException(
                $"Input schema does not contain required column '{columnName}'.");
    }
}
