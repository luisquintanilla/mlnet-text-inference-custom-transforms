namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// Standalone facade that composes preparation, ONNX scoring, and typed decoding.
/// </summary>
public sealed class OnnxTypedDecisions : IDisposable
{
    private readonly TypedDecisionBundle? _ownedBundle;
    private readonly ScoreOnnxDecisionModel _scorer;
    private bool _disposed;

    public OnnxTypedDecisions(TypedDecisionBundle bundle)
    {
        ArgumentNullException.ThrowIfNull(bundle);
        Profile = bundle.Profile;
        PrepareStage = new PrepareDecisionInputs(bundle.Profile, bundle.Tokenizer);
        _scorer = new ScoreOnnxDecisionModel(bundle);
        DecodeStage = new DecodeDecisions(
            bundle.Profile.TemperaturePolicy,
            bundle.Manifest.Decoder);
    }

    public OnnxTypedDecisions(string bundlePath)
    {
        var bundle = TypedDecisionBundle.Open(bundlePath);
        try
        {
            _ownedBundle = bundle;
            Profile = bundle.Profile;
            PrepareStage = new PrepareDecisionInputs(bundle.Profile, bundle.Tokenizer);
            _scorer = new ScoreOnnxDecisionModel(bundle);
            DecodeStage = new DecodeDecisions(
                bundle.Profile.TemperaturePolicy,
                bundle.Manifest.Decoder);
        }
        catch
        {
            bundle.Dispose();
            throw;
        }
    }

    public LayaDecisionProfile Profile { get; }
    public PrepareDecisionInputs PrepareStage { get; }
    public DecodeDecisions DecodeStage { get; }
    public IReadOnlyList<TypedDecisionDiagnostic> Diagnostics => Profile.Diagnostics;

    public DecisionResponse Infer(string state, IReadOnlyList<DecisionQuestion> questions)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Infer([DecisionRequest.Create(state, questions)])[0];
    }

    public IReadOnlyList<DecisionResponse> Infer(IReadOnlyList<DecisionRequest> requests)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(requests);
        var inputs = PrepareStage.Prepare(requests);
        var outputs = _scorer.Score(inputs);
        var decoded = DecodeStage.Decode(inputs, outputs);
        var responses = requests.Select(static _ => new ResponseBuilder()).ToArray();
        for (int row = 0; row < inputs.Items.Count; row++)
        {
            var requestIndex = inputs.Items[row].RequestIndex;
            responses[requestIndex].Results.Add(decoded.Results[row]);
            var attention = inputs.AttentionMask.AsSpan(
                row * inputs.SequenceLength,
                inputs.SequenceLength);
            for (int token = 0; token < attention.Length; token++)
            {
                if (attention[token] != 0)
                    responses[requestIndex].InputTokenCount++;
            }
        }

        return responses.Select(static builder => builder.Build()).ToArray();
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _scorer.Dispose();
        _ownedBundle?.Dispose();
    }

    private sealed class ResponseBuilder
    {
        public List<DecisionResult> Results { get; } = [];
        public int InputTokenCount { get; set; }

        public DecisionResponse Build() => new()
        {
            Results = Results,
            InputTokenCount = InputTokenCount
        };
    }
}
