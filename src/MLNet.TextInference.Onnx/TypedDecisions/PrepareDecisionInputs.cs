using Microsoft.ML.Tokenizers;

namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// Builds the exact Laya request sequence and the five dense graph inputs.
/// </summary>
internal sealed class PrepareDecisionInputs
{
    private readonly LayaDecisionProfile _profile;
    private readonly Tokenizer _tokenizer;
    private readonly LayaTokenizerMetadata _tokenizerMetadata;

    /// <summary>
    /// Creates a preparation stage with the selected profile and tokenizer contract.
    /// </summary>
    /// <param name="profile">The profile limits and model-specific preparation policy.</param>
    /// <param name="tokenizer">The Microsoft tokenizer engine used by the profile.</param>
    /// <param name="tokenizerMetadata">The profile-specific special-token metadata.</param>
    public PrepareDecisionInputs(
        LayaDecisionProfile profile,
        Tokenizer tokenizer,
        LayaTokenizerMetadata tokenizerMetadata)
    {
        _profile = profile ?? throw new ArgumentNullException(nameof(profile));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _tokenizerMetadata = tokenizerMetadata ?? throw new ArgumentNullException(nameof(tokenizerMetadata));
        if (_profile.MaxLength <= 0 || _profile.HeadMaxLength <= 0)
            throw new ArgumentException("The profile dimensions must be positive.", nameof(profile));
    }

    public DecisionInputBatch Prepare(string state, IReadOnlyList<DecisionQuestion> questions)
        => Prepare([DecisionRequest.Create(state, questions)]);

    public DecisionInputBatch Prepare(IReadOnlyList<DecisionRequest> requests)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0)
            throw new ArgumentException("At least one request is required.", nameof(requests));

        var rows = new List<PreparedRow>();
        for (int requestIndex = 0; requestIndex < requests.Count; requestIndex++)
        {
            var request = requests[requestIndex] ?? throw new ArgumentException(
                $"Request {requestIndex} is null.", nameof(requests));

            foreach (var question in request.Questions)
            {
                question.Validate();
                rows.Add(BuildRow(requestIndex, request.State, question));
            }
        }

        if (rows.Count == 0)
            throw new ArgumentException("At least one question is required.", nameof(requests));

        int sequenceLength = rows.Max(static row => row.Ids.Length);
        int markerWidth = rows.Max(static row => row.MarkerPositions.Length);
        sequenceLength = Math.Min(sequenceLength, _profile.MaxLength);
        markerWidth = Math.Min(markerWidth, _profile.HeadMaxLength);
        var inputIds = new long[rows.Count * sequenceLength];
        Array.Fill(inputIds, (long)_tokenizerMetadata.PadTokenId);
        var attention = new long[inputIds.Length];
        var markerPositions = new long[rows.Count * markerWidth];
        var markerMask = new bool[markerPositions.Length];
        var qtypes = new long[rows.Count];
        var items = new DecisionInputItem[rows.Count];

        for (int rowIndex = 0; rowIndex < rows.Count; rowIndex++)
        {
            var row = rows[rowIndex];
            if (row.MarkerPositions.Length > markerWidth)
                throw new InvalidOperationException(
                    $"Question '{row.Question.Id}' has {row.MarkerPositions.Length} options, " +
                    $"which exceeds head_max_len={_profile.HeadMaxLength}.");
            if (row.MarkerPositions.Length != row.OptionLabels.Length)
                throw new InvalidOperationException(
                    $"Question '{row.Question.Id}' has options that do not fit in max_len={_profile.MaxLength}.");

            int idOffset = rowIndex * sequenceLength;
            int copyLength = Math.Min(row.Ids.Length, sequenceLength);
            Array.Copy(row.Ids, 0, inputIds, idOffset, copyLength);
            for (int tokenIndex = 0; tokenIndex < copyLength; tokenIndex++)
                attention[idOffset + tokenIndex] = 1;

            int markerOffset = rowIndex * markerWidth;
            for (int markerIndex = 0; markerIndex < row.MarkerPositions.Length; markerIndex++)
            {
                var marker = row.MarkerPositions[markerIndex];
                if (marker >= sequenceLength)
                    throw new InvalidOperationException(
                        $"Question '{row.Question.Id}' marker {marker} was truncated by max_len={_profile.MaxLength}.");
                markerPositions[markerOffset + markerIndex] = marker;
                markerMask[markerOffset + markerIndex] = true;
            }

            qtypes[rowIndex] = row.QuestionType;
            items[rowIndex] = new DecisionInputItem(
                row.RequestIndex,
                row.Question,
                row.MarkerPositions,
                row.OptionLabels,
                row.QuestionType);
        }

        var batch = new DecisionInputBatch
        {
            BatchSize = rows.Count,
            SequenceLength = sequenceLength,
            MarkerWidth = markerWidth,
            InputIds = inputIds,
            AttentionMask = attention,
            MarkerPositions = markerPositions,
            MarkerMask = markerMask,
            QuestionTypes = qtypes,
            Items = items
        };
        batch.Validate();
        return batch;
    }

    private PreparedRow BuildRow(int requestIndex, string state, DecisionQuestion question)
    {
        var options = question.RenderOptions();
        var optionIds = options.Select(option =>
        {
            var scrubbed = option.Replace(_tokenizerMetadata.MaskToken, " ", StringComparison.Ordinal);
            var encoded = Encode(" " + scrubbed, 49);
            return new[] { _tokenizerMetadata.MaskTokenId }.Concat(encoded).ToArray();
        }).ToArray();

        int optionTokenCount = optionIds.Sum(static ids => ids.Length);
        int optionBudget = _profile.HeadMaxLength - optionTokenCount;
        if (optionBudget < 16)
        {
            int perOption = Math.Max(
                4,
                (_profile.HeadMaxLength - 16) / Math.Max(1, optionIds.Length));
            optionIds = optionIds.Select(ids => ids.Take(perOption).ToArray()).ToArray();
            optionBudget = _profile.HeadMaxLength - optionIds.Sum(static ids => ids.Length);
        }

        var instructions = question.Instructions.Replace(_tokenizerMetadata.MaskToken, " ", StringComparison.Ordinal);
        var head = Encode(
            $"{QuestionTypeName(question.Type)} question: {instructions}",
            Math.Max(8, optionBudget));
        var ids = new List<int>(_profile.MaxLength)
        {
            _tokenizerMetadata.ClsTokenId
        };
        ids.AddRange(head);
        ids.Add(_tokenizerMetadata.SepTokenId);

        var markers = new List<int>(optionIds.Length);
        foreach (var option in optionIds)
        {
            markers.Add(ids.Count);
            ids.AddRange(option);
        }

        ids.Add(_tokenizerMetadata.SepTokenId);
        int stateRoom = Math.Max(0, _profile.MaxLength - ids.Count - 1);
        var stateIds = Encode(
            state.Replace(_tokenizerMetadata.MaskToken, " ", StringComparison.Ordinal),
            stateRoom);
        ids.AddRange(stateIds);
        ids.Add(_tokenizerMetadata.SepTokenId);

        if (ids.Count > _profile.MaxLength)
            ids.RemoveRange(_profile.MaxLength, ids.Count - _profile.MaxLength);

        return new PreparedRow(
            requestIndex,
            question,
            ids.ToArray(),
            markers.Where(marker => marker < _profile.MaxLength).ToArray(),
            question.OptionLabels().ToArray(),
            (long)question.Type);
    }

    private IReadOnlyList<int> Encode(string text, int maxTokenCount)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (maxTokenCount < 0)
            throw new ArgumentOutOfRangeException(nameof(maxTokenCount));
        if (maxTokenCount == 0)
            return [];

        return _tokenizer.EncodeToIds(
            text,
            maxTokenCount,
            out _,
            out _,
            considerPreTokenization: true,
            considerNormalization: true);
    }

    private static string QuestionTypeName(DecisionQuestionType type)
        => type switch
        {
            DecisionQuestionType.Choice => "choice",
            DecisionQuestionType.Score => "score",
            DecisionQuestionType.Noul => "noul",
            _ => throw new ArgumentOutOfRangeException(nameof(type))
        };

    private sealed record PreparedRow(
        int RequestIndex,
        DecisionQuestion Question,
        int[] Ids,
        int[] MarkerPositions,
        string[] OptionLabels,
        long QuestionType);
}
