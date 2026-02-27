# Future Task Expansion

## The Platform Vision

The `MLNet.TextInference.Onnx` platform provides a reusable foundation for ANY transformer-based ONNX model task in ML.NET. The tokenizer and scorer are shared; only the post-processing varies by task.

```
                    TextTokenizerTransformer
                            │
                            ▼
                 OnnxTextModelScorerTransformer       (task-agnostic)
                            │
            ┌───────────────┼────────────────┐
            │               │                │
     EmbeddingPooling  SoftmaxClassify  (more tasks)
     Transformer       Transformer      coming soon
```

## Current Task Status

| Task | Status | Post-processor | Facade |
|------|--------|---------------|--------|
| Embeddings | ✅ Implemented | `EmbeddingPoolingTransformer` | `OnnxTextEmbeddingEstimator` |
| Classification | 🔲 Planned | `SoftmaxClassificationTransformer` | `OnnxTextClassificationEstimator` |
| Reranking | 🔲 Planned | `SigmoidScorerTransformer` | `OnnxRerankerEstimator` |
| NER | 🔲 Planned | `NerDecodingTransformer` | `OnnxNerEstimator` |
| QA | 🔲 Planned | `QaSpanExtractionTransformer` | `OnnxQaEstimator` |
| Text Generation | 🔲 Planned | `ChatClientTransformer` | N/A |

For the step-by-step guide on adding a new task, see [docs/extending.md](../docs/extending.md#how-to-add-a-new-task).

## Task: Text Classification

### Model
Any BERT-derived model fine-tuned for classification. Output: `[batch, num_classes]` logits.

### Post-Processing Transform: SoftmaxClassificationTransformer

```csharp
public class SoftmaxClassificationOptions
{
    public string InputColumnName { get; set; } = "RawOutput";
    public string ProbabilitiesColumnName { get; set; } = "Probabilities";
    public string PredictedLabelColumnName { get; set; } = "PredictedLabel";
    public string[]? Labels { get; set; }  // e.g., ["negative", "positive"]
}
```

**Transform logic:**
1. Read `RawOutput` (float[numClasses] logits)
2. Apply softmax: `probabilities[i] = exp(logits[i]) / Σ exp(logits[j])`
3. Argmax: `predictedIndex = argmax(probabilities)`
4. Map to label: `predictedLabel = Labels[predictedIndex]`
5. Output: Probabilities column + PredictedLabel column

**SIMD acceleration:** `TensorPrimitives.SoftMax()` exists in `System.Numerics.Tensors`.

### Pipeline

```csharp
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.SoftmaxClassify(new SoftmaxClassificationOptions
    {
        Labels = ["negative", "neutral", "positive"]
    }));
```

### Convenience Facade

```csharp
var estimator = mlContext.Transforms.OnnxTextClassification(new OnnxTextClassificationOptions
{
    ModelPath = "sentiment-model.onnx",
    TokenizerPath = "vocab.txt",
    Labels = ["negative", "neutral", "positive"]
});
```

## Task: Named Entity Recognition (NER)

### Model
Token-classification model. Output: `[batch, seq_len, num_labels]` per-token logits.

### Post-Processing Transform: NerDecodingTransformer

```csharp
public class NerDecodingOptions
{
    public string InputColumnName { get; set; } = "RawOutput";
    public string OutputColumnName { get; set; } = "Entities";
    public string TokenIdsColumnName { get; set; } = "TokenIds";
    public string[]? Labels { get; set; }  // BIO labels: ["O", "B-PER", "I-PER", "B-ORG", ...]
}
```

**Transform logic:**
1. Read `RawOutput` (float[seqLen × numLabels]) per row
2. For each token position: argmax over labels to get predicted BIO tag
3. Decode BIO sequence into entity spans (merging B- and I- tags)
4. Map token spans back to text spans (requires tokenizer offset info)
5. Output: structured entity list per row

### Challenges
- Needs token-to-text offset mapping from the tokenizer
- BIO decoding has edge cases (I- tag without preceding B-)
- Multi-word entities require merging

### Pipeline

```csharp
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.DecodeNer(nerOpts));
```

## Task: Semantic Similarity (Cross-Encoder)

### Model
Cross-encoder model that takes a text pair and outputs a similarity score. Uses `token_type_ids` to separate the two segments.

### Differences from Single-Text Pipeline
The tokenizer needs to handle **text pairs**:
- Segment A tokens get `token_type_ids = 0`
- `[SEP]` token
- Segment B tokens get `token_type_ids = 1`

This requires extending `TextTokenizerTransformer` to accept a second text column, or creating a `TextPairTokenizerTransformer`.

### Post-Processing Transform: SigmoidScorerTransformer

```csharp
public class SigmoidScorerOptions
{
    public string InputColumnName { get; set; } = "RawOutput";
    public string OutputColumnName { get; set; } = "Score";
}
```

**Transform logic:**
1. Read `RawOutput` (float[1] logit)
2. Apply sigmoid: `score = 1 / (1 + exp(-logit))`
3. Output: single float score per row

### Pipeline

```csharp
var pipeline = mlContext.Transforms.TokenizeTextPair(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.SigmoidScore(sigmoidOpts));
```

## Task: Question Answering (Extractive)

### Model
QA model that takes question + context and outputs start/end position logits.

### Post-Processing Transform: QaSpanExtractionTransformer

```csharp
public class QaSpanExtractionOptions
{
    public string StartLogitsColumnName { get; set; } = "RawOutput";  // or separate columns
    public string OutputColumnName { get; set; } = "Answer";
    public string TokenIdsColumnName { get; set; } = "TokenIds";
    public int MaxAnswerLength { get; set; } = 30;
}
```

**Transform logic:**
1. Read start and end logits (may be separate ONNX outputs or interleaved)
2. Find best (start, end) pair where `end >= start` and `end - start < maxLength`
3. Extract token span from TokenIds
4. Decode tokens back to text
5. Output: answer text + confidence score

### Challenges
- May require multiple ONNX output tensors (start_logits, end_logits)
- Current scorer outputs a single tensor — may need extension for multi-output models
- Token-to-text decoding requires tokenizer's decode capability

## Task: Reranking

### Model
Cross-encoder model that scores query-document relevance.

### Pipeline
Same as cross-encoder similarity, but used in a retrieval context:

```csharp
// Score all query-document pairs
var pipeline = mlContext.Transforms.TokenizeTextPair(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.SigmoidScore(sigmoidOpts));

// Sort by score descending
var reranked = mlContext.Data.CreateEnumerable<RankedResult>(result)
    .OrderByDescending(r => r.Score)
    .ToList();
```

## Tokenizer Extensions Required

Several future tasks require tokenizer enhancements:

| Feature | Tasks That Need It | Complexity |
|---------|-------------------|-----------|
| Text pair tokenization (`token_type_ids` = 0/1) | Cross-encoder, QA, reranking | Medium |
| Token offset tracking (char positions) | NER, QA (answer extraction) | Medium |
| Decode (token IDs → text) | NER, QA | Low |
| Special token insertion ([CLS], [SEP]) | All (currently implicit in BertTokenizer) | Already handled |

These would be added to `TextTokenizerTransformer` as optional capabilities, not as separate transforms.

## Scorer Extensions Required

| Feature | Tasks That Need It | Complexity |
|---------|-------------------|-----------|
| Multi-output tensor support | QA (start_logits + end_logits) | Medium |
| Dynamic output column names | Tasks with multiple outputs | Low |

Currently the scorer outputs a single tensor. QA models often have two output tensors. This could be handled by allowing multiple output column names in `OnnxTextModelScorerOptions`.

## Implementation Priority

Based on value and complexity:

1. **Text Classification** — Highest value, lowest complexity. Only needs softmax + argmax, which are trivial with `TensorPrimitives.SoftMax()`.
2. **Cross-Encoder Similarity / Reranking** — High value for RAG scenarios. Needs text pair tokenization.
3. **NER** — Medium value, higher complexity. Needs BIO decoding + offset tracking.
4. **QA** — Lower priority. Needs multi-output scorer + token decoding.

Each task requires only a new post-processing transform + a convenience facade. The tokenizer and scorer foundation is shared.
