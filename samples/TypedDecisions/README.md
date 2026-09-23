# Typed decisions with ML.NET

Typed decisions answer a fixed set of questions about supplied text. They do
not generate prose, and `Fit` does not train or fine-tune the ONNX model. The
model scores the alternatives supplied by the caller:

- **Choice** selects one caller-provided label.
- **Score** returns the expected zero-based option index, not a classification
  confidence. For three options, `1.1856464` means the probability-weighted
  index is between options `1` and `2`.
- **Noul (Boolean)** returns `true` when `pTrue >= pFalse` and exposes `pTrue`.

The primary and only file-based sample is
[MLNetPipeline](MLNetPipeline/README.md). It uses the same ML.NET package for
the direct convenience API, the schema-aware facade, the native preparation /
scoring / decoding stages, and an append-composable second facade.

## Inputs

The sample creates two text rows:

```text
The customer supplied reproducible steps and requested an urgent fix.
The report is missing logs and has no clear requested action.
```

Both rows use these static question factories:

```csharp
using MLNet.TextInference.TypedDecisions;
using static MLNet.TextInference.TypedDecisions.DecisionQuestion;

var questions = new[]
{
    Choice("priority", "How urgent is the request?", new[] { "low", "high" }),
    Score("quality", "How strong is the evidence?",
        new[] { "weak", "moderate", "strong" }),
    Noul("actionable", "Can the request be acted on now?")
};
```

State is always text. A caller may serialize JSON into that text column, but
the library does not provide a Python-compatible object serializer. The
question instructions, option labels, state text, and profile-specific
preprocessing are model inputs; they are not business rules or guarantees.

## What happens

1. The configured Microsoft.ML.Tokenizers BPE tokenizer applies the selected
   profile's byte-level and normalization behavior. Special-token IDs are
   separate profile metadata.
2. Each question's instructions, options, and state are combined, truncated,
   marked, and padded. The marker positions identify option tokens.
3. A flattened ONNX batch uses five native tensors:

   | Input | Shape | Element type |
   |---|---|---|
   | `input_ids` | `[B,L]` | `Int64` |
   | `attention_mask` | `[B,L]` | `Int64` |
   | `marker_pos` | `[B,K]` | `Int64` |
   | `marker_mask` | `[B,K]` | `Bool` |
   | `qtype` | `[B]` | `Int64` |

   `B` is the flattened request-question count, not necessarily the
   `IDataView` row count. `L` is the padded sequence length and `K` is the
   maximum option width in the batch.
4. The graph returns `logits [B,K]` and already-softmaxed `act_probs [B,2]`.
   C# decoding applies the bundle temperature policy, masks unused options,
   computes stable option probabilities, and does not softmax `act_probs`
   again. Confidence is entropy-based. `action_probability` is the configured
   action column from `act_probs`, not an option probability.

For a decoded response, `input_tokens` is the aggregate count of nonpadding
tokens across that request's question-specific prepared sequences. It
includes instructions, options, state, and special tokens.

## Model assets

Inference never downloads model files. The normal setup is a local model
directory containing the graph, its external-data sidecar, the Laya
configuration, and the tokenizer directory:

```text
models/laya-english-fp32/
  laya.onnx
  laya.onnx.data
  laya_config.json
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
```

An optional versioned ZIP archive with `typed-decision-bundle.json` is also
accepted for deployment. The directory form does not require a generated
manifest. Paths are validated as relative bundle entries, external data stays
adjacent to the graph, and hashes are checked when a manifest supplies them.

The initial profile is the English FP32 export from public Hugging Face
repository [`receptron/laya-onnx`](https://huggingface.co/receptron/laya-onnx)
at revision
`68f27dfe5a27a54fb2b1fefc432f43f972e90868`. The heavyweight model is not
committed; ordinary tests use a small offline ONNX fixture.

## Run the sample

These are .NET 10 file-based commands with `PublishAot=false`. The path below
is a safe repository-relative placeholder; replace it with a prepared local
directory or optional archive:

```powershell
# Direct API on the fitted transformer
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode direct --model-assets .\models\laya-english-fp32

# Recommended lazy, cursor-batched facade
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode facade --model-assets .\models\laya-english-fp32

# Native inspectable stages
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode stages --model-assets .\models\laya-english-fp32

# Append a second facade to an existing ML.NET estimator
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode composed --model-assets .\models\laya-english-fp32
```

The facade batches source rows up to `BatchSize`, flattens their questions,
performs one model call, and caches pass-through columns. The stage chain
exposes native numeric/vector/Boolean columns for the same five tensors and
the two model outputs; the sample prints their dimensions and the decoded
fields, not an intermediate JSON transport. All paths use the same
preparation, scoring, and decoding kernels.

## Captured output

These representative values were captured from the pinned real-model
facade, stages, and composed runs. Providers and runtime versions can change
floating-point digits; labels, ordering, finite values, probability
normalization, and the typed relationships are the stable expectations.

| State row | Choice (`priority`) | Score (`quality`) | `P(true)` (`actionable`) | Confidence | Action probability |
|---|---|---:|---:|---:|---:|
| Reproducible steps, urgent fix | `high` | `1.1856463` | `0.85206354` | `0.48310703` | `0` |
| Missing logs, no requested action | `low` | `0.61998177` | `0.102742165` | `0.5969056` | `0` |

For Row 1's ordinal score, the arithmetic is
`0 * 0.0903531 + 1 * 0.63364744 + 2 * 0.27599943 ~= 1.1856463`.

The full readable JSON responses, native stage output details, and the actual
per-question column names are in
[MLNetPipeline/README.md](MLNetPipeline/README.md).

## Output columns and limitations

The facade and decoding stage add question-specific columns known from the
configured questions:

| Question type | Columns |
|---|---|
| Choice | `PredictedLabel` text, `Probabilities` vector, `Confidence`, `ActionProbability` |
| Score | `Score`, `Probabilities` vector, `Confidence`, `ActionProbability` |
| Noul | `PredictedLabel` Boolean, `Probability` (`P(true)`), `Confidence`, `ActionProbability` |

Columns are prefixed with the configured question ID, for example
`Decision_priority_PredictedLabel` and
`Decision_actionable_Probability`. Probability vectors are calibrated option
probabilities, not logits, and carry `SlotNames` metadata with the option
labels. Each confidence and action-probability column belongs to the same
question. `DecisionResults` remains an optional full diagnostic JSON column
with every question and distribution.

The direct API and lazy `IDataView` paths are supported. Native ML.NET
`Save`/`Load` and single-row `PredictionEngine` mapping are not advertised:
the transformers reference local external assets and report that row-mapper
support is unavailable. The direct method is
`transformer.Infer(state)` or `transformer.Infer(states)`.
