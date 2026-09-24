# Typed decisions with ML.NET

Typed decisions answer a fixed set of questions about supplied text. They do
not generate prose, and `Fit` does not train or fine-tune the ONNX model. The
model scores the alternatives supplied by the caller:

- **Choice** selects one caller-provided label.
- **Score** returns the expected zero-based option index, not a classification
  confidence. For this three-level question, the range is `0..2`, not a
  percentage; `1.1856464` means the probability-weighted index is between
  options `1` and `2`.
- **Noul (Boolean)** returns `true` when `pTrue >= pFalse` and exposes `pTrue`.

The primary and only file-based sample is
[MLNetPipeline](MLNetPipeline/README.md). It uses the existing
`MLNet.TextInference.Onnx` package through the direct fitted transformer path,
the schema-aware facade, the native preparation / scoring / decoding stages,
and an append-composable second facade.

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

## Results at a glance

These are the exact decoded values for the two fixed sample states used by
the captured runs:

| Sample input | Priority | Quality score | Actionable | P(true) |
|---|---|---:|---|---:|
| Reproducible steps, urgent fix | high | 1.1856464 | true | 0.8520638 |
| Missing logs, no clear action | low | 0.61998236 | false | 0.10274245 |

`Reproducible steps, urgent fix` abbreviates the full state
`The customer supplied reproducible steps and requested an urgent fix.`.
`Missing logs, no clear action` abbreviates
`The report is missing logs and has no clear requested action.`. The values
come from the pinned English FP32 Laya capture at revision
`68f27dfe5a27a54fb2b1fefc432f43f972e90868` with managed/native ONNX Runtime
1.24.2. `quality` is a three-level zero-based Score with range `0..2`, not a
percentage, so `1.1856464` is an expected ordinal index. Changing inputs,
assets, or settings can change the predictions.

## Start here: beginner explanation

Think of this as a model-powered checklist: the application supplies a state,
asks fixed questions, and supplies the alternatives the model is allowed to
choose or score. The model returns signals for those alternatives; it does
not write an answer, enforce a business policy, or perform an action.

The [beginner and developer walkthrough](MLNetPipeline/README.md#beginner-and-developer-walkthrough)
then follows one row through assets, BPE tokenization, five tensors, ONNX
scoring, decoding, and ML.NET output mapping.

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
   computes stable temperature-adjusted option probabilities, and does not
   softmax `act_probs` again. Confidence is one minus normalized entropy.
   `action_probability` is the configured action column from `act_probs`, not
   an option probability. A zero action probability is not an explanation of
   the model's decision and does not gate the decoded result.

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

# Native single-row PredictionEngine mapping
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode prediction-engine --model-assets .\models\laya-english-fp32

# PredictionEngine over the explicit native stages
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode prediction-engine-stages --model-assets .\models\laya-english-fp32

# PredictionEngine over an append-composed facade
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode prediction-engine-composed --model-assets .\models\laya-english-fp32

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
preparation, scoring, and decoding kernels. The `prediction-engine` mode uses
ML.NET's native single-row `PredictionEngine` mapper over the facade; it reads
the typed columns directly for each prediction, rather than creating a new
`MLContext`, `IDataView`, or cursor per getter. `PredictionEngine` instances
are not thread-safe: use one instance per caller or pool instances when
sharing a fitted transformer.

## Captured output

The following is a compact preview from actual CPU runs with the fixed sample
inputs, pinned assets, and model settings above, not from `--help`, a
synthetic fixture, or a business-rule test. The model is the English FP32 Laya
export at revision
`68f27dfe5a27a54fb2b1fefc432f43f972e90868`, using managed/native ONNX Runtime
1.24.2. The independent comparison was within `1e-6` for this CPU snapshot;
changing the pinned inputs, assets, or settings can change predictions, not
merely the last displayed digit. These are model outputs, not guaranteed
business truth, and no empirical probability calibration claim is intended.

### Mode/output guide

| Mode | Output summary |
|---|---|
| `direct` | JSON-only decoded responses. |
| `facade` / `prediction-engine` | Typed values plus diagnostic JSON. |
| `stages` / `prediction-engine-stages` | Native dimensions plus typed values and JSON. |
| `composed` / `prediction-engine-composed` | Original values plus appended `AppendedDecision_*` values. |

### Facade and `PredictionEngine` typed output

`facade` and `prediction-engine` printed these exact typed lines for the two
state rows. The vectors preserve the configured option order.

```text
priority=high; quality_score=1.1856464; actionable=True; actionable_true_probability=0.8520638; priority_confidence=0.4831077; priority_action_probability=0; quality_confidence=0.21570939; quality_action_probability=0; actionable_confidence=0.3953485; actionable_action_probability=0; priority_probabilities=0.11570658,0.88429344; quality_probabilities=0.09035327,0.633647,0.2759997
priority=low; quality_score=0.61998236; actionable=False; actionable_true_probability=0.10274245; priority_confidence=0.596907; priority_action_probability=0; quality_confidence=0.22399896; quality_action_probability=0; actionable_confidence=0.5223708; actionable_action_probability=0; priority_probabilities=0.9197405,0.08025956; quality_probabilities=0.42993295,0.5201518,0.049915284
```

### Reading the typed preview

- **Choice:** `priority` uses the labels in order: `low` is index `0` and
  `high` is index `1`. The decoder returns the argmax label, so Row 1 is
  `high` because `0.88429344` is larger than `0.11570658`; Row 2 is `low`.
- **Score:** `quality` is the expected zero-based option index, not a
  confidence. Row 1 is
  `0*0.09035327 + 1*0.633647 + 2*0.2759997 = 1.1856464` (within displayed
  precision). Row 2 is
  `0*0.42993295 + 1*0.5201518 + 2*0.049915284 = 0.61998236`.
- **Noul:** `actionable` is true when `P(true) >= P(false)`. Its
  `probability_true` is the second probability in the `[false,true]` vector;
  it is not the same field as `action_probability`.
- **Confidence:** the decoder uses one minus normalized entropy,
  `1 - H(p)/ln(valid option count)`, over the valid option distribution. The
  count excludes padded marker slots; it is not padded `K=3`. Confidence
  describes concentration of the distribution, not empirical accuracy or a
  calibrated probability that the prediction is correct.
- **Action probability:** `action_probability` is a separate configured
  channel from the graph's already-softmaxed `act_probs` head. Both captured
  rows happen to show `0`; that value does not imply `actionable=false`, does
  not explain why the graph produced it, and does not gate any decision.

The full exact JSON responses, native shape lines, appended output, and
seven-mode walkthrough are in
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
`Decision_actionable_Probability`. Probability vectors are
temperature-adjusted option probabilities, not logits, and carry `SlotNames`
metadata with the option labels. This documentation makes no empirical
calibration claim. Each confidence and action-probability column belongs to
the same question. `DecisionResults` remains an optional full diagnostic JSON
column with every question and distribution.

The direct API, lazy `IDataView` paths, and native single-row
`PredictionEngine` mapping are supported. Native ML.NET `Save`/`Load` remains
unimplemented in this release. External assets are a packaging consideration,
not an inherent technical limitation of row mapping or persistence. The direct
methods are `transformer.Infer(state)` and `transformer.Infer(states)`; the
bulk form can batch states while remaining a straightforward call-oriented
API. ML.NET's value here is schema-aware composition and lazy `IDataView`
interoperability, not exclusive batching, an automatic speedup, or better
accuracy.
