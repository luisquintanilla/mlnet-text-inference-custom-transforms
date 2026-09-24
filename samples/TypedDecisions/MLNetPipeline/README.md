# ML.NET typed decisions

This file-based .NET 10 sample is the primary entry point for typed-decision
inference. It uses the existing `MLNet.TextInference.Onnx` package. `Fit`
validates the `IDataView` schema and opens local model assets, but does not
train the ONNX model.

## Inputs

The two `State` rows are:

```text
The customer supplied reproducible steps and requested an urgent fix.
The report is missing logs and has no clear requested action.
```

The configured questions are:

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

State is text. If the caller needs JSON state, it serializes that JSON before
putting it in the `State` column. The model scores the supplied alternatives;
it does not generate prose or apply business rules.

## Local model-directory setup

Inference has no implicit network access. Point `ModelAssetsPath` (or the
sample's `--model-assets` option) at a directory such as:

```text
models/laya-english-fp32/
  laya.onnx
  laya.onnx.data
  laya_config.json
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
```

An optional versioned ZIP archive with a manifest is also accepted. A
directory does not require a generated manifest. The graph's external data
must remain adjacent to the graph. The sample targets the public
`receptron/laya-onnx` English FP32 revision
`68f27dfe5a27a54fb2b1fefc432f43f972e90868`; assets are deliberately not
committed.

## Seven modes

All commands use portable JIT execution (`PublishAot=false`):

```powershell
# Direct convenience API on the fitted transformer
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode direct --model-assets .\models\laya-english-fp32

# Recommended: lazy IDataView facade with cursor batching
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

# Preparation -> scoring -> decoding stages
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode stages --model-assets .\models\laya-english-fp32

# Two append-composed facade applications
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode composed --model-assets .\models\laya-english-fp32
```

The program accepts `--bundle` as a compatibility alias for
`--model-assets`, but the value can be an ordinary model directory and need
not be a bundle archive.

The direct API uses the same resources and kernels as the facade:

```csharp
using MLNet.TextInference.Onnx;
using MLNet.TextInference.TypedDecisions;

using var transformer = ml.Transforms.OnnxTypedDecisions(options).Fit(data);
DecisionResponse one = transformer.Infer(state);
IReadOnlyList<DecisionResponse> many = transformer.Infer(states);
```

## Processing and native stage schema

The pipeline is:

1. Microsoft.ML.Tokenizers BPE tokenizes each question-specific sequence,
   while profile metadata supplies special-token IDs and byte-level behavior.
2. Instructions, options, and state are combined, truncated, marker positions
   are recorded, and sequences are padded.
3. The preparation stage emits `Int64` vectors for `input_ids`,
   `attention_mask`, and `marker_pos`, a `Bool` vector for `marker_mask`, an
   `Int64` vector for `qtype`, and `Int32` scalar dimensions:
   `DecisionBatchSize`, `DecisionSequenceLength`, and `DecisionMarkerWidth`.
4. The task-specific scorer batches prepared source rows within a cursor and
   emits `Single` vectors for `logits` and already-softmaxed `act_probs`. It
   does not promise a particular number of physical ORT invocations.
5. The decoder applies the profile temperature policy (valid values are
   clamped to `[0.5, 5.0]` with diagnostics), masks unused options, computes
   stable option probabilities, expected ordinal scores, Boolean decisions,
   entropy confidence, and per-question action probability.

For two source rows and three questions, each prepared source row has
`B=3`: one flattened row for each configured question. The printed
`native_batch=3` is this per-source-row prepared question count; it is not the
`IDataView` row count and is not a promise of the physical ORT invocation
count. The captured per-row flattened lengths are `3*45=135` and `3*46=138`.
`input_tokens` is the aggregate nonpadding-token count over all
question-specific sequences for one request, including instructions, options,
state, and special tokens, so the captured values are `112` and `115`, not
the padded lengths.

The stage transport is native ML.NET numeric/vector/Boolean data, not a JSON
envelope. In `stages` mode the sample prints the prepared/scored vector
lengths and decoded result fields; it does not print every token or vector
element. The `DecisionResults` text column is the optional full diagnostic
JSON output, not stage transport.

## PredictionEngine mode

`prediction-engine` uses
`MLContext.Model.CreatePredictionEngine<StateRow, DecisionRow>` against the
fitted facade. The mapper computes the requested typed columns from the
current input row and reuses the fitted tokenizer, decoder, and ONNX session.
The sample prints the actual probability vectors, typed scalar values, and
`DecisionResults` for both states.

`prediction-engine-stages` uses `StageRow` and exposes the native prepared and
scored vectors through the same single-row mapper. `prediction-engine-composed`
uses `ComposedDecisionRow` and reads both the original and appended typed
column sets from one mapped row.

`PredictionEngine` is a single-row convenience API and is not thread-safe.
Create one instance per caller, or use a pool of instances when sharing a
fitted transformer. Do not share one instance concurrently.

## Output mapping

Each configured question receives an unambiguous prefix:

| Question | Columns in this sample |
|---|---|
| `priority` Choice | `Decision_priority_PredictedLabel`, `Decision_priority_Probabilities`, `Decision_priority_Confidence`, `Decision_priority_ActionProbability` |
| `quality` Score | `Decision_quality_Score`, `Decision_quality_Probabilities`, `Decision_quality_Confidence`, `Decision_quality_ActionProbability` |
| `actionable` Noul | `Decision_actionable_PredictedLabel`, `Decision_actionable_Probability`, `Decision_actionable_Confidence`, `Decision_actionable_ActionProbability` |

Choice labels are text. Score is the expected zero-based option index, not a
confidence. Noul selects true when `P(true) >= P(false)`. Probability vectors
are temperature-adjusted option probabilities (not logits) and carry
`SlotNames` metadata with option labels. This documentation makes no
empirical calibration claim. Every confidence and action-probability scalar is
associated with its own question. `DecisionResults` retains all questions,
distributions, legends, and derived values.

The composed mode uses `OutputPrefix = "AppendedDecision_"` and
`ResultsColumnName = "AppendedDecisionResults"`, so it adds the corresponding
`AppendedDecision_*` columns without overwriting the first facade's columns.
Both applications use the same request and assets, so the values should match
within floating-point tolerance. Accessing multiple output getters does not
repeat inference for the same cursor row.

Native ML.NET model `Save`/`Load` remains unimplemented in this release.
External local assets are a packaging consideration, not an inherent
technical limitation of row mapping or persistence.

## Captured outputs

The following is captured expected output from actual CPU runs of all seven
modes with the fixed sample inputs, pinned assets, and settings described
above. It is not output from `--help`, a synthetic fixture, or a business-rule
test. The model is the English FP32 Laya export at revision
`68f27dfe5a27a54fb2b1fefc432f43f972e90868`, using managed/native ONNX Runtime
1.24.2. The displayed values are a reproducible CPU snapshot: the
independent comparison was within `1e-6`, but these expectations depend on
the pinned inputs, assets, runtime, provider, and settings. They are not
guaranteed business truth, and no empirical probability calibration claim is
intended.

### Mode/output guide

| Mode | Captured output |
|---|---|
| `direct` | JSON response objects only: aggregate `input_tokens` and all decoded questions. |
| `facade` | Typed scalar/vector fields plus the same diagnostic `DecisionResults` JSON. |
| `prediction-engine` | The facade fields and JSON, read one source row at a time through `PredictionEngine`. |
| `stages` | Native tensor dimensions, typed decoded fields, and diagnostic JSON. |
| `prediction-engine-stages` | The native dimensions and typed fields through a single-row `PredictionEngine`. |
| `composed` | The first facade's values plus `AppendedDecision_*` values and `AppendedDecisionResults`. |
| `prediction-engine-composed` | The same original and appended values through a single-row `PredictionEngine`. |

The excerpts below share identical output between equivalent modes; they do
not represent seven different model predictions.

### Direct JSON-only output

These two JSON objects are the exact responses printed by `direct`. The
`DecisionResults` JSON in `facade`, `prediction-engine`, `stages`,
`prediction-engine-stages`, and the first application in each composed mode
has the same values.

```json
{
  "input_tokens": 112,
  "results": [
    {
      "id": "priority",
      "type": "choice",
      "confidence": 0.4831077,
      "action_probability": 0,
      "labels": [
        "low",
        "high"
      ],
      "probabilities": [
        0.11570658,
        0.88429344
      ],
      "choice": "high"
    },
    {
      "id": "quality",
      "type": "score",
      "confidence": 0.21570939,
      "action_probability": 0,
      "labels": [
        "0",
        "1",
        "2"
      ],
      "probabilities": [
        0.09035327,
        0.633647,
        0.2759997
      ],
      "score": 1.1856464,
      "legend": {
        "0": "weak",
        "1": "moderate",
        "2": "strong"
      }
    },
    {
      "id": "actionable",
      "type": "noul",
      "confidence": 0.3953485,
      "action_probability": 0,
      "labels": [
        "false",
        "true"
      ],
      "probabilities": [
        0.14793625,
        0.8520638
      ],
      "noul": true,
      "probability_true": 0.8520638
    }
  ]
}
```

```json
{
  "input_tokens": 115,
  "results": [
    {
      "id": "priority",
      "type": "choice",
      "confidence": 0.596907,
      "action_probability": 0,
      "labels": [
        "low",
        "high"
      ],
      "probabilities": [
        0.9197405,
        0.08025956
      ],
      "choice": "low"
    },
    {
      "id": "quality",
      "type": "score",
      "confidence": 0.22399896,
      "action_probability": 0,
      "labels": [
        "0",
        "1",
        "2"
      ],
      "probabilities": [
        0.42993295,
        0.5201518,
        0.049915284
      ],
      "score": 0.61998236,
      "legend": {
        "0": "weak",
        "1": "moderate",
        "2": "strong"
      }
    },
    {
      "id": "actionable",
      "type": "noul",
      "confidence": 0.5223708,
      "action_probability": 0,
      "labels": [
        "false",
        "true"
      ],
      "probabilities": [
        0.89725757,
        0.10274245
      ],
      "noul": false,
      "probability_true": 0.10274245
    }
  ]
}
```

### Facade and `PredictionEngine` exact typed lines

`facade` and `prediction-engine` printed these exact typed lines; the
corresponding diagnostic JSON is the two objects above.

```text
priority=high; quality_score=1.1856464; actionable=True; actionable_true_probability=0.8520638; priority_confidence=0.4831077; priority_action_probability=0; quality_confidence=0.21570939; quality_action_probability=0; actionable_confidence=0.3953485; actionable_action_probability=0; priority_probabilities=0.11570658,0.88429344; quality_probabilities=0.09035327,0.633647,0.2759997
priority=low; quality_score=0.61998236; actionable=False; actionable_true_probability=0.10274245; priority_confidence=0.596907; priority_action_probability=0; quality_confidence=0.22399896; quality_action_probability=0; actionable_confidence=0.5223708; actionable_action_probability=0; priority_probabilities=0.9197405,0.08025956; quality_probabilities=0.42993295,0.5201518,0.049915284
```

### Native stage shape lines

`stages` and `prediction-engine-stages` printed these exact shape lines,
followed by the same typed lines and JSON:

```text
native_batch=3; sequence_length=45; marker_width=3; input_ids=135; logits=9; action_probabilities=6
native_batch=3; sequence_length=46; marker_width=3; input_ids=138; logits=9; action_probabilities=6
```

`native_batch=3` is the three-question prepared batch for each source row,
not the number of `IDataView` rows and not a promise of physical ORT
invocation count. `B=3`, `L=45` or `46`, and `K=3`, so the flattened lengths
are `B*L` (`3*45=135`, `3*46=138`), `B*K=9` for `logits`, and `B*2=6`
for the action head. `K=3` is the maximum marker/option width. `priority`
and `actionable` each have two valid slots and one padded slot; `marker_mask`
identifies the valid marker positions. The aggregate nonpadding counts are
`input_tokens=112` and `input_tokens=115`, not the padded `B*L` lengths.

### Composed appended lines

`composed` and `prediction-engine-composed` also printed these exact appended
lines. Their JSON appears under `AppendedDecisionResults` and has the same
two decoded response objects.

```text
appended_priority=high; appended_priority_confidence=0.4831077; appended_priority_action_probability=0; appended_priority_probabilities=0.11570658,0.88429344; appended_quality_score=1.1856464; appended_quality_confidence=0.21570939; appended_quality_action_probability=0; appended_quality_probabilities=0.09035327,0.633647,0.2759997; appended_actionable=True; appended_actionable_true_probability=0.8520638; appended_actionable_confidence=0.3953485; appended_actionable_action_probability=0
appended_priority=low; appended_priority_confidence=0.596907; appended_priority_action_probability=0; appended_priority_probabilities=0.9197405,0.08025956; appended_quality_score=0.61998236; appended_quality_confidence=0.22399896; appended_quality_action_probability=0; appended_quality_probabilities=0.42993295,0.5201518,0.049915284; appended_actionable=False; appended_actionable_true_probability=0.10274245; appended_actionable_confidence=0.5223708; appended_actionable_action_probability=0
```

### Reading the values

- **Choice/order/argmax:** `priority` labels are ordered `[low, high]`.
  Row 1 is `high` because `0.88429344` is the larger probability; Row 2 is
  `low` because `0.9197405` is the larger probability.
- **Score:** `quality` is the expected zero-based option index, not a
  confidence. Row 1 is
  `0*0.09035327 + 1*0.633647 + 2*0.2759997 = 1.1856464` within displayed
  precision. Row 2 is
  `0*0.42993295 + 1*0.5201518 + 2*0.049915284 = 0.61998236`.
- **Noul:** `actionable` is true when `P(true) >= P(false)`.
  `probability_true` is the second value in the `[false, true]` vector.
- **Confidence:** confidence is one minus normalized entropy,
  `1 - H(p)/ln(valid option count)`. The count is the number of valid
  options for that question, not padded `K=3`; it measures distribution
  concentration, not predicted probability, correctness, or empirical
  calibration. Here `H(p) = -sum(p * ln(p))`; confidence approaches `0` for a
  uniform distribution and `1` for a concentrated distribution. It is not the
  winning probability: Row 1 `priority` has winning probability `0.88429344`
  and confidence `0.4831077`.
- **Action probability:** `action_probability` is the separate configured
  graph-head channel from already-softmaxed `act_probs`. Both captured rows
  show `0`; that does not mean `actionable=false`, does not explain why the
  graph produced zero, and does not gate the decoded decision.
