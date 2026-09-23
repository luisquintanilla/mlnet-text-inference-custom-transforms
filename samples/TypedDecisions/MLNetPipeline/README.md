# ML.NET typed decisions

This file-based .NET 10 sample is the primary entry point for typed-decision
inference. It uses `MLNet.TextInference.Onnx`; there is no separate standalone
core package. `Fit` validates the `IDataView` schema and opens local model
assets, but does not train the ONNX model.

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

## Four modes

All commands use portable JIT execution (`PublishAot=false`):

```powershell
# Direct convenience API on the fitted transformer
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode direct --model-assets .\models\laya-english-fp32

# Recommended: lazy IDataView facade with cursor batching
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode facade --model-assets .\models\laya-english-fp32

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
4. The task-specific scorer batches prepared rows across cursor boundaries and
   emits `Single` vectors for `logits` and already-softmaxed `act_probs`.
5. The decoder applies the profile temperature policy (valid values are
   clamped to `[0.5, 5.0]` with diagnostics), masks unused options, computes
   stable option probabilities, expected ordinal scores, Boolean decisions,
   entropy confidence, and per-question action probability.

For two source rows and three questions, each prepared row has `B=3`; the
direct/facade batch contains six flattened question rows when both source
rows are processed together. `L` and `K` are dynamic per prepared batch.
`input_tokens` is the aggregate nonpadding-token count over all
question-specific sequences for one request, including instructions, options,
state, and special tokens.

The stage transport is native ML.NET numeric/vector/Boolean data, not a JSON
envelope. In `stages` mode the sample prints the prepared/scored vector
lengths and decoded result fields; it does not print every token or vector
element. The `DecisionResults` text column is the optional full diagnostic
JSON output, not stage transport.

## Output mapping

Each configured question receives an unambiguous prefix:

| Question | Columns in this sample |
|---|---|
| `priority` Choice | `Decision_priority_PredictedLabel`, `Decision_priority_Probabilities`, `Decision_priority_Confidence`, `Decision_priority_ActionProbability` |
| `quality` Score | `Decision_quality_Score`, `Decision_quality_Probabilities`, `Decision_quality_Confidence`, `Decision_quality_ActionProbability` |
| `actionable` Noul | `Decision_actionable_PredictedLabel`, `Decision_actionable_Probability`, `Decision_actionable_Confidence`, `Decision_actionable_ActionProbability` |

Choice labels are text. Score is the expected zero-based option index, not a
confidence. Noul selects true when `P(true) >= P(false)`. Probability vectors
are calibrated option probabilities (not logits) and carry `SlotNames`
metadata with option labels. Every confidence and action-probability scalar
is associated with its own question. `DecisionResults` retains all questions,
distributions, legends, and derived values.

The composed mode uses `OutputPrefix = "AppendedDecision_"` and
`ResultsColumnName = "AppendedDecisionResults"`, so it adds the corresponding
`AppendedDecision_*` columns without overwriting the first facade's columns.
Both applications use the same request and assets, so the values should match
within floating-point tolerance. Accessing multiple output getters does not
repeat inference for the same cursor row.

Native ML.NET model `Save`/`Load` and single-row `PredictionEngine` mapping are
not implemented in this release. The transformers report those capabilities
as unavailable; external local assets are a packaging consideration, not an
inherent technical limitation. Use the direct `Infer` method or lazy
`IDataView` materialization today.

## Captured outputs

The following values were captured from the pinned real-model direct/facade,
stages, and composed runs. They are formatted for readability. Provider,
runtime, and calibration differences can change float digits; labels,
ordering, finite values, and typed relationships are the stable expectations.

| Source row | Choice | Score | `P(true)` | Priority confidence | Action probability |
|---|---:|---:|---:|---:|---:|
| Row 1, reproducible steps | `high` | `1.1856464` | `0.8520637` | `0.4831077` | `0` |
| Row 2, missing logs | `low` | `0.6199823` | `0.10274245` | `0.5969069` | `0` |

### Row 1 full `DecisionResults`

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
        0.115706585,
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
        0.09035328,
        0.6336471,
        0.27599967
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
      "confidence": 0.39534837,
      "action_probability": 0,
      "labels": [
        "false",
        "true"
      ],
      "probabilities": [
        0.14793624,
        0.8520637
      ],
      "noul": true,
      "probability_true": 0.8520637
    }
  ]
}
```

### Row 2 full `DecisionResults`

```json
{
  "input_tokens": 115,
  "results": [
    {
      "id": "priority",
      "type": "choice",
      "confidence": 0.5969069,
      "action_probability": 0,
      "labels": [
        "low",
        "high"
      ],
      "probabilities": [
        0.91974044,
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
        0.52015173,
        0.04991528
      ],
      "score": 0.6199823,
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

Row 1's ordinal score is visible as
`0 * 0.09035328 + 1 * 0.6336471 + 2 * 0.27599967 ~= 1.1856464`.
The composed run printed the same two rows under the `AppendedDecision_*`
columns and the `AppendedDecisionResults` JSON column.
