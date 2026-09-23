# ML.NET typed decisions

This is the primary sample for the feature. It exposes the shared
ML.NET-independent Laya core through schema-aware, lazy ML.NET transforms.
`Fit` validates the input schema and constructs a transformer; it does not
train or fine-tune the ONNX model.

## Input rows and questions

The sample creates an in-memory `IDataView` with two scalar `State` rows:

```text
The customer supplied reproducible steps and requested an urgent fix.
The report is missing logs and has no clear requested action.
```

Both rows use the same three questions:

```csharp
using static MLNet.TextInference.TypedDecisions.DecisionQuestion;

var questions = new[]
{
    Choice("priority", "How urgent is the request?", new[] { "low", "high" }),
    Score("quality", "How strong is the evidence?",
        new[] { "weak", "moderate", "strong" }),
    Noul("actionable", "Can the request be acted on now?")
};
```

The state column is text. Applications may put serialized JSON in that text
column, but serialization is the caller's responsibility. The model scores
the supplied alternatives; it does not generate a response or enforce a
business rule.

## Facade, stages, and composition

```powershell
# Recommended: one lazy, cursor-batched facade
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode facade --bundle .\models\laya-english-fp32.bundle

# Inspect preparation, scoring, and decoding contracts
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode stages --bundle .\models\laya-english-fp32.bundle

# Append a second facade and inspect the appended columns
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode composed --bundle .\models\laya-english-fp32.bundle

dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- --help
```

The file app sets `PublishAot=false`, so these commands use portable JIT
execution rather than .NET 10's native-AOT default. The app never downloads
the model.

The facade is `ml.Transforms.OnnxTypedDecisions(options)`. Its lazy cursor
collects source rows up to `BatchSize`, flattens each request's questions into
the core batch, performs one ONNX call, and caches pass-through columns so
downstream enumeration does not re-enumerate the source.

The inspectable chain is:

```csharp
ml.Transforms.PrepareDecisionInputs(prepared)
    .Append(ml.Transforms.ScoreOnnxDecisionModel(scored))
    .Append(ml.Transforms.DecodeDecisions(decoded));
```

Those stages deliberately transport prepared inputs and scored outputs as
scalar `Text` JSON columns. They are row-oriented and score one source row at
a time; they are not native tensor columns and do not provide the facade's
cursor batching. Use the facade for normal ML.NET use and the stages when
inspecting or composing the intermediate contracts.

`composed` applies the facade twice:

```csharp
IEstimator<ITransformer> pipeline = ml.Transforms.OnnxTypedDecisions(options);
output = pipeline.AppendOnnxTypedDecisions(ml, appendedOptions)
    .Fit(data)
    .Transform(data);
```

The sample materializes both the original columns and the appended columns:
`AppendedDecisionResults`, `AppendedDecisionChoice`,
`AppendedDecisionScore`, `AppendedDecisionProbabilityTrue`,
`AppendedDecisionConfidence`, and `AppendedDecisionActionProbability`.
They are expected to match because both transforms use the same request and
bundle, not because the two applications share a cached model result.

The scalar projection is intentionally compact for ML.NET schemas:

- choice, score, and `probability_true` select the first result of the
  corresponding question type;
- confidence and action probability select the first result overall;
- `DecisionResults` retains every question, all distributions, legends, and
  derived fields.

With multiple questions of one type, inspect `DecisionResults` when the
first-match scalar is not sufficient. Typed-decision transformers reference
the explicit bundle path and do not currently implement ML.NET model
Save/Load or embed the bundle.

## Captured expected output

These excerpts were captured by running the facade, stages, and composed modes
against the pinned English FP32 Laya bundle. They document the output shape
and representative values; floating-point digits can vary with runtime,
provider, or calibration changes. Stable labels/order and relationships are
the compatibility expectations.

Facade and stages produce the same two rows:

| State row | `DecisionChoice` | `DecisionScore` | `DecisionProbabilityTrue` | `DecisionConfidence` | `DecisionActionProbability` |
|---|---|---:|---:|---:|---:|
| Reproducible steps, urgent fix | `high` | `1.1856464` | `0.8520637` | `0.4831077` | `0` |
| Missing logs, no requested action | `low` | `0.6199823` | `0.10274245` | `0.5969069` | `0` |

The full result JSON for the two rows is:

```json
{"input_tokens":112,"results":[{"id":"priority","type":"choice","confidence":0.4831077,"action_probability":0,"labels":["low","high"],"probabilities":[0.115706585,0.88429344],"choice":"high"},{"id":"quality","type":"score","confidence":0.21570939,"action_probability":0,"labels":["0","1","2"],"probabilities":[0.09035328,0.6336471,0.27599967],"score":1.1856464,"legend":{"0":"weak","1":"moderate","2":"strong"}},{"id":"actionable","type":"noul","confidence":0.39534837,"action_probability":0,"labels":["false","true"],"probabilities":[0.14793624,0.8520637],"noul":true,"probability_true":0.8520637}]}
{"input_tokens":115,"results":[{"id":"priority","type":"choice","confidence":0.5969069,"action_probability":0,"labels":["low","high"],"probabilities":[0.91974044,0.08025956],"choice":"low"},{"id":"quality","type":"score","confidence":0.22399896,"action_probability":0,"labels":["0","1","2"],"probabilities":[0.42993295,0.52015173,0.04991528],"score":0.6199823,"legend":{"0":"weak","1":"moderate","2":"strong"}},{"id":"actionable","type":"noul","confidence":0.5223708,"action_probability":0,"labels":["false","true"],"probabilities":[0.89725757,0.10274245],"noul":false,"probability_true":0.10274245}]}
```

The composed run printed the same values from the appended columns:

```text
appended_choice=high; appended_score=1.1856464; appended_true_probability=0.8520637; appended_confidence=0.4831077; appended_action_probability=0
appended_choice=low; appended_score=0.6199823; appended_true_probability=0.10274245; appended_confidence=0.5969069; appended_action_probability=0
```

The appended JSON column retained all three results for each row. The
acceptance comparison checked both rows, original versus appended labels,
Booleans, distributions, scores, probabilities, confidence, and action
probabilities within `1e-6`.

For the tensor contract, bundle requirements, decoder policy, and direct-core
sample, see the [shared typed-decision README](../README.md).
