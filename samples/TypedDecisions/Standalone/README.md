# Standalone typed decisions

This .NET 10 file-based app is the small direct-core example. It uses the
same bundle and the same preparation, ONNX, and decoding implementation as the
ML.NET facade, but does not create an `MLContext` or an `IDataView`.

## Input

The sample passes this state as text:

```text
The customer supplied reproducible steps and requested an urgent fix.
```

It asks three questions using the core's static factories:

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

State is a string by design. If an application starts with an object or a
structured document, it must serialize that value before calling the API.
There is no implicit Python-compatible serializer.

## Run

Use a local directory or ZIP prepared as described in the
[shared typed-decision documentation](../README.md):

```powershell
dotnet run --file samples/TypedDecisions/Standalone/Program.cs -- `
  --mode facade --bundle .\models\laya-english-fp32.bundle
dotnet run --file samples/TypedDecisions/Standalone/Program.cs -- `
  --mode stages --bundle .\models\laya-english-fp32.bundle
dotnet run --file samples/TypedDecisions/Standalone/Program.cs -- --help
```

The file app opts out of .NET 10's native-AOT default with
`#:property PublishAot=false`; this sample is ordinary portable JIT execution.
Inference never downloads model assets.

`facade` calls `OnnxTypedDecisions.Infer`, which composes all three core
stages. `stages` calls `PrepareDecisionInputs`, `ScoreOnnxDecisionModel`, and
`DecodeDecisions` separately and prints the prepared/scored JSON envelopes and
decoded response. The five tensor inputs and output meanings are documented in
the [shared README](../README.md).

## Captured output

The following is the captured first-request facade response from the pinned
English FP32 Laya bundle. It is an expected shape and representative numeric
example, not a bit-for-bit promise across runtimes:

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

`input_tokens` is the aggregate count of nonpadding tokens across the three
question-specific prepared sequences for this request. It includes
instructions, options, state, and special tokens, not only the state text.
For the `quality` result, the score is the probability-weighted ordinal index:
`0 * 0.09035328 + 1 * 0.6336471 + 2 * 0.27599967 ~= 1.1856464`.

Notice that the noul `confidence` (`0.39534837`) is not its
`probability_true` (`0.8520637`). Confidence is entropy-based; the Boolean
probability is the probability of the `true` option, and the decoder chooses
the Boolean value using `pTrue >= pFalse`. `action_probability` is a separate
model output selected from `act_probs`.
