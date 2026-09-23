# Typed decisions samples

Typed decisions answer a fixed set of questions about supplied text. They do
not generate prose and they do not train the ONNX model when `Fit` is called.
The model scores the alternatives supplied by the caller:

- **Choice** selects one label, such as `low` or `high`.
- **Score** returns the expected zero-based option index. It is not a
  classification confidence; for three levels, a score of `1.1856464` is the
  probability-weighted index between `0`, `1`, and `2`.
- **Noul** ("no/yes") returns a Boolean value and the probability of the
  `true` option.

The ML.NET sample is the primary user-facing path. The standalone sample is a
small direct-core example that is useful when ML.NET is not needed and for
inspecting each intermediate contract.

## Inputs used by both samples

The examples configure these questions with the static factory methods exposed
by the core:

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

State is always a string. It can be ordinary text or JSON that the caller
serialized explicitly; the library does not include a Python-compatible object
serializer. The ML.NET sample evaluates these two rows:

```text
The customer supplied reproducible steps and requested an urgent fix.
The report is missing logs and has no clear requested action.
```

The standalone sample evaluates the first state. The state, question
instructions, option labels, and profile-specific preprocessing are model
inputs. They are not business rules and the predictions do not guarantee that
an action is correct.

## What happens inside

1. The pinned profile loads the Hugging Face tokenizer assets through
   `Microsoft.ML.Tokenizers`. The configured BPE engine is kept as the
   tokenizer boundary; Laya special-token IDs and the mask token are separate
   profile metadata.
2. Each question is rendered with its options, combined with the instructions
   and state, then bounded to the profile limits. Options receive marker
   positions. Sequences are padded with the profile pad ID and attention masks
   distinguish real tokens from padding.
3. One ONNX call uses five dense inputs:

   | Input | Shape | Element type |
   |---|---|---|
   | `input_ids` | `[B,L]` | `Int64` |
   | `attention_mask` | `[B,L]` | `Int64` |
   | `marker_pos` | `[B,K]` | `Int64` |
   | `marker_mask` | `[B,K]` | `Bool` |
   | `qtype` | `[B]` | `Int64` |

   `B` is the flattened request-question count, not necessarily the
   `IDataView` row count. For the ML.NET example, two input rows and three
   questions produce six decision rows. `L` is the padded sequence length and
   `K` is the maximum option count in the batch.
4. The graph returns `logits [B,K]` and already-softmaxed `act_probs [B,2]`.
   The decoder applies the profile temperature policy to logits, masks unused
   options, and computes the typed result. It does not apply a second softmax
   to `act_probs`.
5. `confidence` is `1 - normalized entropy` of the option distribution. It is
   intentionally distinct from `probability_true` and `action_probability`.
   The profile's decoder metadata selects the `act_probs` index for
   `action_probability` (the default is index `1`). Temperature values are
   clamped to `[0.5, 5.0]` with diagnostics when a bundle requests values
   outside that policy.

## Bundle and public assets

Inference never downloads assets. Prepare a directory or ZIP containing:

```text
typed-decision-bundle.json
laya.onnx
laya.onnx.data                 # kept beside laya.onnx
laya_config.json
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
```

The initial profile is the English FP32 export from the public
[`receptron/laya-onnx`](https://huggingface.co/receptron/laya-onnx) repository
at revision
`68f27dfe5a27a54fb2b1fefc432f43f972e90868`. The model file, external-data
sidecar, Laya configuration, and tokenizer files must be acquired separately
with normal certificate-verified downloads. `TypedDecisionBundle.Open` validates
the manifest, relative paths, required sidecars, and hashes; ZIP extraction is
confined to a temporary directory.

After placing the files, the exact public bundle API can create the manifest
and its SHA-256 entries:

```csharp
TypedDecisionBundle.WriteManifest(
    @".\models\laya-english-fp32.bundle",
    new TypedDecisionBundleManifest
    {
        Profile = new TypedDecisionBundleProfile(
            LayaDecisionProfile.EnglishFp32.Name,
            LayaDecisionProfile.EnglishFp32Revision)
    });
```

The heavyweight bundle is intentionally not committed. Ordinary tests use
small offline fixtures; real-model parity is an explicit opt-in acceptance
run.

## Run the samples

These are .NET 10 file-based apps with portable JIT execution
(`PublishAot=false`). The bundle path below is a safe repository-relative
placeholder; replace it with the directory or ZIP you prepared.

```powershell
# ML.NET facade (recommended)
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode facade --bundle .\models\laya-english-fp32.bundle

# ML.NET inspectable stages
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode stages --bundle .\models\laya-english-fp32.bundle

# ML.NET facade appended to another estimator
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- `
  --mode composed --bundle .\models\laya-english-fp32.bundle

# Direct core facade and stages
dotnet run --file samples/TypedDecisions/Standalone/Program.cs -- `
  --mode facade --bundle .\models\laya-english-fp32.bundle
dotnet run --file samples/TypedDecisions/Standalone/Program.cs -- `
  --mode stages --bundle .\models\laya-english-fp32.bundle
```

`facade` composes preparation, ONNX scoring, and decoding. `stages` prints
prepared and scored JSON envelopes before decoded results. The ML.NET facade
uses a lazy cursor and batches source rows; the staged transforms intentionally
transport JSON through scalar `Text` columns and score one row at a time so
their intermediate contracts can be inspected. They are not native tensor
columns or a replacement for facade batching. `composed` applies the facade
twice and prints both the original and appended result columns.

## Captured expected output

The following excerpts were captured from the pinned real-bundle runs of the
five commands above. They are representative outputs, not universal snapshots:
provider, runtime, package, or calibration changes can alter floating-point
digits. Stable labels, result ordering, finite values, and scalar-to-JSON
relationships are the important expectations; numerical comparisons use a
small tolerance.

Standalone facade (one request, three results):

```json
{"input_tokens":112,"results":[{"id":"priority","type":"choice","confidence":0.4831077,"action_probability":0,"labels":["low","high"],"probabilities":[0.115706585,0.88429344],"choice":"high"},{"id":"quality","type":"score","confidence":0.21570939,"action_probability":0,"labels":["0","1","2"],"probabilities":[0.09035328,0.6336471,0.27599967],"score":1.1856464,"legend":{"0":"weak","1":"moderate","2":"strong"}},{"id":"actionable","type":"noul","confidence":0.39534837,"action_probability":0,"labels":["false","true"],"probabilities":[0.14793624,0.8520637],"noul":true,"probability_true":0.8520637}]}
```

ML.NET facade rows (the scalar columns select the first matching result for
choice, score, and `probability_true`; confidence and action probability come
from the first result overall):

| State row | Choice | Score | `probability_true` | Confidence | Action probability |
|---|---:|---:|---:|---:|---:|
| Reproducible steps, urgent fix | `high` | `1.1856464` | `0.8520637` | `0.4831077` | `0` |
| Missing logs, no requested action | `low` | `0.6199823` | `0.10274245` | `0.5969069` | `0` |

The corresponding full result JSON excerpts are:

```json
{"input_tokens":112,"results":[{"id":"priority","type":"choice","confidence":0.4831077,"action_probability":0,"labels":["low","high"],"probabilities":[0.115706585,0.88429344],"choice":"high"},{"id":"quality","type":"score","confidence":0.21570939,"action_probability":0,"labels":["0","1","2"],"probabilities":[0.09035328,0.6336471,0.27599967],"score":1.1856464,"legend":{"0":"weak","1":"moderate","2":"strong"}},{"id":"actionable","type":"noul","confidence":0.39534837,"action_probability":0,"labels":["false","true"],"probabilities":[0.14793624,0.8520637],"noul":true,"probability_true":0.8520637}]}
{"input_tokens":115,"results":[{"id":"priority","type":"choice","confidence":0.5969069,"action_probability":0,"labels":["low","high"],"probabilities":[0.91974044,0.08025956],"choice":"low"},{"id":"quality","type":"score","confidence":0.22399896,"action_probability":0,"labels":["0","1","2"],"probabilities":[0.42993295,0.52015173,0.04991528],"score":0.6199823,"legend":{"0":"weak","1":"moderate","2":"strong"}},{"id":"actionable","type":"noul","confidence":0.5223708,"action_probability":0,"labels":["false","true"],"probabilities":[0.89725757,0.10274245],"noul":false,"probability_true":0.10274245}]}
```

The composed mode prints the same first and second rows twice: once from the
original facade columns and once from the appended facade columns. The
captured appended scalar excerpts were:

```text
appended_choice=high; appended_score=1.1856464; appended_true_probability=0.8520637; appended_confidence=0.4831077; appended_action_probability=0
appended_choice=low; appended_score=0.6199823; appended_true_probability=0.10274245; appended_confidence=0.5969069; appended_action_probability=0
```

Each appended JSON result retained the same three typed results and
distributions as the corresponding original row. The acceptance comparison
checked both rows, original versus appended columns, and probability
normalization within `1e-6`.

For the complete API/type mapping and the direct-core sample, see
[Standalone/README.md](Standalone/README.md). For schema-aware ML.NET
composition and column details, see
[MLNetPipeline/README.md](MLNetPipeline/README.md).
