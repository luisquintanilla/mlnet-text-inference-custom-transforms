# ML.NET typed decisions: a guided tutorial

This guide starts with one useful result, teaches the vocabulary needed to
understand it, follows one source row through preparation, ONNX execution, and
decoding, and ends with execution choices and portable deployment. The small
[orientation page](../README.md) tells you where to begin; this page is the
step-by-step lesson and technical reference.

## Quick navigation

- [Run the sample](#3-run-one-recommended-path)
- [Copy the complete facade example](#5-a-complete-followable-mlnet-pipeline)
- [Understand the request boundary](#6-walk-one-request-through-the-implementation)
- [Choose direct, staged, or composed execution](#8-choose-an-execution-mode)
- [Save and load for deployment](#portable-saveload-for-deployment)
- [Read exact captures and implementation links](#captured-outputs)

## 1. The problem and the result

A typed-decision model scores alternatives supplied by the application. It
does not generate prose, infer an unbounded answer, execute a side effect, or
replace application policy. For each `State`, this sample asks:

```csharp
using MLNet.TextInference.TypedDecisions;
using static MLNet.TextInference.TypedDecisions.DecisionQuestion;

var questions = new[]
{
    Choice("priority", "How urgent is the request?", ["low", "high"]),
    Score("quality", "How strong is the evidence?", ["weak", "moderate", "strong"]),
    Noul("actionable", "Can the request be acted on now?")
};
```

`Choice` selects a caller-supplied label. `Score` returns an expected
zero-based option index (`0..2` here), not a percentage. `Noul` is this API's
name for a Boolean question and exposes both the Boolean result and `P(true)`.
The full state text is:

```text
The customer supplied reproducible steps and requested an urgent fix.
The report is missing logs and has no clear requested action.
```

The pinned CPU capture decodes those rows as:

| State | Priority | Quality score | Actionable | `P(true)` |
|---|---|---:|---|---:|
| Reproducible steps, urgent fix | `high` | `1.1856464` | `true` | `0.8520638` |
| Missing logs, no clear action | `low` | `0.61998236` | `false` | `0.10274245` |

The exact distributions, confidence values, legends, and provenance are
collected later in [Captured outputs](#captured-outputs). They are a pinned
model/runtime snapshot, not universal business truth or an empirical
calibration claim.

## 2. Before you run the sample

Use a checkout of this repository and install .NET 10. The file-based sample's
`#:project` directive references
`src/MLNet.TextInference.Onnx/MLNet.TextInference.Onnx.csproj` in that
checkout; it is not a claim that a published NuGet version already contains
these PR APIs. The first `dotnet run` may restore NuGet packages and therefore
needs network access. After restore and asset acquisition, inference itself is
offline: it reads only the local directory or portable ZIP you provide.

```text
models/laya-english-fp32/
  laya.onnx
  laya.onnx.data
  laya_config.json
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
```

The graph's external-data sidecar must stay beside `laya.onnx`, and the
tokenizer/profile must belong to the graph. This tutorial targets
[`receptron/laya-onnx` revision
`68f27dfe5a27a54fb2b1fefc432f43f972e90868`](https://huggingface.co/receptron/laya-onnx/tree/68f27dfe5a27a54fb2b1fefc432f43f972e90868).
The heavyweight weights are not committed; ordinary tests use a small offline
ONNX fixture instead.

### Optional one-time asset acquisition

The following PowerShell commands download the exact five files from that
pinned revision. Run them **from the repository root**. The `laya.onnx.data`
file is about 1.7 GB; use a stable connection and keep it beside
`laya.onnx`. These are real model assets, not placeholders. Do not run these
commands if you only want to build the sample or inspect the offline tests.

Each filename is also a direct source link:

- [`laya.onnx` (3.8 MB)](https://huggingface.co/receptron/laya-onnx/resolve/68f27dfe5a27a54fb2b1fefc432f43f972e90868/laya.onnx)
- [`laya.onnx.data` (about 1.7 GB)](https://huggingface.co/receptron/laya-onnx/resolve/68f27dfe5a27a54fb2b1fefc432f43f972e90868/laya.onnx.data)
- [`laya_config.json`](https://huggingface.co/receptron/laya-onnx/resolve/68f27dfe5a27a54fb2b1fefc432f43f972e90868/laya_config.json)
- [`tokenizer/tokenizer.json`](https://huggingface.co/receptron/laya-onnx/resolve/68f27dfe5a27a54fb2b1fefc432f43f972e90868/tokenizer/tokenizer.json)
- [`tokenizer/tokenizer_config.json`](https://huggingface.co/receptron/laya-onnx/resolve/68f27dfe5a27a54fb2b1fefc432f43f972e90868/tokenizer/tokenizer_config.json)

```powershell
$revision = "68f27dfe5a27a54fb2b1fefc432f43f972e90868"
$root = Join-Path (Get-Location) "models\laya-english-fp32"
$tokenizer = Join-Path $root "tokenizer"
New-Item -ItemType Directory -Force -Path $tokenizer | Out-Null
$base = "https://huggingface.co/receptron/laya-onnx/resolve/$revision"

Invoke-WebRequest "$base/laya.onnx" `
  -OutFile (Join-Path $root "laya.onnx")
Invoke-WebRequest "$base/laya.onnx.data" `
  -OutFile (Join-Path $root "laya.onnx.data")
Invoke-WebRequest "$base/laya_config.json" `
  -OutFile (Join-Path $root "laya_config.json")
Invoke-WebRequest "$base/tokenizer/tokenizer.json" `
  -OutFile (Join-Path $tokenizer "tokenizer.json")
Invoke-WebRequest "$base/tokenizer/tokenizer_config.json" `
  -OutFile (Join-Path $tokenizer "tokenizer_config.json")
```

The first run of the sample may trigger a NuGet restore after the model
download; later runs with the same packages and files do not need network
access.

## 3. Run one recommended path

Start with the facade. It is the all-in-one preparation, ONNX scoring, and
decoding transform; it is the smallest example that demonstrates schema
validation, lazy ML.NET execution, cursor batching, typed columns, and
explicit ownership:

```powershell
dotnet run --file .\samples\TypedDecisions\MLNetPipeline\Program.cs -- `
  --mode facade --model-assets .\models\laya-english-fp32
```

`--bundle` is a compatibility alias for `--model-assets`; either value may be
a directory or a portable archive. The command uses portable JIT execution
(`PublishAot=false`) and the same source rows/questions shown above. If it
fails before inference, check the working directory, the graph/sidecar pair,
the tokenizer directory, and the pinned profile before changing code.

You should see typed scalar/vector fields and a diagnostic `DecisionResults`
JSON column. The next sections explain how to consume those values without
confusing a score with confidence, `P(true)` with an action policy, or a
lazy `IDataView` with already-materialized data.

## 4. Vocabulary before implementation

These terms are introduced here so the later trace has a concrete meaning:

- **State** is the per-row text being evaluated. If an application has JSON,
  it serializes that JSON into the text column itself; there is no implicit
  Python-compatible object serializer.
- An **`IDataView`** is ML.NET's schema-aware view of rows and columns. A
  **schema** is the list of column names, types, and vector shapes that a view
  promises. A view may be lazy: it describes how to obtain rows without
  having read every row yet.
- An **estimator** is a recipe. Calling `Fit` checks the input schema and
  returns an initialized **transformer**. For this pretrained typed-decision
  estimator, `Fit` initializes tokenizer/profile/session resources; it does
  not train or fine-tune the graph. `Transform` returns another `IDataView`
  recipe, and `foreach`, a cursor, `ToArray`, or a requested getter enumerates
  it and performs the work.
- A **DTO** (data-transfer object) is an ordinary C# result class whose
  properties receive mapped columns. `PredictionEngine` is a convenient
  single-row mapper from an input DTO to an output DTO; it is not a batch
  engine and is not thread-safe.
- **ONNX** is the exported graph format. **ONNX Runtime** executes that graph.
  Microsoft.ML.Tokenizers supplies the selected BPE encoding boundary.
  Preparation and decoding remain explicit C#; they are not automatically
  embedded in the ONNX file.
- A **token** is one ID produced by the profile's tokenizer. BPE (byte-pair
  encoding) repeatedly joins configured symbol pairs into model-specific
  pieces; a piece may be a word fragment or a byte-level space-marked piece,
  not necessarily a whole word. The adapter uses
  `BpeOptions`/`BpeTokenizer` once at this boundary and then passes IDs to the
  graph.
- A **tensor** is a typed multidimensional array. A shape such as `[B,L]`
  says how many values exist along each dimension; `B`, `L`, and `K` are
  explained before the captured trace. ML.NET transports some tensors as flat
  `VBuffer<T>` columns plus scalar dimensions, while ONNX Runtime receives
  shaped buffers.
- **Batching** is an implementation choice. Direct bulk `Infer` can batch
  states, while ML.NET cursor stages can batch prepared question rows and
  regroup them. ML.NET is not promised to be faster, more accurate, or the
  only batching option.

**Checkpoint:** you should now know the difference between a schema and its
rows, an estimator and its fitted transformer, and a flat `VBuffer<T>` and a
shaped graph tensor. You also know why seeing no work at `Transform` is
expected until something enumerates the result.

## 5. A complete, followable ML.NET pipeline

This is a small complete file-based app, not an excerpt with undefined
`ml`, `data`, or `options` variables. Create a file named
`DocsWalkthrough.cs` beside `Program.cs`, paste the complete block below, and
run it from the repository root. The directives are the same project/package
surface used by the sample:

```powershell
dotnet run --file .\samples\TypedDecisions\MLNetPipeline\DocsWalkthrough.cs -- `
  .\models\laya-english-fp32
```

The one argument is the local asset directory. The app reads the two sample
states, materializes the typed result rows, and prints scalar values. It does
not print probability vectors yet; those are explained after the shape and
mapping boundaries.

```csharp
// Save as DocsWalkthrough.cs beside Program.cs.
#:project ../../../src/MLNet.TextInference.Onnx/MLNet.TextInference.Onnx.csproj
#:package Microsoft.ML@5.0.0
#:package Microsoft.ML.OnnxRuntime@1.24.2
#:property PublishAot=false

using Microsoft.ML;
using MLNet.TextInference.Onnx;
using MLNet.TextInference.TypedDecisions;
using static MLNet.TextInference.TypedDecisions.DecisionQuestion;

var modelAssetsPath = args is [var assetPath]
    ? assetPath
    : throw new ArgumentException(
        "Pass the local laya-english-fp32 directory as the only argument.");
var ml = new MLContext(seed: 1);
var data = ml.Data.LoadFromEnumerable(new[]
{
    new StateRow { State = "The customer supplied reproducible steps and requested an urgent fix." },
    new StateRow { State = "The report is missing logs and has no clear requested action." }
});

var options = new OnnxTypedDecisionsOptions
{
    ModelAssetsPath = modelAssetsPath,
    Questions =
    [
        Choice("priority", "How urgent is the request?", ["low", "high"]),
        Score("quality", "How strong is the evidence?", ["weak", "moderate", "strong"]),
        Noul("actionable", "Can the request be acted on now?")
    ],
    BatchSize = 16
};

using var fitted = ml.Transforms.OnnxTypedDecisions(options).Fit(data);
var lazyResults = fitted.Transform(data);
var rows = ml.Data.CreateEnumerable<DecisionRow>(
    lazyResults, reuseRowObject: false).ToArray();

foreach (var row in rows)
{
    Console.WriteLine(
        $"priority={row.Decision_priority_PredictedLabel}; " +
        $"quality_score={row.Decision_quality_Score}; " +
        $"actionable={row.Decision_actionable_PredictedLabel}; " +
        $"actionable_true_probability={row.Decision_actionable_Probability}");
}

public sealed class StateRow
{
    public string State { get; set; } = string.Empty;
}

public class DecisionRow
{
    public string Decision_priority_PredictedLabel { get; set; } = string.Empty;
    public float Decision_quality_Score { get; set; }
    public bool Decision_actionable_PredictedLabel { get; set; }
    public float Decision_actionable_Probability { get; set; }
}
```

The direct, stage, composition, and Save/Load examples later in this guide
are contextual top-level continuations. Paste those statements **before** the
`StateRow` and `DecisionRow` declarations at the end of this file; C# requires
top-level statements to come before type declarations. For the stage example,
add `using Microsoft.ML.Data;` and copy the `StageRow` DTO from
[`Program.cs`](Program.cs); it inherits from the non-sealed `DecisionRow`
shown above.

For the pinned assets, the two printed lines are:

```text
priority=high; quality_score=1.1856464; actionable=True; actionable_true_probability=0.8520638
priority=low; quality_score=0.61998236; actionable=False; actionable_true_probability=0.10274245
```

Read the code in four small steps:

1. `LoadFromEnumerable` creates an `IDataView` whose schema contains one
   `State` text column.
2. `OnnxTypedDecisions(options)` creates an estimator recipe. `Fit` validates
   that schema and initializes the local tokenizer/profile/session.
3. `Transform` creates a lazy result view. `CreateEnumerable` returns a
   deferred `IEnumerable<DecisionRow>`; `ToArray()` is what enumerates it and
   causes preparation, graph execution, decoding, and mapping. A `foreach`
   would also enumerate it.
4. The DTO receives only the scalar columns needed for this first lesson.
   Probability vectors and native stage columns still exist in the output
   schema; they are deliberately introduced later.

The input property is named `State` because the options use the default
`StateColumnName = "State"`. If you change that option, rename the input
property or map it with an ML.NET column attribute. Output property names are
generated from each question ID and `OutputPrefix`: for example,
`priority` plus the default `Decision_` prefix produces
`Decision_priority_PredictedLabel`. If you rename a question ID or prefix,
update the DTO properties (or their `[ColumnName]` mappings) to match.

The fitted transformer owns the tokenizer, profile, decoder, and ONNX
session, so it is disposed only after `rows` and any other borrowed consumer
are finished. The optional explicit stages and their additional tensor DTO
are shown in [execution choices](#8-choose-an-execution-mode), after the
facade has established the basic vocabulary. The public extension methods and
role types are in
[`MLContextExtensions.cs`](../../../src/MLNet.TextInference.Onnx/MLContextExtensions.cs)
and
[`TypedDecisionEstimators.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/TypedDecisionEstimators.cs).

**Checkpoint:** you should now be able to explain why a transformer can be
fitted once and consumed by many materializations, and why disposing it too
early invalidates borrowed mappers/cursors.

<a id="beginner-and-developer-walkthrough"></a>

## 6. Walk one request through the implementation

The following subsections use the same four boundaries as the complete
example: prepare, score, decode, and map. The visual overview is useful
before the numeric details:

![Flow diagram showing caller state and fixed questions moving through C# preparation, five tensors, ONNX Runtime, separate logits and action-head decoding, and direct or ML.NET outputs.](images/pipeline-overview.svg)

*Figure 1. The conceptual flow: the logits path decodes valid options into
typed decisions, while the already-normalized action head remains a separate
diagnostic channel. [Open the full-size editable SVG](images/pipeline-overview.svg).*

ONNX Runtime evaluates the graph; it does not expose a reasoning trace. The
direct facade, lazy facade, and explicit stages share the same kernels and
asset contract. The four conceptual boundaries are:

1. **Prepare** renders and encodes each state/question pair into five graph
   inputs.
2. **Score** validates the declared types/shapes and executes ONNX Runtime.
3. **Decode** converts logits and the already-softmaxed action head into
   Choice, Score, and Noul values.
4. **Map** exposes those values as direct C# results, ML.NET columns, or DTO
   properties.

### 6.1 Trusted assets and `Fit`

Bundle validation keeps paths relative to the bundle root, preserves
external-data locations, checks declared file hashes, and rejects ZIP entries
that escape the extraction destination. It does not download a missing file
or silently select another revision. See
[`TypedDecisionBundle.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/TypedDecisionBundle.cs)
and [`AssetArchive.cs`](../../../src/MLNet.TextInference.Onnx/AssetArchive.cs).

The profile's `max_len`, `head_max_len`, special-token metadata, and
temperature policy must stay paired with its graph and tokenizer. A
different-looking-but-compatible tokenizer can still change IDs, marker
positions, and sequence layout.

### 6.2 Token IDs and the Microsoft tokenizer boundary

The selected profile is a Hugging Face byte-level BPE. BPE pieces are
model-specific subwords, not necessarily words. The adapter uses
`BpeOptions`/`BpeTokenizer` from `Microsoft.ML.Tokenizers`; it does not add a
second tokenizer runtime. It reads vocabulary, merges, special tokens, and
added-token metadata, preserving the selected byte-level/NFC behavior and
rejecting unsupported approximations. See
[`HuggingFaceBpeTokenizerLoader.cs`](../../../src/MLNet.TextInference.Onnx/HuggingFaceBpeTokenizerLoader.cs),
[`TokenizerEncoding.cs`](../../../src/MLNet.TextInference.Onnx/TokenizerEncoding.cs),
and [`LayaTokenizer.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/LayaTokenizer.cs).

### 6.3 Prepare one sequence per state-question pair

`PrepareDecisionInputs` creates one sequence for every question for every
state. With two states and three questions, a bulk request has six logical
flattened rows. The profile-specific construction is:

1. Render options in caller order. Choice uses labels (or `label:
   description`); Score uses `level {index}: {level}`; Noul uses its explicit
   false/true criteria.
2. Scrub the configured mask-token literal from caller text so it cannot
   create an extra marker.
3. Encode each option separately as BPE of a string with a literal leading
   space, then prefix that option's tokens with the configured MASK ID.
   `marker_pos` points to the inserted MASK, not option text.
4. Encode the head separately as
   `{choice|score|noul} question: {instructions}`.
5. Assemble `[CLS] + head + [SEP] + option sequences + [SEP] + state + [SEP]`.
6. Bound head/option material with `head_max_len`, then give remaining
   `max_len` room to the state. There is no automatic chunking; a long state
   tail can be truncated. Marker loss is rejected.
7. Pad to the prepared batch's maximum sequence length using the configured
   PAD ID.

This is why one `EncodeToIds` call over the displayed sentence is not
equivalent: separate BPE chunks, the leading option space, and inserted MASK
IDs are part of the contract. The implementation is in
[`PrepareDecisionInputs.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/PrepareDecisionInputs.cs).

### 6.4 `B`, `L`, `K`, masks, and the five graph inputs

Define the shape vocabulary before reading the trace:

- `B` is the flattened number of question sequences, not source
  `IDataView` rows. In the first source state, `B=3`.
- `L` is the padded token sequence length (`45` or `46` in the captured
  source rows).
- `K` is the maximum option/marker width (`3` here). Choice and Noul have
  two valid slots and one padded slot; Score has three.

The graph contract is:

| Input | Type and shape | Meaning |
|---|---|---|
| `input_ids` | `Int64 [B,L]` | Padded BPE token IDs. |
| `attention_mask` | `Int64 [B,L]` | `1` for nonpadding, `0` for padding. |
| `marker_pos` | `Int64 [B,K]` | Positions of option MASK markers. |
| `marker_mask` | `Bool [B,K]` | Which marker slots are real options. |
| `qtype` | `Int64 [B]` | Choice `0`, Score `1`, Noul `2`. |

ML.NET stages carry flat `VBuffer<T>` values and scalar dimensions such as
`DecisionBatchSize`, `DecisionSequenceLength`, and `DecisionMarkerWidth`.
The scorer wraps those values into shaped ORT tensors. That is a
representation boundary, not a second tokenizer or universal Tensor layer.

![Batching diagram showing two source states with captured prepared shapes 3 by 45 and 3 by 46, an illustrative repadded combined batch 6 by 46, regrouping, token attention masking, and the K equals 3 marker-mask matrix.](images/batching-and-masks.svg)

*Figure 2. Source rows are prepared per state, can be repadded and combined
within a cursor, then sliced back to source rows. The combined `[6,46]` is
an explanatory illustration, not a replacement for the captured per-source
shapes. [Open the full-size editable SVG](images/batching-and-masks.svg).*

**Checkpoint:** `B` is not source-row count, `K` is not always the number of
valid options, and `marker_mask`—not a zero position—identifies padding.

### 6.5 Follow one source row across stage boundaries

<a id="5a-follow-one-source-row-across-stage-boundaries"></a>

![Worked stage I/O trace showing the first source state, rendered questions, real tokenizer-only preparation dimensions and ID slices, flat ML.NET VBuffers versus shaped ONNX Runtime tensors, separate output heads, an illustrative temperature-one softmax calculation, and the real decoded response and typed ML.NET columns.](images/stage-io-trace.svg)

*Figure 3. A boundary-by-boundary trace for the first `State` value and its
three configured questions. Green cards are source-backed evidence; the
hatched card is a complete tiny fixture used only to show decoder mechanics.
[Open the full-size editable SVG](images/stage-io-trace.svg).*

This graphic combines two evidence sets rather than implying a new
end-to-end run:

- **Preparation rerun:** only the pinned tokenizer files
  (`tokenizer.json`, `tokenizer_config.json`) and `laya_config.json` were
  downloaded from the English FP32 Laya revision. No weights were downloaded
  and no graph was executed. The public preparation stage reproduced the
  first source row as `B=3`, `L=45`, `K=3`, with 135 flattened ID positions,
  `marker_pos` rows `[11,13]`, `[11,16,21]`, `[14,24]`, `qtype=[0,1,2]`,
  and nonpadding counts `[28,39,45]`. Their sum is `input_tokens=112`; it
  is not the padded array length.
- **Earlier inference capture:** retained graph outputs were `logits [3,3]`
  (9 float values) and `act_probs [3,2]` (6 values) per source after
  regrouping. If scored alone, this source uses ORT input `[3,45]`; a cursor
  combining two source rows may repad a forward call to `[6,46]`. Raw logits
  and every action-head channel were not retained, so this guide does not
  reconstruct them from normalized probabilities.

For this sample, `RenderOptions()` produces `low`, `high`; `level 0: weak`,
`level 1: moderate`, `level 2: strong`; and the exact Noul strings
`false: no, the statement does not hold` and
`true: yes, the statement holds`. The public preparation surface is:

The following is a **contextual continuation, not a standalone snippet**. It
uses the `ml`, `data`, and `options` variables from the complete app in
[section 5](#5-a-complete-followable-mlnet-pipeline), or the equivalent setup
in `Program.cs`.

```csharp
using var preparationTransformer = ml.Transforms.PrepareDecisionInputs(
        new DecisionInputPreparationOptions
        {
            ModelAssetsPath = options.ModelAssetsPath,
            Questions = options.Questions
        })
    .Fit(data);
var prepared = preparationTransformer.Transform(data);
```

The fitted owner must remain alive while `prepared` is consumed. For the
first source row, the preparation columns are:

```text
DecisionBatchSize       = 3
DecisionSequenceLength = 45
DecisionMarkerWidth    = 3
DecisionInputIds        = VBuffer<long> length 135
DecisionAttentionMask   = VBuffer<long> length 135
DecisionMarkerPositions = VBuffer<long> length 9
DecisionMarkerMask      = VBuffer<bool> length 9
DecisionQuestionTypes   = VBuffer<long> values [0, 1, 2]

first 15 DecisionInputIds =
  [50281, 22122, 1953, 27, 1359, 21007, 310, 253,
   2748, 32, 50282, 50284, 1698, 50284, 1029]
marker positions by question =
  priority [11, 13]; quality [11, 16, 21]; actionable [14, 24]
marker mask by question =
  priority [true, true, false];
  quality [true, true, true];
  actionable [true, true, false]
```

### 6.6 Complete tensor capture and token IDs

The compact excerpt above is enough to understand the boundary. The complete
tokenizer-only capture is retained here for readers who need to inspect every
value or compare a stage dump.

<details>
<summary>All five prepared tensors as shaped JSON from the tokenizer-only rerun</summary>

```json
{
  "input_ids": [
    [
      50281, 22122, 1953, 27, 1359, 21007, 310, 253, 2748, 32, 50282, 50284, 1698, 50284, 1029,
      50282, 510, 7731, 12164, 41374, 5018, 285, 9521, 271, 21007, 4993, 15, 50282, 50283, 50283,
      50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283, 50283
    ],
    [
      50281, 18891, 1953, 27, 1359, 2266, 310, 253, 1941, 32, 50282, 50284, 1268, 470, 27,
      5075, 50284, 1268, 337, 27, 10290, 50284, 1268, 374, 27, 2266, 50282, 510, 7731, 12164,
      41374, 5018, 285, 9521, 271, 21007, 4993, 15, 50282, 50283, 50283, 50283, 50283, 50283, 50283
    ],
    [
      50281, 79, 3941, 1953, 27, 2615, 253, 2748, 320, 14001, 327, 1024, 32, 50282, 50284,
      3221, 27, 642, 13, 253, 3908, 1057, 417, 2186, 50284, 2032, 27, 4754, 13, 253,
      3908, 6556, 50282, 510, 7731, 12164, 41374, 5018, 285, 9521, 271, 21007, 4993, 15, 50282
    ]
  ],
  "attention_mask": [
    [
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    [
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0
    ],
    [
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
  ],
  "marker_pos": [
    [11, 13, 0],
    [11, 16, 21],
    [14, 24, 0]
  ],
  "marker_mask": [
    [true, true, false],
    [true, true, true],
    [true, true, false]
  ],
  "qtype": [0, 1, 2]
}
```

Each `input_ids` and `attention_mask` row has 45 values. The padded
`marker_pos` zeros are storage for the unused K=3 slots; the corresponding
`marker_mask=false` entries exclude them. The first attention row has 28 ones
and 17 zeros, while its two valid option slots are selected independently by
`marker_mask=[true,true,false]`.
</details>

The pinned tokenizer vocabulary maps the most useful first-row positions as
follows:

| Position | ID | Pinned tokenizer piece / meaning |
|---:|---:|---|
| 0 | 50281 | `[CLS]` |
| 1 | 22122 | `choice` |
| 2 | 1953 | `Ġquestion` (`Ġ` is the byte-level space marker) |
| 3 | 27 | `:` |
| 4-9 | 1359, 21007, 310, 253, 2748, 32 | `ĠHow Ġurgent Ġis Ġthe Ġrequest ?` |
| 10 | 50282 | `[SEP]` |
| 11 | 50284 | `[MASK]` before `low` |
| 12 | 1698 | `Ġlow` |
| 13 | 50284 | `[MASK]` before `high` |
| 14 | 1029 | `Ġhigh` |
| 15 | 50282 | `[SEP]` |
| 16 | 510 | `The` (start of the state) |
| 28+ | 50283 | `[PAD]` (padding begins after the first row's 28 attention ones) |

These pieces were looked up in the pinned tokenizer vocabulary; they are not
inferred from the decoded probabilities. The public preparation stage emits
IDs and offsets, not a token-string column. The five ML.NET vectors are flat
`VBuffer<T>` values plus the three scalar dimension columns. The scorer uses
those dimensions to wrap equivalent ORT shapes: `input_ids` and
`attention_mask` become `Int64 [3,45]`, `marker_pos` and `marker_mask` become
`[3,3]`, and `qtype` becomes `Int64 [3]`. This is a representation boundary,
not a second tokenizer or a universal `Tensor<T>` layer.

The first priority row's rendered text view is:

```text
[CLS] choice question: How urgent is the request? [SEP]
[MASK] low [MASK] high [SEP]
The customer supplied reproducible steps and requested an urgent fix. [SEP]
```

The implementation encodes the question head, each option, and the state as
separate chunks, then assembles them around the configured special-token
markers; a single `EncodeToIds` call over the displayed sentence is not
guaranteed to produce the same IDs because byte-level BPE boundaries, the
leading option space, and the inserted MASK IDs are part of the contract. The
table maps the selected real IDs from that assembled row, including the two
valid priority markers at positions 11 and 13.

For the missing raw-score arithmetic, the hatched lane uses an explicitly
illustrative fixture, not a model output:

```text
valid marker_mask = [true, true, false]
illustrative logits = [0.4, 1.2, 0.0]
valid-only logits = [0.4, 1.2]       # slot 2 is excluded, not zeroed
temperature T = 1.0                  # illustrative, not the pinned profile values
m = 1.2
exp(0.4 - m) = 0.4493; exp(1.2 - m) = 1
sum = 1.4493
softmax = [0.4493 / 1.4493, 1 / 1.4493] ~= [0.3100, 0.6900]
illustrative act_probs = [0.25, 0.75] -> select configured channel 0.75
```

The real first-row decoder output is the separate captured result:
`priority` probabilities `[0.11570658, 0.88429344]` select `"high"`;
`quality` probabilities `[0.09035327, 0.633647, 0.2759997]` produce
`1.1856464` on the `0..2` Score range; and `actionable` probabilities
`[0.14793625, 0.8520638]` produce `true`. Direct `Infer` returns a
`DecisionResponse` with typed `DecisionResult` records and
`InputTokenCount=112`; facade/stage paths expose properties such as
`Decision_priority_PredictedLabel`, `Decision_quality_Score`,
`Decision_actionable_PredictedLabel`, and
`Decision_actionable_Probability`, plus optional `DecisionResults` diagnostic
JSON. `ActionProbability=0` remains the separate graph head and does not
override the Boolean result. `DecisionRequest` and `DecisionInputBatch` are internal
implementation types, not public caller records.

In a debugger, the direct response and its native-column projection look like
this:

```text
DecisionResponse.Results[0] = ChoiceDecisionResult
  Choice = "high"
  -> Decision_priority_PredictedLabel = "high"

DecisionResponse.Results[1] = ScoreDecisionResult
  Score = 1.1856464
  -> Decision_quality_Score = 1.1856464

DecisionResponse.Results[2] = NoulDecisionResult
  Value = true; ProbabilityTrue = 0.8520638
  -> Decision_actionable_PredictedLabel = true
  -> Decision_actionable_Probability = 0.8520638
```

| Direct property | Value | Native output column |
|---|---:|---|
| `ChoiceDecisionResult.Choice` | `"high"` | `Decision_priority_PredictedLabel` |
| `ScoreDecisionResult.Score` | `1.1856464` | `Decision_quality_Score` |
| `NoulDecisionResult.Value` | `true` | `Decision_actionable_PredictedLabel` |
| `NoulDecisionResult.ProbabilityTrue` | `0.8520638` | `Decision_actionable_Probability` |

### 6.7 Run the graph, then decode only valid options

![Decision-decoding diagram with proportional 0 to 1 probability bars for the first captured source row: priority chooses high, quality yields expected index 1.1856464, and actionable chooses true.](images/decision-decoding.svg)

*Figure 4. The first captured source row, decoded with one shared probability
scale. The bars show observed probabilities, not invented logits. See
[Captured outputs](#captured-outputs) for the pinned model/runtime context.
The confidence and action-probability caveats are called out separately.
[Open the full-size editable SVG](images/decision-decoding.svg).*

The graph's `logits` are unnormalized learned scores, not probabilities.
Forward inference applies the trained weights to the whole prepared context
and scores the supplied options. The exported graph does not expose a
reasoning trace, so the adapter explains the observable tensors and decoded
outputs rather than inventing an internal model explanation.

The custom scorer binds the five named inputs and validates actual element
types and output shapes before decoding. It preserves adjacent ONNX external
data and expects:

- `logits` with shape `[B,K]`, containing one score per marker slot;
- `act_probs` with shape `[B,2]`, already softmaxed by the graph.

For logits, the decoder applies the profile temperature policy, clamps valid
temperatures to `[0.5, 5.0]`, subtracts the maximum valid logit, exponentiates,
and divides by the sum. The resulting softmax values are nonnegative and sum
to one over the valid choices. Subtracting the maximum preserves the
probability ratios while reducing overflow risk. A temperature below `1`
sharpens the distribution; a temperature above `1` flattens it. This is
decoding, not training or an accuracy-calibration claim. It runs only over
valid options: an invalid slot must be excluded, not assigned logit `0`,
because `exp(0)` would still add probability mass. `act_probs` is already a
probability vector, so it is read directly and must not be softmaxed again.

The scorer uses ONNX Runtime for model execution. It wraps the prepared
`long` and `bool` arrays with `OrtValue.CreateTensorValueFromMemory` and the
declared dimensions. In this decision decoder, `System.Numerics.Tensors` is
used only by the shared stable-softmax helper after `MathF.Exp`;
`TensorPrimitives.Sum` performs the reduction and `TensorPrimitives.Divide`
performs element-wise normalization. This is not a
replacement inference engine, a universal `Tensor<T>` wrapper, or an
end-to-end zero-copy guarantee.

The implementation is in
[ScoreOnnxDecisionModel.cs](../../../src/MLNet.TextInference.Onnx/TypedDecisions/ScoreOnnxDecisionModel.cs),
[DecodeDecisions.cs](../../../src/MLNet.TextInference.Onnx/TypedDecisions/DecodeDecisions.cs),
and [StableSoftmax.cs](../../../src/MLNet.TextInference.Onnx/Numerics/StableSoftmax.cs).
The [ONNX Runtime C# guide](https://onnxruntime.ai/docs/get-started/with-csharp.html)
provides the general session/inference background; this scorer adds the
Laya-specific input and output contract described above.

The decoded values follow the question semantics:

- **Choice:** row 1 has `high` because `0.88429344` is larger than
  `0.11570658`; row 2 has `low` because `0.9197405` is larger than
  `0.08025956`.
- **Score:** the first row is
  `0*0.09035327 + 1*0.633647 + 2*0.2759997 = 1.1856464`, and the second
  is `0*0.42993295 + 1*0.5201518 + 2*0.049915284 = 0.61998236`.
  This is an expected ordinal index under the equal-spacing assumption, not
  a claim that the distance from `weak` to `moderate` equals the distance
  from `moderate` to `strong` in the real world.
- **Noul:** `P(true)` is the second value in `[false, true]`. The Boolean
  result is true when `P(true) >= P(false)`.

Confidence is a normalized concentration measure, not the winning
probability, accuracy, or empirical calibration:

```text
H(p) = -sum(p * ln(p))
confidence = 1 - H(p) / ln(valid option count)
```

Entropy approaches its maximum for a uniform distribution, so confidence
approaches `0`; it approaches `1` when probability is concentrated. For
example, row 1 `priority` has winning probability `0.88429344` but confidence
`0.4831077`. The separate `action_probability` comes from the graph's
configured action head. A captured zero does not mean `actionable=false`, does
not override the Noul result, and is not explained here beyond what the graph
returned.

## 7. Read typed results

Direct `Infer` returns typed C# `DecisionResponse` objects; it does not
materialize an `IDataView` or ML.NET columns on every call. The facade and
stage paths produce typed columns named from the configured question IDs:
Choice gets a text label and probabilities, Score gets a float score and
probabilities, and Noul gets a Boolean plus `P(true)`. All questions also
expose confidence and the separate action probability. Probability vectors
carry `SlotNames` metadata. `DecisionResults` is optional diagnostic JSON
containing the complete response, including distributions, legends, and
derived values; it is not the transport format between native stages.

Each configured question receives an unambiguous prefix:

| Question | Columns in this sample |
|---|---|
| `priority` Choice | `Decision_priority_PredictedLabel`, `Decision_priority_Probabilities`, `Decision_priority_Confidence`, `Decision_priority_ActionProbability` |
| `quality` Score | `Decision_quality_Score`, `Decision_quality_Probabilities`, `Decision_quality_Confidence`, `Decision_quality_ActionProbability` |
| `actionable` Noul | `Decision_actionable_PredictedLabel`, `Decision_actionable_Probability`, `Decision_actionable_Confidence`, `Decision_actionable_ActionProbability` |

Choice labels are text. Score is the expected zero-based option index, not a
confidence. Noul selects true when `P(true) >= P(false)`. Probability vectors
are temperature-adjusted option probabilities (not logits) and carry
`SlotNames` metadata with option labels. Every confidence and
action-probability scalar is associated with its own question.

The composed mode uses `OutputPrefix = "AppendedDecision_"` and
`ResultsColumnName = "AppendedDecisionResults"`, so it adds corresponding
`AppendedDecision_*` columns without overwriting the first facade's columns.
Both facades use the same request and assets, so equivalent values should
match within floating-point tolerance. Accessing multiple output getters does
not repeat inference for the same cursor row.

The direct call, facade, and stage chain use the same kernels and asset
contract, but each separately fitted transformer owns its own resources. The
native vector DTOs used by the sample are in [`Program.cs`](Program.cs).
The public extension methods and typed mapping are implemented in
[`MLContextExtensions.cs`](../../../src/MLNet.TextInference.Onnx/MLContextExtensions.cs),
[`TypedDecisionEstimators.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/TypedDecisionEstimators.cs),
and
[`TypedDecisionRowMappers.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/TypedDecisionRowMappers.cs).

## 8. Choose an execution mode

All eleven sample modes use the same questions, states, assets, and fitted
package surface:

| Mode | Use it when you need to... |
|---|---|
| `direct` | Call `transformer.Infer` and inspect JSON response objects. |
| `facade` | Enumerate a lazy `IDataView` with typed columns and diagnostic JSON. |
| `prediction-engine` | Map one facade input row to a DTO. |
| `prediction-engine-stages` | Map one explicit-stage row while exposing native tensors. |
| `prediction-engine-composed` | Map both appended facade outputs through one DTO. |
| `stages` | Inspect preparation/scoring tensors and decoded columns. |
| `composed` | Append a second facade with `AppendedDecision_` output names. |
| `portable-writer` | Fit a facade and write a portable archive. |
| `portable-reader` | Load that archive without `Fit` or the original asset directory. |
| `portable-pipeline-writer` | Fit the demonstrated composed pipeline and write both outputs. |
| `portable-pipeline-reader` | Load both composed outputs without `Fit` or source assets. |

Writer modes require both `--model-assets` and `--portable-path`; reader modes
require only `--portable-path`.

The direct facade is useful when the application already owns text states and
does not need an `IDataView` boundary. Reuse the fitted facade from the
complete example:

```csharp
var firstState =
    "The customer supplied reproducible steps and requested an urgent fix.";
var secondState =
    "The report is missing logs and has no clear requested action.";
DecisionResponse one = fitted.Infer(
    firstState);
IReadOnlyList<DecisionResponse> many = fitted.Infer(
    new[] { firstState, secondState });
```

The bulk overload can batch states internally. It has the same tokenizer,
graph, and decoder contract as the facade path; choosing ML.NET is about
schema-aware composition and lazy interoperability, not exclusive batching.

If you need to see the native preparation and scoring columns, append the
three stages explicitly. This is a **contextual continuation** of section 5:
`ml`, `data`, and `options` already exist, and `StageRow` is the larger DTO
defined in [`Program.cs`](Program.cs).

```csharp
var stagePipeline = ml.Transforms.PrepareDecisionInputs(
        new DecisionInputPreparationOptions
        {
            ModelAssetsPath = options.ModelAssetsPath,
            Questions = options.Questions
        })
    .Append(ml.Transforms.ScoreOnnxDecisionModel(
        new OnnxDecisionModelScorerOptions
        {
            ModelAssetsPath = options.ModelAssetsPath
        }))
    .Append(ml.Transforms.DecodeDecisions(
        new DecisionDecodingOptions
        {
            ModelAssetsPath = options.ModelAssetsPath,
            Questions = options.Questions
        }));

using var staged = stagePipeline.Fit(data);
var stagedRows = ml.Data.CreateEnumerable<StageRow>(
    staged.Transform(data), reuseRowObject: false).ToArray();
Console.WriteLine($"source_rows={stagedRows.Length}");
```

`source_rows` is the number of source `IDataView` rows (two in this example);
it is not the flattened graph batch `B`, which is three question sequences for
each source state.

The compiled helper for appending another facade is:

```csharp
var appendedOptions = new OnnxTypedDecisionsOptions
{
    ModelAssetsPath = options.ModelAssetsPath,
    Questions = options.Questions,
    OutputPrefix = "AppendedDecision_",
    ResultsColumnName = "AppendedDecisionResults",
    BatchSize = 2
};

var composedPipeline = ml.Transforms.OnnxTypedDecisions(options)
    .AppendOnnxTypedDecisions(ml, appendedOptions);
var composed = composedPipeline.Fit(data);
try
{
    var composedView = composed.Transform(data);
    // Enumerate composedView with the ComposedDecisionRow DTO from Program.cs.
}
finally
{
    (composed as IDisposable)?.Dispose();
}
```

Composition reuses the same input rows and fitted asset paths; it is not a
merged model and does not produce a better prediction. Each facade has its
own fitted resources, while per-stage/per-row caching prevents repeated getter
execution within that stage and row. It does not promise cross-facade cache
sharing.

ML.NET transformations are lazy: `Transform` constructs a schema-aware
`IDataView`, while cursor enumeration and requested getters cause the row
work to happen. This is the behavior described by
[`ITransformer.Transform`](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.itransformer.transform).
Mapped rows and cursors borrow the fitted transformer's resources. In this
sample, `PredictionEngine` is created with `OwnsTransformer=false`, so
disposing these borrowed consumers does not dispose the shared fitted session;
other `PredictionEngine` ownership options may differ. The caller owns the
resource-owning fitted transformer and disposes it after its consumers finish.

`PredictionEngine` is a convenient single-row API, not a batch engine and not
thread-safe. Use one instance per concurrent caller, or use a pool. For
ordinary `IDataView` workloads, prefer the facade or explicit stages so cursor
batching can combine prepared rows and regroup their outputs.

Compatibility was demonstrated with ordinary `ApplyOnnxModel` binding for
this real Laya graph, including its Boolean input and rank-one `qtype`.
The custom production scorer remains intentional for direct execution and
ownership of the bundle/session contract, plus source-row batching,
re-padding, flattening, scoring, and regrouping. It is not retained because
built-in ONNX scoring is limited to single-input or static-dimension graphs;
these decision-specific responsibilities are simply clearer and safer in the
custom path.

## 9. Shared infrastructure versus task-specific behavior

The package shares tokenizer adaptation, numeric helpers, ONNX session
management, schema-aware mappers, and cursor lifetime/batching infrastructure
with the other text transforms. Typed decisions add the Laya-specific
sequence layout, option marker masks, qtype values, and Choice/Score/Noul
decoder. A conventional `PoolEmbedding` transform cannot substitute for this
decision head: pooling expects hidden states such as `[B,L,H]`, while this
graph consumes marker positions and question types and returns decision logits
plus an action head.

## 10. What belongs in an application

Treat the outputs as model signals that require application validation:

- Evaluate on representative, labelled data and choose review/error policies
  appropriate to the cost of false positives and false negatives.
- Do not treat concentration as accuracy or calibration. Temperature-adjusted
  option probabilities are precise about the decoder operation, not a claim
  of empirical calibration.
- Do not let a model output authorize an external side effect by itself.
  Application code should make any action explicit and separately governed.
- Changing the state, question wording, option order, tokenizer, model
  revision, provider, or temperature settings can change the result.

Deployment and persistence are covered separately in
[Portable Save/Load](#portable-saveload-for-deployment). Keeping that
boundary separate matters: native `MLContext.Model.Save`/`Load` remains
unsupported for these path-based custom components.

<a id="portable-saveload-for-deployment"></a>

## 11. Portable Save/Load for deployment

Portable persistence is a deployment artifact, not native ML.NET model
serialization. Individual facade and preparation/scoring/decoding stage
archives are supported. `SavePipeline` supports the demonstrated flat
prepare -> score -> decode chain and naturally inferred appended typed
facades with distinct prefixes, results columns, and question widths.

The archive stores fitted question metadata, column names, batching settings,
profile/decoder policy, tokenizer files, graph-relative external data, and
hashes. It does not store native sessions, cursors, delegates, absolute
paths, or a network dependency. Loading rebuilds runtime resources according
to the load context. Native `MLContext.Model.Save`/`Load` and automatic
whole-pipeline ONNX export remain unsupported.

Pipeline sources must reference the same complete asset payload. Separately
loaded selective profile-only/scorer-only stage archives cannot currently be
recombined and saved together; the API rejects that narrower v1 case
explicitly. Arbitrary, nested, or mixed native/custom chains are also
rejected. Individual selective stage archives remain supported, but they are
not a promise that independently loaded subsets can be unioned later.

### Save and load the facade in code

Continuing the complete app from section 5, the fitted facade can save a
portable ZIP. The same `Load` call can run in a new process without calling
`Fit` or retaining the original asset directory; the short continuation below
shows that call in place:

```csharp
var portablePath = Path.Combine("artifacts", "typed-decisions.zip");
Directory.CreateDirectory(Path.GetDirectoryName(portablePath)!);
fitted.Save(portablePath);

using var loaded = OnnxTypedDecisionsTransformer.Load(ml, portablePath);
var loadedRows = ml.Data.CreateEnumerable<DecisionRow>(
    loaded.Transform(data), reuseRowObject: false).ToArray();

foreach (var row in loadedRows)
{
    Console.WriteLine(
        $"loaded priority={row.Decision_priority_PredictedLabel}; " +
        $"quality_score={row.Decision_quality_Score}; " +
        $"actionable={row.Decision_actionable_PredictedLabel}");
}
```

This continuation uses the `fitted`, `ml`, `data`, and `DecisionRow` values
from section 5. `Load` rebuilds tokenizer and ONNX Runtime resources from the
ZIP; it does not call `Fit`, and it does not serialize native handles. Ship
the ZIP together with the application, its .NET/runtime dependencies, and the
ONNX Runtime provider/runtime files required by the deployment. A portable
typed-decision ZIP is not a native ML.NET model file, so do not pass it to
`MLContext.Model.Load` or assume
`PredictionEnginePool.FromFile` can load it.

### Run the portable sample

The writer needs local source assets and an output ZIP. The reader needs only
the ZIP, so it demonstrates the offline boundary:

```powershell
dotnet run --file .\samples\TypedDecisions\MLNetPipeline\Program.cs -- `
  --mode portable-writer `
  --model-assets .\models\laya-english-fp32 `
  --portable-path .\artifacts\typed-decisions.zip

dotnet run --file .\samples\TypedDecisions\MLNetPipeline\Program.cs -- `
  --mode portable-reader `
  --portable-path .\artifacts\typed-decisions.zip

dotnet run --file .\samples\TypedDecisions\MLNetPipeline\Program.cs -- `
  --mode portable-pipeline-writer `
  --model-assets .\models\laya-english-fp32 `
  --portable-path .\artifacts\typed-decisions-pipeline.zip

dotnet run --file .\samples\TypedDecisions\MLNetPipeline\Program.cs -- `
  --mode portable-pipeline-reader `
  --portable-path .\artifacts\typed-decisions-pipeline.zip
```

For a portability check, run the writer in one process, move the ZIP to a
different test-owned directory, remove only the isolated source-assets
directory, and run the reader in a fresh process. Do not delete a
caller-owned model directory. The tracked
[`PortableProcessHarness`](../PortableProcessHarness/Program.cs) provides the
offline acceptance shape for `facade`, `stages`, and `composed` archives; the
`TrackedFreshProcessHarnessRoundTripsStructuredFacadeStagesAndComposedArtifacts`
test runs separate writer and reader processes and compares structured JSON
including every typed field, ordered labels/probabilities, score legends,
confidence/action channels, tensor vectors, schema dimensions, hiddenness, and
SlotNames.

<a id="captured-outputs"></a>

## 12. Captured outputs

The following is captured expected output from actual CPU runs of the seven
real-model inference modes listed in the table below with the fixed sample
inputs, pinned assets, and settings described above. It is not output from
`--help`, a synthetic fixture, or a business-rule test. The two portable
facade/pipeline writer-reader modes are deliberately not presented as fresh
heavyweight Laya captures; their acceptance evidence uses tiny offline
fixtures and fresh processes. The model is the English FP32 Laya export at revision
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
not represent seven different model predictions, and they do not imply that
all eleven sample modes have independent real-model captures.

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

<a id="api-and-source-map"></a>

## 13. API and source map

Use this section after the tutorial when you need to connect a concept to the
public API or implementation. All typed-decision behavior remains in the
existing `MLNet.TextInference.Onnx` assembly:

- facade: `OnnxTypedDecisions`;
- stages: `PrepareDecisionInputs`, `ScoreOnnxDecisionModel`,
  `DecodeDecisions`;
- composition: `AppendOnnxTypedDecisions`;
- portable persistence: `OnnxTypedDecisionsTransformer.Save`/`Load` and
  `TypedDecisionPortableModel.SavePipeline`/`LoadPipeline`.

The native stage schema is ML.NET data, not a JSON envelope. Preparation adds
`Int64` vectors for `input_ids`, `attention_mask`, `marker_pos`, and `qtype`,
a `Bool` vector for `marker_mask`, and `Int32` scalars
`DecisionBatchSize`, `DecisionSequenceLength`, and `DecisionMarkerWidth`.
Scoring adds `Single` vectors for `logits` and already-softmaxed `act_probs`.
The five vectors are flat `VBuffer<T>` values in the `IDataView`; the scalar
dimensions tell the scorer how to wrap them into shaped ONNX tensors.

| Concern | Source |
|---|---|
| Public options, estimators, transformers | [`TypedDecisionOptions.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/TypedDecisionOptions.cs), [`TypedDecisionEstimators.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/TypedDecisionEstimators.cs) |
| Preparation and profile-specific layout | [`PrepareDecisionInputs.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/PrepareDecisionInputs.cs), [`LayaTokenizer.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/LayaTokenizer.cs) |
| ONNX binding and output validation | [`ScoreOnnxDecisionModel.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/ScoreOnnxDecisionModel.cs) |
| Choice/Score/Noul decoding | [`DecodeDecisions.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/DecodeDecisions.cs) |
| Bundle validation and external data | [`TypedDecisionBundle.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/TypedDecisionBundle.cs), [`AssetArchive.cs`](../../../src/MLNet.TextInference.Onnx/AssetArchive.cs) |
| Portable artifact and supported chains | [`TypedDecisionPortableModel.cs`](../../../src/MLNet.TextInference.Onnx/TypedDecisions/TypedDecisionPortableModel.cs) |
| Developer/acceptance process harness | [`PortableProcessHarness/Program.cs`](../PortableProcessHarness/Program.cs) |

If a local asset, schema, or output does not match the tutorial, start with
the troubleshooting checkpoint beside the relevant step before changing the
model or tokenizer independently.
