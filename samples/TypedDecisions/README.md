# Typed decisions with ML.NET

This sample teaches one practical idea: an application supplies text, asks
fixed questions, and supplies the alternatives that a model may score. The
model returns typed decisions; it does not generate prose, execute actions,
or replace application policy.

## What you will build

The recommended path is the
[guided ML.NET tutorial](MLNetPipeline/README.md). You will run one facade,
read a few scalar results, then learn how the same request becomes tokenizer
inputs, ONNX outputs, decoded distributions, and ML.NET columns.

For the pinned CPU capture, the two sample states produce:

| State | Priority | Quality score | Actionable | `P(true)` |
|---|---|---:|---|---:|
| Reproducible steps, urgent fix | `high` | `1.1856464` | `true` | `0.8520638` |
| Missing logs, no clear action | `low` | `0.61998236` | `false` | `0.10274245` |

`Choice` selects a caller-supplied label, `Score` returns an expected
zero-based option index (not a percentage), and `Noul` is this API's name for
a Boolean question. The full distributions, exact wording, and provenance
are in the tutorial's [captured outputs](MLNetPipeline/README.md#captured-outputs).

## Before you run

Use a checkout of this repository and install .NET 10. The sample references
the local `src/MLNet.TextInference.Onnx/MLNet.TextInference.Onnx.csproj`, so
it does not assume a published NuGet version already contains these PR APIs.
You need the five pinned English FP32 Laya files in this layout:

```text
models/laya-english-fp32/
  laya.onnx
  laya.onnx.data
  laya_config.json
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
```

The graph's external-data sidecar must remain beside `laya.onnx`, and
inference never downloads missing files. The tutorial's
[asset and acquisition steps](MLNetPipeline/README.md#2-before-you-run-the-sample)
link the exact pinned source. Ordinary tests use a small offline fixture; they
do not download the roughly 1.7 GB Laya weights.

## Start here

Run the tutorial's [recommended facade command](MLNetPipeline/README.md#3-run-one-recommended-path)
from the repository root. Then follow these links in order:

1. [Problem and result](MLNetPipeline/README.md#1-the-problem-and-the-result) —
   see the questions and expected scalar outcome.
2. [Vocabulary](MLNetPipeline/README.md#4-vocabulary-before-implementation) —
   learn `IDataView`, `Fit`, `Transform`, BPE, tokens, and tensors in context.
3. [Complete pipeline](MLNetPipeline/README.md#5-a-complete-followable-mlnet-pipeline) —
   copy a small file-based app and print typed values.
4. [Request walkthrough](MLNetPipeline/README.md#6-walk-one-request-through-the-implementation) —
   follow preparation, the five graph inputs, scoring, and decoding.
5. [Execution choices](MLNetPipeline/README.md#8-choose-an-execution-mode) —
   compare direct inference, stages, `PredictionEngine`, and composition.
6. [Portable deployment](MLNetPipeline/README.md#portable-saveload-for-deployment) —
   save/load a facade or supported flat pipeline without `Fit` on load.

Use the tutorial's
[mode table](MLNetPipeline/README.md#8-choose-an-execution-mode) when you
already know which boundary you need:

| Need | Start with |
|---|---|
| Call the fitted transformer directly, including a batch | `Infer` / `direct` |
| Use schema-aware lazy ML.NET processing | `facade` |
| Inspect tensors and native stage columns | `stages` |
| Map one row into a DTO | `prediction-engine` |
| Append another typed-decision facade | `composed` |
| Package a fitted facade or supported pipeline | portable writer/reader modes |

The tracked
[`PortableProcessHarness`](PortableProcessHarness/Program.cs) is a
developer/acceptance harness for deterministic offline fixtures and fresh
processes. It is not the beginner path or a replacement for the real-model
capture.

## Where to go next

- Read the tutorial's [API and source map](MLNetPipeline/README.md#api-and-source-map)
  when you need the public names or implementation files.
- Read [captured outputs](MLNetPipeline/README.md#captured-outputs) when you
  need exact JSON, tensor dimensions, token IDs, or SVG diagrams.
- Keep the [portable boundary](MLNetPipeline/README.md#portable-saveload-for-deployment)
  in mind: native `MLContext.Model.Save`/`Load`, arbitrary mixed chains, and
  automatic whole-pipeline ONNX export remain unsupported.
