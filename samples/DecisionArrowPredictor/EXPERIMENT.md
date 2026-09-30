# Decision-to-Arrow experiment reference

Start with the [beginner walkthrough](README.md). This reference preserves the
full acquisition, reproducibility, split, training, evaluation and acceptance
details; its commands run from the repository root.

**Study status:** the completed real 5,574-row scalar export has been imported,
trained and evaluated. [Results and limitations](#real-scalar-study-results)
include all five arms/three curves; synthetic smoke remains a separate exercise.

An experimental, CPU-only **consumer**, not a decision provider. It trains
ordinary ML.NET 5.0.0 pipelines from persisted decision probabilities. No
reference to `MLNet.TextInference.Onnx`, local Laya session, tokenizer, ONNX
Runtime, GPU package, or cloud inference belongs in this process.

The existing typed-transform portable persistence format is **not** an
arbitrary ML.NET pipeline serializer. This sample uses `MLContext.Model.Save`
and `Load`, with a separate validated feature-contract/model receipt.

## Artifact kit and offline start

The beginner [walkthrough](README.md#2-try-the-offline-table-reading-route-first)
uses one producer-supplied local artifact root. It needs these package feeds,
their immutable receipts, and the exact package files listed in
[`decision-arrow.dependencies.json`](../../eng/experiments/decision-arrow.dependencies.json):

```text
<artifact root>
  feeds
    arrow-baseline
      receipt.json
      Apache.Arrow.<pinned version>.nupkg
      Apache.Arrow.Scalars.<pinned version>.nupkg
    decision-adapter
      <pinned adapter version>
        receipt.json
        DecisionInference.Arrow.<pinned version>.nupkg
        DecisionInference.Abstractions.<pinned version>.nupkg
```

Do not substitute a model-assets or corpus directory. Experimental packages
are not on public NuGet; obtain the package/receipt kit from the owning
experiment producer. Paths are inputs, not references into its checkout.
Checked-in synthetic fixtures require no corpus or model assets.

From the repository root, using PowerShell 7 and the .NET 10 SDK:

```powershell
.\samples\DecisionArrowPredictor\Start.ps1 -ArtifactRoot "<artifact root>"
```

`Start.ps1` verifies the kit and invokes the existing restore helper in
project-only locked mode, builds this sample, then runs its existing `smoke`.
It does not run inference or training. Standard public .NET dependencies may
be fetched on a first restore; model/corpus downloads are never implicit.
The focused solution/test restore below remains available for contributors.

The application separates beginner orientation from full syntax:

```powershell
dotnet samples\DecisionArrowPredictor\bin\Debug\net10.0\DecisionArrowPredictor.dll help
dotnet samples\DecisionArrowPredictor\bin\Debug\net10.0\DecisionArrowPredictor.dll help commands
```

## Explicit preparation

Use a local artifact directory outside the repository. Never commit the corpus,
states, models, or raw-message diagnostics, and never upload them as CI artifacts.
`help`, `smoke`, builds, and tests do not acquire data or model assets.

```powershell
$Project = "samples\DecisionArrowPredictor\DecisionArrowPredictor.csproj"
$Artifacts = "<local experiment artifact root>"
$Cli = "samples\DecisionArrowPredictor\bin\Debug\net10.0\DecisionArrowPredictor.dll"

# Use only an explicitly supplied experiment NuGet config and isolated cache.
# This verifies every local package byte/hash and clean producer source receipt.
# -InitializeLock is only for deliberate dependency changes; omit normally.
.\eng\RestoreDecisionArrowPredictor.ps1 -ArtifactRoot $Artifacts
dotnet build eng\DecisionArrowPredictor.slnx --no-restore -p:ImportDirectoryBuildTargets=false
dotnet $Cli help commands
dotnet $Cli smoke
dotnet $Cli smoke --precision FourDecimalPlaces
dotnet $Cli smoke --precision TwoDecimalPlaces

# This is the ONLY network acquisition command. Output must be new/empty.
dotnet $Cli prepare --download --out "$Artifacts\corpus\uci-sms-v1"

# Validate recorded acquisition evidence/hashes, parse, then inspect grouping.
dotnet $Cli prepare --input "$Artifacts\corpus\uci-sms-v1\SMSSpamCollection" `
  --acquisition "$Artifacts\corpus\uci-sms-v1\acquisition.json" `
  --out "$Artifacts\corpus\group-review-v1"

# Review sizes, chaining, mixed labels, source count and bundled license first.
# Supply the hash you actually reviewed, not a computed approval shortcut.
dotnet $Cli prepare --input "$Artifacts\corpus\uci-sms-v1\SMSSpamCollection" `
  --acquisition "$Artifacts\corpus\uci-sms-v1\acquisition.json" `
  --questions "samples\DecisionArrowPredictor\questions.v1.json" `
  --groups "$Artifacts\corpus\group-review-v1\group-diagnostics.json" `
  --reviewed-groups-sha256 <reviewed-diagnostics-sha256> `
  --out "$Artifacts\corpus\frozen-v1"
```

`ImportDirectoryBuildTargets=false` prevents unrelated workstation-ancestor
Arcade targets from leaking into these isolated builds. It does not disable
`samples\Directory.Build.props`: the new project removes both inherited CPU and
GPU runtime references **after** that props file evaluates. Shared build and
NuGet settings are unchanged. Verify the resolved `project.assets.json` and
runtime `.deps.json`, including with `-p:UseGpuRuntime=true`.

### Acquisition evidence and count

Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
DOI **10.24432/C5CC84**. Attribution: Almeida, T. & Hidalgo, J. (2011).
Introductory paper: Almeida, Hidalgo & Yamakami (2011),
DOI **10.1145/2034691.2034742**.

The official page links CC BY 4.0. The bundled original README also retains
copyright, attribution requests, warranty/risk and liability terms. Acquisition
stores both evidence files and their hashes, metadata, URL, archive checksum,
raw corpus checksum, time, and parser version **before preparation accepts the
corpus**. This records evidence, not permission to redistribute messages.

The pinned archive hash is
`1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3`;
the parsed raw file hash is
`7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`.
Strict first-tab/UTF-8 parsing of these bytes gives **5,574 rows: 4,827 ham,
747 spam**, agreeing with both UCI metadata and the bundled README. Copies
processed into 5,572 rows are not silently substituted. Empty records, missing
first tabs/messages and invalid/case-changed labels fail with original row IDs,
without printing SMS text.

Messages retain original case, spaces, tabs and punctuation. IDs are one-based
source record lines. Exported JSONL contains **only** `rowId` and `state`;
class labels, groups and partition assignments stay in `split.v1.json`.
`preparation.v1.json` is published last, with hashes of immutable artifacts.

### Grouping and frozen split

Comparison-only normalization uses invariant lower case and collapsed
whitespace. Templates replace `http://`, `https://`, or `www.` URLs through the
next whitespace and digit runs of at least four characters. For normalized
texts of at least 20 characters, character 5-gram **set** Jaccard >= 0.90 adds
edges. Short texts use exact/template equality only. Deterministic connected
components use source-ID order; the group ID is the smallest source row ID.
Original inference text is never replaced.

Before freezing, inspect largest groups, mixed-label groups and components
whose members are linked only transitively. Retain them; never split a component
to hit round row counts. The acquired corpus yields **5,102 groups**, largest
30, with no mixed-label or chained groups for these bytes/rules.

Seed **1729**, descending group size, SHA-256 of `seed:groupId` tie order, and a
greedy squared class-deficit objective target 60/20/20. Reject partitions
without both classes. These are grouped, class-balanced, **not temporal** splits;
the source explicitly says messages are not chronologically sorted.

| Partition | Rows | Ham | Spam | Groups |
|---|---:|---:|---:|---:|
| Train | 3,344 | 2,896 | 448 | 3,061 |
| Validation | 1,116 | 966 | 150 | 1,021 |
| Holdout | 1,114 | 965 | 149 | 1,020 |

`corpus.freeze.v1.json` commits safe attribution/count/hash receipts, not raw
data. Exact frozen questions were committed before scoring:
`66bea2992130a26eaeecafe2dacfb9b23b7ba98f`.

## Feature contract and study

The portable adapter is pinned to clean TypeSafe commit
`ab4eebf29b9fdca49ce13e3b4afac0ec16b22983`, version
`0.1.0-exp.decisions.1.gab4eebf29b9f`; Arrow core/scalars use the baseline version
`23.0.0-exp.decisions.1.g6aafc634d65c`. Exact package byte receipts are committed
in `eng\experiments\decision-arrow.dependencies.json`. Restore maps only the
designated experimental IDs to their immutable feeds, uses NuGet's approved
public v2 endpoint for everything else, and isolates the cache. The unsigned
exception is explicit and local: hashes identify bytes, **not author signatures**.
No global trust or root settings are changed. Compute is not a predictor dependency.

Default smoke uses three **synthetic** 257-row, five-question fixtures produced
from that clean source. Their original complete manifest/contract/Arrow bytes
are preserved; `eng\ImportDecisionArrowFixtures.ps1` verifies the immutable
producer receipt before importing them. These fixtures exercise short final
batches and all declared precisions, not real corpus inference.
`eng\experiments\decision-arrow.fixtures.json` independently pins their
manifest/contract/data byte hashes, sizes, counts and identities. Import paths
are derived from the supplied artifact root, not producer-machine absolute paths.

`questions.v1.json` predeclares two Binary semantic questions, a three-level
Score, a five-option Choice, and a separate Binary direct spam baseline.
The semantic vector is exactly **ten probabilities**, in declared order:
commercial solicitation, requested contact action, the three time-pressure
probabilities, and the five message-purpose probabilities. Do not add spam
probability, selected winners, confidence/entropy, an action head, text or
labels. Exclusive distribution coordinates are redundant, not ten independent
signals.

The producer in `typesafe-meai` exclusively owns Arrow schema, canonicalization,
feature fingerprints, completed manifests and public reader APIs. Integration
requires its pinned package receipt, independently pinned expected contract,
and a completed dataset, not self-guessed metadata. Import joins by `row_id`;
missing/extra/duplicate IDs and wrong source, questions, contract or partial
artifacts are fatal. Conversion is checked Arrow `float64` probabilities to
ML.NET `float32` **only at this boundary**; source Arrow values are unchanged.
Real producer modes `scalar-cpu` and `native-cpu` are accepted explicitly
(alongside the original `scalar`/`native` aliases); synthetic and unknown modes
remain rejected.

Five arms use frozen splits: training prevalence prior, stored direct spam
probability, text+SDCA logistic regression, semantic+the same learner, and
combined+the same learner. Text vocabulary and normalization are fitted only
on training. Learned arms use seed **1**, one CPU thread, at most **100**
iterations, and L2 `[0.0001, 0.001, 0.01]`; validation AUPRC selects a candidate,
with lower log loss as the tie-break. F1 and <=1% validation FPR thresholds,
models and their receipts are frozen **before holdout evaluation**.

Learning subsets target 100, 500 and full training rows. SHA-256
`seed1:groupId` ordering selects whole groups until the target and both classes
are supported; actual counts can exceed targets. All learned arms share the
same source IDs. The prior uses each matched subset's prevalence.

Evaluation reports tie-aware step average precision (AUPRC, not trapezoidal
interpolation), tie-aware ROC AUC, natural-log log loss, Brier score, confusion,
precision/recall and actual holdout FPR at the validation-selected thresholds.
Only log-loss calculation clips probabilities to `[1e-15, 1-1e-15]` at endpoints.
Saved holdout predictions are resampled in **1,000 seeded whole-group**
bootstrap draws; no additional model inference or threshold retuning is done.
Undefined class cases remain visible, along with paired differences from the
text arm at each arm's own frozen threshold.

Model receipts record exact source/contract/split/question identity, conversion,
projection, model bytes/hash, selected fit time, tuning/validation cost and warm
ML.NET batch head time per row. Head timing includes ML.NET transforms and
enumeration; it is **not** foundation-model inference time. Full semantic
extraction costs must be supplied from real producer measurements, with their
measured scopes, before interpreting operational cost.

## Limits and acceptance

Synthetic fixtures and small offline tests are the default; they are never
labelled real UCI results. Actual study training requires the completed real
export. There are no silent row exclusions, zero-filled feature failures,
hidden downloads, skipped prerequisite successes, live cloud calls, package
publication, automatic merges or upstream Arrow promotion.

Public-corpus pretraining contamination limits generalization claims. A
successful result demonstrates within-corpus engineering/predictive behavior,
not unseen deployment quality, calibration of intermediate questions, or
replacement of semantic extraction. Semantic/combined heads still need the
extractor for every new message. Negative quality/performance results are
valid; do not tune using holdout or revise frozen questions to disguise them.

## Completed-real-data commands

These commands are implemented but require a **completed matching real**
scalar/native export, its independently pinned expected contract/fingerprint,
and extraction timing provenance. A synthetic manifest is rejected by `import`,
`train`, and `evaluate`. Do not take the expectation from an arbitrary incoming
manifest: pin it from the producer's immutable real dataset receipt.

```powershell
$Dataset = "<completed real producer dataset directory>"
$Fingerprint = "<exact fingerprint from immutable real producer receipt>"
$ImportOptions = @(
  "--manifest", "$Dataset\manifest.json", "--contract", "$Dataset\contract.json",
  "--feature-fingerprint", $Fingerprint,
  "--preparation", "$Artifacts\corpus\frozen-v1\preparation.v1.json",
  "--split", "$Artifacts\corpus\frozen-v1\split.v1.json",
  "--states", "$Artifacts\corpus\frozen-v1\states.v1.jsonl",
  "--questions", "$Artifacts\corpus\frozen-v1\questions.v1.json"
)
dotnet $Cli import @ImportOptions --out "$Artifacts\predictors\import.v1.json"
# Coordinate a quiet CPU window with the model exporter before timing training.
dotnet $Cli train @ImportOptions --out "$Artifacts\predictors\models-v1"
# Record the training freeze hash BEFORE opening holdout evaluation.
$FrozenHash = (Get-FileHash "$Artifacts\predictors\models-v1\training.freeze.json" -Algorithm SHA256).Hash
dotnet $Cli evaluate @ImportOptions `
  --training-freeze "$Artifacts\predictors\models-v1\training.freeze.json" `
  --training-freeze-sha256 $FrozenHash --out "$Artifacts\predictors\evaluation-v1"
```

Evaluation rejects missing entire learning curves or missing/duplicate arms
before creating output. Each model receipt is published only after successful
saved-model replay with matching row count/IDs/groups/labels and finite
probabilities. Models, thresholds and all sidecars are immutable; a failed run
does not publish a completed training/evaluation receipt.

## Real scalar study results

This is the **completed real frozen UCI SMS study**, not the synthetic smoke.
The producer's immutable receipt passed byte/hash/source/identity checks and
its pinned-package public reader checked IDs 1..5,574 exactly once. The
consumer used clean Release source `a3f1b7e6bb12c4163d82f125e1a470ed72edc67f`.
It imported all rows, saved/reloaded nine learned models, froze all 15
arm/curve entries and their validation-selected thresholds **before** holdout
evaluation, then bootstrapped the saved holdout predictions. No holdout tuning,
question revision, row filtering or additional extraction was performed.

Exact aggregate metrics, all intervals, nine model byte/hash/validation/cost
receipts and source/import/pre-holdout/completion receipts are published in
[`study.scalar.v1.json`](study.scalar.v1.json). It contains **no messages,
per-row labels/predictions, training-ID lists, model weights or raw diagnostics**.
The hashes refer to immutable local artifacts, not publicly redistributed data.

### Rows and learning curves

Validation is 1,116 rows (150 spam / 966 ham; 1,021 groups). Holdout is
1,114 rows (149 spam / 965 ham; 1,020 groups). Every arm uses the same holdout;
every arm within a curve uses the same intact-group training subset.

| Target / actual training rows | Spam | Ham | Groups |
|---|---:|---:|---:|
| 100 / 100 | 19 | 81 | 96 |
| 500 / 500 | 67 | 433 | 471 |
| 3,344 / 3,344 | 448 | 2,896 | 3,061 |

### Ranking and probability losses

AUPRC below is tie-aware step average precision; loss uses natural logarithms.
Values are rounded for display; the JSON retains the original precision.
Higher AP/ROC and lower log loss/Brier are better.

| Training rows | Arm | AUPRC | ROC AUC | Log loss | Brier |
|---:|---|---:|---:|---:|---:|
| 100 | Prior | 0.13375 | 0.50000 | 0.40466 | 0.11903 |
| 100 | Direct stored spam score | 0.86585 | 0.96237 | 0.96879 | 0.32926 |
| 100 | Text | 0.88383 | 0.96304 | 0.16371 | 0.04730 |
| 100 | Semantic | 0.93499 | 0.98494 | 0.08481 | 0.02027 |
| 100 | Combined | 0.93882 | 0.98501 | 0.08657 | 0.02041 |
| 500 | Prior | 0.13375 | 0.50000 | 0.39346 | 0.11586 |
| 500 | Direct stored spam score | 0.86585 | 0.96237 | 0.96879 | 0.32926 |
| 500 | Text | 0.88500 | 0.96205 | 0.14355 | 0.03921 |
| 500 | Semantic | 0.95506 | 0.98734 | 0.07551 | 0.01836 |
| 500 | Combined | 0.96470 | 0.98994 | 0.06630 | 0.01674 |
| 3,344 | Prior | 0.13375 | 0.50000 | 0.39346 | 0.11586 |
| 3,344 | Direct stored spam score | 0.86585 | 0.96237 | 0.96879 | 0.32926 |
| 3,344 | Text | 0.94135 | 0.97756 | 0.11372 | 0.02896 |
| 3,344 | Semantic | 0.95476 | 0.98715 | 0.07791 | 0.01966 |
| 3,344 | Combined | 0.96503 | 0.99008 | 0.06415 | 0.01566 |

The direct score ranks messages usefully, but its raw probability losses are
**worse than the prevalence prior**. It is not a reliable calibrated spam
probability. Semantic/combined heads have promising within-corpus point
estimates, especially with 100/500 training rows; these are not deployment or
foundation-model-independent results.

### Confusion at frozen validation-F1 thresholds

`TP / FP / TN / FN` is the holdout confusion matrix. These thresholds were
chosen on validation, never on holdout. Exact threshold values are in the JSON.

| Training rows | Arm | TP / FP / TN / FN | Precision | Recall |
|---:|---|---|---:|---:|
| 100 | Prior | 149 / 965 / 0 / 0 | 0.13375 | 1.00000 |
| 100 | Direct | 119 / 26 / 939 / 30 | 0.82069 | 0.79866 |
| 100 | Text | 118 / 31 / 934 / 31 | 0.79195 | 0.79195 |
| 100 | Semantic | 130 / 8 / 957 / 19 | 0.94203 | 0.87248 |
| 100 | Combined | 132 / 10 / 955 / 17 | 0.92958 | 0.88591 |
| 500 | Prior | 149 / 965 / 0 / 0 | 0.13375 | 1.00000 |
| 500 | Direct | 119 / 26 / 939 / 30 | 0.82069 | 0.79866 |
| 500 | Text | 108 / 13 / 952 / 41 | 0.89256 | 0.72483 |
| 500 | Semantic | 137 / 13 / 952 / 12 | 0.91333 | 0.91946 |
| 500 | Combined | 134 / 10 / 955 / 15 | 0.93056 | 0.89933 |
| 3,344 | Prior | 149 / 965 / 0 / 0 | 0.13375 | 1.00000 |
| 3,344 | Direct | 119 / 26 / 939 / 30 | 0.82069 | 0.79866 |
| 3,344 | Text | 124 / 10 / 955 / 25 | 0.92537 | 0.83221 |
| 3,344 | Semantic | 130 / 9 / 956 / 19 | 0.93525 | 0.87248 |
| 3,344 | Combined | 133 / 8 / 957 / 16 | 0.94326 | 0.89262 |

### Validation-budget thresholds: actual holdout behavior

The selection rule maximizes validation recall subject to **<=1% validation
FPR**. The following numbers are **holdout** recall and actual holdout FPR at
that frozen threshold. A validation constraint is **not** a holdout guarantee;
bold values exceeded 1%. The prior predicts no positives at its budget
threshold, so its precision there is undefined (reported as `null`, not zero).

| Training rows | Arm | Holdout recall | Actual holdout FPR |
|---:|---|---:|---:|
| 100 | Prior | 0.00000 | 0.0000% |
| 100 | Direct | 0.48993 | 0.7254% |
| 100 | Text | 0.61074 | 0.3109% |
| 100 | Semantic | 0.88591 | **1.0363%** |
| 100 | Combined | 0.88591 | **1.0363%** |
| 500 | Prior | 0.00000 | 0.0000% |
| 500 | Direct | 0.48993 | 0.7254% |
| 500 | Text | 0.69799 | **1.2435%** |
| 500 | Semantic | 0.88591 | 0.7254% |
| 500 | Combined | 0.91946 | **1.2435%** |
| 3,344 | Prior | 0.00000 | 0.0000% |
| 3,344 | Direct | 0.48993 | 0.7254% |
| 3,344 | Text | 0.80537 | 0.4145% |
| 3,344 | Semantic | 0.89262 | 0.9326% |
| 3,344 | Combined | 0.89262 | 0.8290% |

### Uncertainty and comparison limits

These are 95% percentile intervals from **1,000 seed-1729 non-stratified
whole-group resamples of saved holdout predictions**. Paired comparisons use
the same resampled rows and each arm's own frozen threshold. All reported
metric/paired intervals had 1,000 defined and **zero undefined-class** cases.
This conditions on the fitted models and one grouped split: it does not
resample training, model selection or question design, and comparisons are
not adjusted for multiple arms/curves.

| Training rows | Arm | AP interval | Paired AP delta vs text interval |
|---:|---|---|---|
| 100 | Semantic | [0.86909, 0.97945] | [-0.02238, 0.11015] |
| 100 | Combined | [0.87556, 0.97998] | [-0.01427, 0.11121] |
| 500 | Semantic | [0.91209, 0.98262] | [0.02590, 0.12101] |
| 500 | Combined | [0.93063, 0.98713] | [0.04038, 0.12584] |
| 3,344 | Semantic | [0.90939, 0.98530] | [-0.03080, 0.05103] |
| 3,344 | Combined | [0.92980, 0.98910] | [-0.01019, 0.05565] |

**Full-training AP superiority over text is not established:** both paired
AP intervals include zero. Full-training probability-loss deltas favor the
learned semantic/combined heads within this fixed-prediction analysis:

| Full-training arm | Paired log-loss delta vs text | Paired Brier delta vs text |
|---|---|---|
| Semantic | [-0.05967, -0.00946] | [-0.01654, -0.00228] |
| Combined | [-0.07205, -0.02572] | [-0.01990, -0.00707] |

The public corpus may be in model pretraining. Together with the single
grouped non-chronological split and conditional bootstrap, that limits
generalization claims. Better in-corpus losses do not establish calibration
on unseen deployment data. The ten coordinates include redundant exclusive
distributions, not ten independent signals.

### Model and extraction costs

All nine learned models use the same learner/grid; below are the selected
L2, model bytes, selected fit time, grid-fit/validation time, and warm batch
head time per row. Head timing averages five passes over 1,116 validation
rows after a warmup and includes ML.NET transforms/enumeration, not Arrow I/O,
model load or Laya extraction. These are one quiet-lease operational
observations, not a sustained online/single-message benchmark.

| Rows | Arm | L2 | Model bytes | Selected fit ms | Grid/validation ms | Warm head ms/row |
|---:|---|---:|---:|---:|---:|---:|
| 100 | Text | 0.0001 | 71,379 | 754.181 | 1,121.007 | 0.035857 |
| 100 | Semantic | 0.001 | 3,212 | 10.930 | 108.948 | 0.002498 |
| 100 | Combined | 0.001 | 96,486 | 81.122 | 598.234 | 0.003897 |
| 500 | Text | 0.0001 | 196,373 | 180.071 | 267.101 | 0.004003 |
| 500 | Semantic | 0.0001 | 3,211 | 35.962 | 46.878 | 0.001234 |
| 500 | Combined | 0.0001 | 287,125 | 263.095 | 369.869 | 0.004534 |
| 3,344 | Text | 0.0001 | 725,337 | 188.054 | 355.680 | 0.004737 |
| 3,344 | Semantic | 0.0001 | 3,211 | 42.163 | 74.800 | 0.001001 |
| 3,344 | Combined | 0.0001 | 1,083,570 | 331.007 | 563.833 | 0.004312 |

Prior/direct arms do not fit ML.NET models; separate head timing was not
measured for them. The full `train` command took 9.716 s and `evaluate` plus
bootstrap took 9.186 s, including their import/orchestration work; do not
confuse those with selected fit time.

**Small head != cheap complete inference.** The producer's full five-question
scalar export took **147.47 minutes**, approximately **1.58743 s/message**
(0.62995 rows/s), with **2.01488 GiB** peak process working set. Model load
was a separate 2.508 s; cold first result was 1.907 s, first export result
after warmup 1.780 s. There were 5,574 export ONNX calls plus one warmup call.
The export scope includes preparation/scoring/decoding/Arrow append/flush/hash,
but excludes preflight/model load/final manifest serialization. Overlapping
first-result and inference-and-append measurements are **not summed**.

Features were extracted once and reused by these learning curves. For a new
message, semantic/combined heads still require extraction; text does not.
The stored direct baseline also came from this full five-question export;
no direct-only extraction benchmark was measured. No full-corpus native
speedup/parity claim follows from this scalar study.

### Immutable receipt chain

All paths below are relative to the local shared experiment artifact root.
The source/data receipt was independently pinned before import; training
freeze hash was persisted before invoking holdout evaluation. The complete
consumer receipt pins all 38 local artifacts, including nine model byte
hashes and saved prediction hashes. CPU was returned explicitly before the
producer's next native16 stage. Only aggregate inspection/docs followed.
The public aggregate envelope is byte-preserved by the sample attributes;
its SHA-256 is
`ca3997631bc25b46d788725b9b1f885784d46f1a1c7573f8f6aefd21790bfc53`.

| Artifact | SHA-256 |
|---|---|
| `datasets\laya\scalar-corpus-915b85b.receipt.json` | `829dea0cccfeef32015798896fa51037d7586e3f2585559315653be98f8c2e58` |
| Producer `manifest.json` | `76748fdf24a0a33efb1a6509e00d25fbc8caddd3b002005cb1bf77e2911c8c2c` |
| Producer `contract.json` / feature identity | `72acebb6036bfa8fdaa1d47a1490f5ea1bfc6e08fe71e7e84e8c8922ec4de514` |
| Producer `decisions.arrow` | `fae87f65effbb7cac23f6e9de8af382dd6dc1b145bf843c40001947f159dff46` |
| `predictors\real-scalar-v1\models\training.freeze.json` | `6ebea3a135c808b31a996f5e31835f7608c78f2c22eaa03d6a3a760a1c5bcd1c` |
| `predictors\real-scalar-v1\evaluation\evaluation.json` | `780e07e259ee7158390e78af82ea2c7599f1ded45fa3bf764dd4118591b9b753` |
| `predictors\real-scalar-v1\study.complete.receipt.json` | `3a253ccac138e6db362b2ae1172626d4bfeff9d4c89bf515b0131782bc936b21` |

Portable adapter source remains `ab4eebf29b9fdca49ce13e3b4afac0ec16b22983`;
actual extraction runtime source is
`915b85b8dbe123e465523120b0cc2c4c7d57f7b6`. Dataset provenance records its
actual runtime package closure and CPU4/inter-op1 configuration; it is not
the consumer's portable no-ONNX closure.

## Offline validation evidence

The integrated offline gate before the CPU-mode addition had **369 passing
cases, no skips** (236 initial core cases plus 133 focused cases). The real
handoff added four exact-mode cases; its focused importer gate passed **13/13,
zero skips**, after rebuilding the changed sample. The full suite was not
rerun during the producer's subsequent native timing window. Targeted commands:

```powershell
dotnet test tests\MLNet.DecisionArrowPredictor.Tests\MLNet.DecisionArrowPredictor.Tests.csproj --no-restore -p:ImportDirectoryBuildTargets=false
.\eng\VerifyDecisionArrowPredictorClosure.ps1 `
  -AssetsFile samples\DecisionArrowPredictor\obj\project.assets.json `
  -DepsFile samples\DecisionArrowPredictor\bin\Debug\net10.0\DecisionArrowPredictor.deps.json
```

| Requirement | Exact test evidence |
|---|---|
| "Parse first tab preserving original message" | `Parse_FirstTabPreservesTextIdsAndLabels` |
| "reject malformed lines/labels with row diagnostics" | `Parse_MalformedRowReportsOnlyRowAndRule` |
| "mixed-label groups retained/reported" | `Build_TransitiveChainHasMinimumIdOrderedMembershipAndMixedLabels` |
| "char 5-gram set Jaccard >=0.90" | `Jaccard_InclusivePointNineEdgeAndBelowThreshold`, `Build_FuzzyEdgesRequireBothLengthsAtLeastTwenty` |
| "Grouped 60/20/20 target with class balancing and split seed 1729" | `Create_BalancedMixedGroupsProduceDeterministicThirtySixTwelveTwelveRows` |
| "Semantic vector is 10 numeric probabilities" | `ConvertProbabilities_CastsInclusiveEndpointsWithoutMutatingInput`, `Fit_SemanticArmTrainsOfflineWithExactlyTenFeatures` |
| "Fit text vocabulary/normalization/learner on training ONLY" | `Fit_TextAndCombinedVocabularyExcludesValidationOnlyToken` |
| "matched rows across learned arms" | `Subset_Targets100500FullPreserveNestedIntactOriginalGroups` |
| "recall under1% VALIDATION FPR" | `SelectThreshold_BudgetIncludesOnePercentAndNeverSplitsTies` |
| "1000 seeded group-bootstrap resamples" | `Bootstrap_GroupedDrawsMatchIndependentSeededCountsAndWeightedBounds` |
| "report undefined class cases" | `Bootstrap_SingleClassUndefinedMetricsHaveNullBounds` |
| "Save/load ordinary ML.NET trained model with external feature-contract validation" | `Predict_AndRealLoadPreserveIdentitiesAndReplayProbabilities`, `Load_ContractAndReceiptMutationsAreRejected` |
| "reject missing/extra/duplicate IDs or partial artifacts" | `PublicReader_ExactExternalSourceIdsRejectExtraAndMissingRows`, `PublicReader_HashConsistentIpcStillRejectsDuplicateSourceIds`, `SmokeAsync_PublicReaderRejectsIncompleteOrChangedFixture` |
| "No synthetic results labelled real" | `ImportAsync_RealImportRejectsOfficialSyntheticFixtureEvenWhenSourceHashesAndIdsMatch` |
| Five arms and three declared curves; no empty completed reports | `Train_FiveArmsUseMatchedIntactTrainingOnlySubsetsAndPublishFreeze`, `Evaluate_ExactFiveArmsAndDeclaredTargetsAreRequired` |
| Completion receipt only after saved-model verification | `Fit_FailedSavedModelVerificationDoesNotFinalizeReceipt` |

The complete bounded research/requirement mapping and assertion/gap review are
local ignored `.testagent` artifacts, not corpus reports. Real five-arm results
are **pending the completed real export and separate timing window**, not
inferred from any offline fixture.
