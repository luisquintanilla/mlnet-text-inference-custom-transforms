# Decision-to-Arrow predictor study

An experimental, CPU-only **consumer**, not a decision provider. It trains
ordinary ML.NET 5.0.0 pipelines from persisted decision probabilities. No
reference to `MLNet.TextInference.Onnx`, local Laya session, tokenizer, ONNX
Runtime, GPU package, or cloud inference belongs in this process.

The existing typed-transform portable persistence format is **not** an
arbitrary ML.NET pipeline serializer. This sample uses `MLContext.Model.Save`
and `Load`, with a separate validated feature-contract/model receipt.

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
dotnet $Cli help
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

## Offline validation evidence

The current offline suite has **369 passing cases, no skips** (236 initial
core cases plus 133 focused study/import/persistence cases). Targeted commands:

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
