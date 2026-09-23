# Standalone typed decisions

Build and run with a local bundle:

```powershell
dotnet run --project samples/TypedDecisions/Standalone -- --mode facade --bundle C:\models\laya-english-fp32.bundle
dotnet run --project samples/TypedDecisions/Standalone -- --mode stages --bundle C:\models\laya-english-fp32.bundle
```

`facade` calls `OnnxTypedDecisions.Infer` directly. `stages` calls
`PrepareDecisionInputs`, `ScoreOnnxDecisionModel`, and `DecodeDecisions`
separately and prints the JSON envelope after preparation and scoring. State is
always text; callers that use structured state must serialize it explicitly.
