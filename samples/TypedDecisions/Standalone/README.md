# Standalone typed decisions

This sample is a .NET 10 file-based app. Run it with a local bundle:

```powershell
dotnet run --file samples/TypedDecisions/Standalone/Program.cs -- --mode facade --bundle C:\models\laya-english-fp32.bundle
dotnet run --file samples/TypedDecisions/Standalone/Program.cs -- --mode stages --bundle C:\models\laya-english-fp32.bundle
dotnet run --file samples/TypedDecisions/Standalone/Program.cs -- --help
```

`facade` calls `OnnxTypedDecisions.Infer` directly. `stages` calls
`PrepareDecisionInputs`, `ScoreOnnxDecisionModel`, and `DecodeDecisions`
separately and prints the JSON envelope after preparation and scoring. State is
always text; callers that use structured state must serialize it explicitly.
The ML.NET sample is the primary pipeline integration; this app is a small
direct-core example and a useful intermediate-contract diagnostic.

Because file-based apps materialize a temporary project outside the repository,
their first run needs the configured NuGet feeds to resolve project/package
dependencies. `--help` still performs the same restore/build path; a missing
feed fails visibly before execution.
