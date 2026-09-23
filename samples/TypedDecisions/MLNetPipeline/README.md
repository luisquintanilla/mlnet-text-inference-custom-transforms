# ML.NET typed decisions

This sample is a .NET 10 file-based app. Run it with a local bundle:

```powershell
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- --mode facade --bundle C:\models\laya-english-fp32.bundle
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- --mode stages --bundle C:\models\laya-english-fp32.bundle
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- --mode composed --bundle C:\models\laya-english-fp32.bundle
dotnet run --file samples/TypedDecisions/MLNetPipeline/Program.cs -- --help
```

The facade estimator is schema-aware and lazy. Its cursor collects rows up to
`BatchSize`, performs one model call for that batch, and caches pass-through
columns so downstream enumeration does not re-enumerate the source. The stages
are composable with ordinary ML.NET `Append`; preparation and scoring expose
JSON envelopes for intermediate inspection, and decoding adds typed scalar
columns plus the complete JSON result.

The compiled facade can also be appended to an existing estimator chain with
`AppendOnnxTypedDecisions(mlContext, options)`. ML.NET model serialization is
deliberately not implemented because the bundle remains an explicit local
runtime dependency.

Because file-based apps materialize a temporary project outside the repository,
their first run needs NuGet access to resolve project/package dependencies and
the platform runtime packs. `--help` still performs the same restore/build
path; a missing feed or unavailable runtime pack fails visibly before execution.
