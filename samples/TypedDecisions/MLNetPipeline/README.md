# ML.NET typed decisions

Build and run with a local bundle:

```powershell
dotnet run --project samples/TypedDecisions/MLNetPipeline -- --mode facade --bundle C:\models\laya-english-fp32.bundle
dotnet run --project samples/TypedDecisions/MLNetPipeline -- --mode stages --bundle C:\models\laya-english-fp32.bundle
dotnet run --project samples/TypedDecisions/MLNetPipeline -- --mode composed --bundle C:\models\laya-english-fp32.bundle
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
