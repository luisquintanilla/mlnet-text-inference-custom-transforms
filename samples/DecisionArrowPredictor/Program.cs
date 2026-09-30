using System.Text.Json;
using DecisionArrowPredictor;

try
{
    if (args.Length == 0 || args is ["help"] or ["--help"])
    {
        Console.WriteLine("""
            DecisionArrowPredictor: learn a small spam classifier from numeric answers to fixed questions.
            Laya in TypeSafe produces the probabilities; ML.NET reads the saved Arrow table here.

            Beginner walkthrough: samples\DecisionArrowPredictor\README.md
            Recommended first run, from the repository root:
              .\samples\DecisionArrowPredictor\Start.ps1 -ArtifactRoot <local artifact-kit folder>
            Already built? Run: smoke
            Smoke reads a synthetic fixture; it does not run Laya, train a model or measure accuracy.

            Advanced experiment syntax: help commands
            Real-data workflow and prerequisites: samples\DecisionArrowPredictor\EXPERIMENT.md
            """);
        return 0;
    }
    if (args is ["help", "commands"])
    {
        Console.WriteLine("""
            DecisionArrowPredictor advanced experiment commands (see EXPERIMENT.md)
              smoke [--precision HighPrecision|FourDecimalPlaces|TwoDecimalPlaces]
              prepare --download --out <new acquisition directory>
              prepare --input <corpus> --acquisition <receipt> --out <review directory>
              prepare --input <corpus> --acquisition <receipt> --out <freeze directory>
                      --questions <questions.v1.json> --groups <group-diagnostics.json>
                      --reviewed-groups-sha256 <sha256>
              import --manifest <completed real manifest> --contract <pinned contract.json>
                     --feature-fingerprint <independently pinned identity>
                     --preparation <preparation.v1.json> --split <split.v1.json>
                     --states <states.v1.jsonl> --questions <questions.v1.json> --out <receipt>
              train  <same import options> --out <new model directory>
              evaluate <same import options> --training-freeze <training.freeze.json>
                       --training-freeze-sha256 <pinned hash> --out <new result directory>
            The first prepare downloads only the explicitly approved public UCI corpus.
            Review grouping and bundled license/count evidence before the final freeze command.
            """);
        return 0;
    }
    if (args[0] is not ("prepare" or "import" or "train" or "evaluate" or "smoke"))
        throw new ArgumentException($"Unsupported command: {args[0]}.");
    var options = new Dictionary<string, string>(StringComparer.Ordinal);
    for (int i = 1; i < args.Length; i++)
    {
        string key = args[i];
        if (key == "--download") options.Add(key, "true");
        else
        {
            if (!key.StartsWith("--", StringComparison.Ordinal) || ++i == args.Length)
                throw new ArgumentException("Expected named option and value.");
            options.Add(key, args[i]);
        }
    }
    string Required(string name) => options.TryGetValue(name, out var value) ? value :
        throw new ArgumentException($"Required option: {name}.");
    string[] allowed = args[0] switch
    {
        "prepare" => ["--download", "--out", "--input", "--acquisition", "--questions", "--groups", "--reviewed-groups-sha256"],
        "smoke" => ["--precision"],
        "evaluate" => ["--manifest", "--contract", "--feature-fingerprint", "--preparation", "--split", "--states", "--questions",
            "--out", "--training-freeze", "--training-freeze-sha256"],
        _ => ["--manifest", "--contract", "--feature-fingerprint", "--preparation", "--split", "--states", "--questions", "--out"]
    };
    if (options.Keys.Any(k => !allowed.Contains(k, StringComparer.Ordinal)))
        throw new ArgumentException($"Unknown {args[0]} option.");
    if (args[0] == "smoke")
    {
        string precision = options.GetValueOrDefault("--precision", "HighPrecision");
        string fingerprint = precision switch
        {
            "HighPrecision" => "06654530d526e33db86ca9f59e937d79dda9d83f1447d675edb69856a9017513",
            "FourDecimalPlaces" => "66e0d62923ab097c24016eb711bd65d3d07a5f3a77fef8bec3b95e735fc8a2c1",
            "TwoDecimalPlaces" => "b717b6c1c8250773be6d5594395405a81c64bae64d416d1c029a5e8ec3ed41e4",
            _ => throw new ArgumentException("Unsupported fixture precision.")
        };
        string fixture = Path.Combine(AppContext.BaseDirectory, "fixtures", precision);
        int rows = await ArrowFeatureReader.SmokeAsync(Path.Combine(fixture, "manifest.json"),
            Path.Combine(fixture, "contract.json"), fingerprint, Path.Combine(AppContext.BaseDirectory, "questions.v1.json"));
        Console.WriteLine($"Synthetic-only Arrow smoke: {rows} validated rows, {precision}; no inference or downloads.");
        return 0;
    }
    string output = Required("--out");
    if (args[0] is "import" or "train" or "evaluate")
    {
        var study = await ArrowFeatureReader.ImportAsync(Required("--manifest"), Required("--contract"),
            Required("--feature-fingerprint"), Required("--preparation"), Required("--split"), Required("--states"),
            Required("--questions"));
        switch (args[0])
        {
            case "import":
                ArtifactFiles.Write(output, ArrowFeatureReader.Receipt(study));
                Console.WriteLine($"Imported {study.Rows.Length} real rows; complete matching contract and source IDs.");
                break;
            case "train":
                var freeze = StudyWorkflow.Train(study, output);
                Console.WriteLine($"Frozen {freeze.Arms.Length} arms/curves; no holdout evaluation. SHA-256=" +
                    ArtifactFiles.Hash(Path.Combine(output, "training.freeze.json")));
                break;
            case "evaluate":
                string trainingFreeze = Required("--training-freeze");
                ArtifactFiles.RequireHash(trainingFreeze, Required("--training-freeze-sha256"));
                var evaluation = StudyWorkflow.Evaluate(study, trainingFreeze, output);
                Console.WriteLine($"Evaluated {evaluation.Arms.Length} frozen arms/curves; saved predictions and bootstrap report. SHA-256=" +
                    ArtifactFiles.Hash(Path.Combine(output, "evaluation.json")));
                break;
        }
        return 0;
    }
    if (options.ContainsKey("--download"))
    {
        if (options.Count != 2) throw new ArgumentException("Download accepts only --download and --out.");
        var receipt = await CorpusPreparation.AcquireAsync(output);
        Console.WriteLine(JsonSerializer.Serialize(receipt, ArtifactFiles.Json));
    }
    else if (options.ContainsKey("--reviewed-groups-sha256"))
        Console.WriteLine(JsonSerializer.Serialize(PreparationWorkflow.Freeze(Required("--acquisition"), Required("--input"),
            Required("--questions"), Required("--groups"), Required("--reviewed-groups-sha256"), output), ArtifactFiles.Json));
    else
    {
        if (options.Count != 3) throw new ArgumentException("Review accepts only --input, --acquisition, --out.");
        var groups = PreparationWorkflow.Review(Required("--acquisition"), Required("--input"), output);
        Console.WriteLine($"Review only: rows={groups.Rows}, groups={groups.Groups}, largest={groups.LargestGroup}, " +
            $"mixed={groups.MixedLabelGroups}, chained={groups.ChainedGroups}; SHA-256=" +
            ArtifactFiles.Hash(Path.Combine(output, "group-diagnostics.json")));
    }
    return 0;
}
catch (Exception error) when (error is ArgumentException or InvalidDataException or IOException or HttpRequestException or JsonException)
{
    Console.Error.WriteLine($"{error.GetType().Name}: {error.Message}");
    return 1;
}
