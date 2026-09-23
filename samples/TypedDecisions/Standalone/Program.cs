using System.Text.Json;
using MLNet.TextInference.TypedDecisions;

var mode = GetOption(args, "--mode") ?? "facade";
var bundlePath = GetOption(args, "--bundle");
if (string.IsNullOrWhiteSpace(bundlePath))
{
    Console.Error.WriteLine("A local --bundle path is required. Inference never downloads model assets.");
    return 2;
}

var questions = new[]
{
    DecisionQuestion.Choice(
        "priority",
        "How urgent is the request?",
        new[] { "low", "high" }),
    DecisionQuestion.Score(
        "quality",
        "How strong is the evidence?",
        new[] { "weak", "moderate", "strong" }),
    DecisionQuestion.Noul(
        "actionable",
        "Can the request be acted on now?")
};
const string state = "The customer supplied reproducible steps and requested an urgent fix.";

using var bundle = TypedDecisionBundle.Open(bundlePath);
var request = DecisionRequest.Create(state, questions);

switch (mode.ToLowerInvariant())
{
    case "facade":
        using (var facade = new OnnxTypedDecisions(bundle))
            PrintJson(DecisionJsonCodec.SerializeResponse(facade.Infer(state, questions)));
        break;

    case "stages":
        var preparer = new PrepareDecisionInputs(bundle.Profile, bundle.Tokenizer);
        var inputs = preparer.Prepare([request]);
        PrintLabeledJson("prepared", DecisionJsonCodec.SerializeInputs(inputs));

        using (var scorer = new ScoreOnnxDecisionModel(bundle))
        {
            var outputs = scorer.Score(inputs);
            PrintLabeledJson("scored", DecisionJsonCodec.SerializeScored(inputs, outputs));
            var decoder = new DecodeDecisions(
                bundle.Profile.TemperaturePolicy, bundle.Manifest.Decoder);
            PrintLabeledJson("decoded", DecisionJsonCodec.SerializeResponse(
                decoder.Decode(inputs, outputs)));
        }
        break;

    default:
        Console.Error.WriteLine($"Unknown mode '{mode}'. Use facade or stages.");
        return 2;
}

return 0;

static string? GetOption(string[] args, string name)
{
    var index = Array.IndexOf(args, name);
    return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
}

static void PrintJson(string json) => Console.WriteLine(json);

static void PrintLabeledJson(string label, string json)
{
    Console.WriteLine($"[{label}]");
    using var document = JsonDocument.Parse(json);
    Console.WriteLine(JsonSerializer.Serialize(document, new JsonSerializerOptions
    {
        WriteIndented = true
    }));
}
