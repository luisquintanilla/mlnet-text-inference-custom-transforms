using System.Text.Json;

namespace DecisionArrowPredictor;

public static class FeatureContract
{
    public const int Width = 10;
    public const string QuestionsV1Sha256 = "9e6e6c5c580934309cb1f9b4572c7b87a93ce22b32b50cb8b3af20caa20068ea";
    public const string Conversion = "Arrow float64 probabilities -> ML.NET float32, checked finite [0,1], original Arrow unchanged";
    public static readonly string[] Projection =
    [
        "commercial_solicitation", "requested_contact_action",
        "time_pressure[0]", "time_pressure[1]", "time_pressure[2]",
        "message_purpose[0]", "message_purpose[1]", "message_purpose[2]", "message_purpose[3]", "message_purpose[4]"
    ];

    public static void ValidateQuestions(byte[] bytes)
    {
        using var doc = JsonDocument.Parse(bytes);
        var root = doc.RootElement;
        string[] ids = ["commercial_solicitation", "requested_contact_action", "time_pressure", "message_purpose", "spam_baseline"];
        string[] kinds = ["binary", "binary", "score", "choice", "binary"];
        var questions = root.GetProperty("questions").EnumerateArray().ToArray();
        if (root.GetProperty("schemaVersion").GetInt32() != 1 || questions.Length != 5 ||
            !root.GetProperty("featureProjection").EnumerateArray().Select(p => p.GetString()).SequenceEqual(Projection))
            throw new InvalidDataException("Expected frozen v1 ten-coordinate semantic projection.");
        for (int i = 0; i < questions.Length; i++)
            if (questions[i].GetProperty("id").GetString() != ids[i] ||
                questions[i].GetProperty("kind").GetString() != kinds[i])
                throw new InvalidDataException("Question identity/order/kind mismatch.");
        if (!questions[2].GetProperty("rubric").EnumerateArray().Select(x => x.GetString())
            .SequenceEqual(new[] { "none", "mild", "explicit urgent deadline" }) ||
            !questions[3].GetProperty("candidates").EnumerateArray().Select(x => x.GetProperty("id").GetString())
            .SequenceEqual(new[] { "personal", "service_notice", "promotion", "financial_offer", "other" }))
            throw new InvalidDataException("Question candidate/rubric order mismatch.");
    }

    public static float[] ConvertProbabilities(IReadOnlyList<double> values)
    {
        if (values.Count != Width || values.Any(v => !double.IsFinite(v) || v < 0 || v > 1))
            throw new InvalidDataException("Expected ten finite probabilities in [0,1].");
        return values.Select(v => (float)v).ToArray();
    }

    public static void RequireIdentity(string actual, string expected)
    {
        if (string.IsNullOrWhiteSpace(expected) || !string.Equals(actual, expected, StringComparison.Ordinal))
            throw new InvalidDataException("External producer feature-contract identity mismatch.");
    }
}
