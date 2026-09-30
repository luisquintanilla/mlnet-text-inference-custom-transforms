using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace DecisionArrowPredictor;

public static class ArtifactFiles
{
    public static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web) { WriteIndented = true };
    public static readonly UTF8Encoding Utf8 = new(false, true);

    public static string Hash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    public static void RequireHash(string path, string expected)
    {
        if (expected.Length != 64 || !expected.All(Uri.IsHexDigit) ||
            !string.Equals(Hash(path), expected, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"SHA-256 mismatch: {path}");
    }

    public static T Read<T>(string path) =>
        JsonSerializer.Deserialize<T>(File.ReadAllText(path, Utf8), Json)
        ?? throw new InvalidDataException($"Null JSON document: {path}");

    public static void Write<T>(string path, T value) =>
        WriteBytes(path, JsonSerializer.SerializeToUtf8Bytes(value, Json));

    public static void WriteBytes(string path, byte[] bytes)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using (var stream = new FileStream(path + ".partial", FileMode.CreateNew, FileAccess.Write))
        {
            stream.Write(bytes);
            stream.Flush(true);
        }
        File.Move(path + ".partial", path, overwrite: false);
    }
}
