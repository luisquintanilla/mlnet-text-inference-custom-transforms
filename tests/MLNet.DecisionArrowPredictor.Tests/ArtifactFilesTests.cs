using System.Text;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class ArtifactFilesTests
{
    [TestMethod]
    [DataRow("", ArtifactExpectations.EmptyHash)]
    [DataRow("abc", ArtifactExpectations.AbcHash)]
    public void Hash_KnownBytesMatchLowercaseSha256Goldens(string text, string golden)
    {
        using var temp = new TempDirectory();
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        string path = temp.Put("hash.bin", bytes);

        string actual = ArtifactFiles.Hash(path);

        Assert.AreEqual(golden, actual);
        Assert.AreEqual(64, actual.Length);
        Assert.AreEqual(actual.ToLowerInvariant(), actual);
        ArtifactExpectations.Bytes(bytes, path);
    }

    [TestMethod]
    [DataRow("lower")]
    [DataRow("upper")]
    [DataRow("short")]
    [DataRow("long")]
    [DataRow("nonhex")]
    [DataRow("mismatch")]
    public void RequireHash_ValidatesFormatCaseAndMismatch(string caseId)
    {
        using var temp = new TempDirectory();
        byte[] bytes = Encoding.UTF8.GetBytes("private-content-marker");
        string path = temp.Put("pinned.bin", bytes);
        string correct = ArtifactExpectations.Hash(bytes);
        string expected = caseId switch
        {
            "lower" => correct,
            "upper" => correct.ToUpperInvariant(),
            "short" => correct[..63],
            "long" => correct + "0",
            "nonhex" => "g" + correct[1..],
            "mismatch" => new string('0', 64),
            _ => throw new ArgumentOutOfRangeException(nameof(caseId))
        };

        if (caseId is "lower" or "upper")
        {
            ArtifactFiles.RequireHash(path, expected);
            Assert.AreEqual(correct, ArtifactFiles.Hash(path));
        }
        else
        {
            var error = Assert.ThrowsExactly<InvalidDataException>(() => ArtifactFiles.RequireHash(path, expected));
            Assert.AreEqual("SHA-256 mismatch: " + path, error.Message);
            Assert.IsFalse(error.Message.Contains("private-content-marker", StringComparison.Ordinal));
        }
        ArtifactExpectations.Bytes(bytes, path);
    }

    [TestMethod]
    public void Read_DeserializesCamelCaseAndRejectsNullOrMalformedJson()
    {
        using var temp = new TempDirectory();
        string json = """{"count":2,"text":"  Keep\tinner tab\n\u03A9  ","ids":[7,42],"labels":[false,true]}""";
        string path = temp.PutText("authored.json", json);

        var result = ArtifactFiles.Read<SerializationFixture>(path);

        Assert.AreEqual(2, result.Count);
        Assert.AreEqual("  Keep\tinner tab\nΩ  ", result.Text);
        CollectionAssert.AreEqual(new long[] { 7, 42 }, result.Ids);
        CollectionAssert.AreEqual(new[] { false, true }, result.Labels);
        Assert.AreEqual(json, File.ReadAllText(path));
        string nullPath = temp.PutText("null.json", "null");
        var error = Assert.ThrowsExactly<InvalidDataException>(() => ArtifactFiles.Read<SerializationFixture>(nullPath));
        Assert.AreEqual("Null JSON document: " + nullPath, error.Message);
        string malformed = temp.PutText("malformed.json", "{\"count\":");
        Assert.ThrowsExactly<JsonException>(() => ArtifactFiles.Read<SerializationFixture>(malformed));
        Assert.AreEqual("{\"count\":", File.ReadAllText(malformed));
    }

    [TestMethod]
    public void Write_UsesIndentedCamelCaseStrictUtf8WithoutBom()
    {
        using var temp = new TempDirectory();
        string path = temp.FilePath("serialized.json");
        var value = new SerializationFixture(2, "A\t\"quoted\"\nΩ", [7, 42], [false, true]);

        ArtifactFiles.Write(path, value);

        byte[] bytes = File.ReadAllBytes(path);
        Assert.AreEqual((byte)'{', bytes[0]);
        Assert.IsFalse(bytes.Take(3).SequenceEqual(new byte[] { 0xef, 0xbb, 0xbf }));
        string text = new UTF8Encoding(false, true).GetString(bytes);
        Assert.IsTrue(text.Contains("\n  \"count\": 2", StringComparison.Ordinal));
        Assert.IsFalse(text.Contains("\"Count\"", StringComparison.Ordinal));
        Assert.IsTrue(text.Contains("\\t", StringComparison.Ordinal));
        Assert.IsTrue(text.Contains("\\n", StringComparison.Ordinal));
        using var document = JsonDocument.Parse(bytes);
        var root = document.RootElement;
        CollectionAssert.AreEqual(new[] { "count", "text", "ids", "labels" },
            root.EnumerateObject().Select(p => p.Name).ToArray());
        Assert.AreEqual(2, root.GetProperty("count").GetInt32());
        Assert.AreEqual("A\t\"quoted\"\nΩ", root.GetProperty("text").GetString());
        CollectionAssert.AreEqual(new long[] { 7, 42 }, root.GetProperty("ids").EnumerateArray().Select(p => p.GetInt64()).ToArray());
        CollectionAssert.AreEqual(new[] { false, true }, root.GetProperty("labels").EnumerateArray().Select(p => p.GetBoolean()).ToArray());
        Assert.AreEqual(JsonNamingPolicy.CamelCase, ArtifactFiles.Json.PropertyNamingPolicy);
        Assert.IsTrue(ArtifactFiles.Json.WriteIndented);
        Assert.AreEqual(0, ArtifactFiles.Utf8.GetPreamble().Length);
        Assert.ThrowsExactly<EncoderFallbackException>(() => ArtifactFiles.Utf8.GetBytes("\ud800"));
        Assert.IsFalse(File.Exists(path + ".partial"));
        Assert.AreEqual("A\t\"quoted\"\nΩ", value.Text);
    }

    [TestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void WriteAndWriteBytes_CreateParentsAndPreserveExistingDestination(bool json)
    {
        using var temp = new TempDirectory();
        string path = temp.FilePath("new/nested/payload");
        byte[] expected = Payload(json);
        Assert.IsFalse(System.IO.Directory.Exists(Path.GetDirectoryName(path)));

        Publish(path, json);

        Assert.IsTrue(System.IO.Directory.Exists(Path.GetDirectoryName(path)));
        ArtifactExpectations.Bytes(expected, path);
        Assert.IsFalse(File.Exists(path + ".partial"));
        Assert.ThrowsExactly<IOException>(() => Publish(path, json));
        ArtifactExpectations.Bytes(expected, path);
        ArtifactExpectations.Bytes(expected, path + ".partial");
    }

    [TestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void WriteAndWriteBytes_RejectExistingPartialAndRetainItsBytes(bool json)
    {
        using var temp = new TempDirectory();
        string path = temp.FilePath("blocked");
        byte[] sentinel = [17, 0, 255, 42];
        temp.Put("blocked.partial", sentinel);

        Assert.ThrowsExactly<IOException>(() => Publish(path, json));

        ArtifactExpectations.Bytes(sentinel, path + ".partial");
        Assert.IsFalse(File.Exists(path));
        CollectionAssert.AreEqual(new[] { path + ".partial" }, System.IO.Directory.GetFiles(temp.Root));
    }

    [TestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void WriteAndWriteBytes_FailedFinalMoveLeavesNewPartial(bool json)
    {
        using var temp = new TempDirectory();
        byte[] original = Encoding.UTF8.GetBytes("immutable-original");
        string path = temp.Put("published", original);

        Assert.ThrowsExactly<IOException>(() => Publish(path, json));

        ArtifactExpectations.Bytes(original, path);
        ArtifactExpectations.Bytes(Payload(json), path + ".partial");
        Assert.ThrowsExactly<IOException>(() => Publish(path, json));
        ArtifactExpectations.Bytes(original, path);
        ArtifactExpectations.Bytes(Payload(json), path + ".partial");
        Assert.AreEqual(2, System.IO.Directory.GetFiles(temp.Root).Length);
    }

    [TestMethod]
    public void Utf8_InvalidBytesThrowDecoderFallbackException()
    {
        byte[] invalid = [0xc3, 0x28];

        Assert.ThrowsExactly<DecoderFallbackException>(() => ArtifactFiles.Utf8.GetString(invalid));

        CollectionAssert.AreEqual(new byte[] { 0xc3, 0x28 }, invalid);
        Assert.AreEqual("AΩ\tB", ArtifactFiles.Utf8.GetString([0x41, 0xce, 0xa9, 0x09, 0x42]));
        Assert.AreEqual(0, ArtifactFiles.Utf8.GetPreamble().Length);
    }

    private static byte[] Payload(bool json) =>
        json ? Encoding.UTF8.GetBytes($"{{{Environment.NewLine}  \"attempt\": 42{Environment.NewLine}}}") : [0, 255, 17, 42];

    private static void Publish(string path, bool json)
    {
        if (json) ArtifactFiles.Write(path, new { Attempt = 42 });
        else ArtifactFiles.WriteBytes(path, Payload(false));
    }
}
