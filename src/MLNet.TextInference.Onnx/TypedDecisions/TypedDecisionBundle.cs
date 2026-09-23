using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// A local, versioned typed-decision bundle. Opening a bundle never downloads anything.
/// </summary>
internal sealed class TypedDecisionBundle : IDisposable
{
    public const string ManifestFileName = "typed-decision-bundle.json";

    private readonly bool _ownsRoot;
    private bool _disposed;

    private TypedDecisionBundle(string rootPath, bool ownsRoot, TypedDecisionBundleManifest manifest)
    {
        RootPath = rootPath;
        _ownsRoot = ownsRoot;
        Manifest = manifest;
        Profile = LayaDecisionProfile.Load(rootPath, manifest);
        Tokenizer = LayaTokenizer.Load(Path.Combine(rootPath, manifest.TokenizerDirectory));
    }

    public string RootPath { get; }
    public TypedDecisionBundleManifest Manifest { get; }
    public LayaDecisionProfile Profile { get; }
    public LayaTokenizer Tokenizer { get; }
    public string ModelPath => Path.Combine(RootPath, Manifest.ModelFile);

    public static TypedDecisionBundle Open(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        path = Path.GetFullPath(path);
        if (Directory.Exists(path))
            return OpenDirectory(path, ownsRoot: false);
        if (!File.Exists(path))
            throw new FileNotFoundException("Typed-decision bundle was not found.", path);
        if (!string.Equals(Path.GetExtension(path), ".zip", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Typed-decision bundles must be a directory or a .zip archive.");

        var extractionRoot = Path.Combine(
            Path.GetTempPath(),
            "mlnet-typed-decisions",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(extractionRoot);
        try
        {
            ExtractZipSafely(path, extractionRoot);
            return OpenDirectory(extractionRoot, ownsRoot: true);
        }
        catch
        {
            if (Directory.Exists(extractionRoot))
                Directory.Delete(extractionRoot, recursive: true);
            throw;
        }
    }

    public static TypedDecisionBundle OpenDirectory(string directory, bool ownsRoot = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        directory = Path.GetFullPath(directory);
        var manifestPath = Path.Combine(directory, ManifestFileName);
        var manifest = File.Exists(manifestPath)
            ? JsonSerializer.Deserialize<TypedDecisionBundleManifest>(
                File.ReadAllText(manifestPath), JsonOptions)
            : CreateDirectoryManifest(directory);
        if (manifest is null)
            throw new InvalidDataException($"Could not parse {ManifestFileName}.");
        manifest.Validate();
        ValidateFiles(directory, manifest);
        return new TypedDecisionBundle(directory, ownsRoot, manifest);
    }

    private static TypedDecisionBundleManifest CreateDirectoryManifest(string directory)
    {
        var externalData = File.Exists(Path.Combine(directory, "laya.onnx.data"))
            ? new[] { "laya.onnx.data" }
            : Array.Empty<string>();
        return new TypedDecisionBundleManifest
        {
            ExternalDataFiles = externalData,
            Profile = new TypedDecisionBundleProfile(
                LayaDecisionProfile.EnglishFp32.Name,
                LayaDecisionProfile.EnglishFp32.Revision)
        };
    }

    public static void WriteManifest(
        string bundleDirectory,
        TypedDecisionBundleManifest? manifest = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(bundleDirectory);
        Directory.CreateDirectory(bundleDirectory);
        manifest ??= new TypedDecisionBundleManifest();
        manifest.Validate();
        var hashes = new Dictionary<string, string>(StringComparer.Ordinal);
        AddHash(bundleDirectory, manifest.ModelFile, hashes);
        foreach (var external in manifest.ExternalDataFiles)
            AddHash(bundleDirectory, external, hashes);
        AddHash(bundleDirectory, "laya_config.json", hashes);
        AddHash(bundleDirectory, Path.Combine(manifest.TokenizerDirectory, "tokenizer.json"), hashes);
        var configPath = Path.Combine(bundleDirectory, Path.Combine(manifest.TokenizerDirectory, "tokenizer_config.json"));
        if (File.Exists(configPath))
            AddHash(bundleDirectory, Path.Combine(manifest.TokenizerDirectory, "tokenizer_config.json"), hashes);
        manifest.FileSha256.Clear();
        foreach (var pair in hashes)
            manifest.FileSha256[pair.Key] = pair.Value;
        File.WriteAllText(
            Path.Combine(bundleDirectory, ManifestFileName),
            JsonSerializer.Serialize(manifest, JsonOptionsIndented));
    }

    public static void CreateZip(string bundleDirectory, string zipPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(bundleDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(zipPath);
        using var bundle = OpenDirectory(bundleDirectory);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(zipPath))!);
        if (File.Exists(zipPath))
            File.Delete(zipPath);
        ZipFile.CreateFromDirectory(bundle.RootPath, zipPath, CompressionLevel.Optimal, includeBaseDirectory: false);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        if (_ownsRoot && Directory.Exists(RootPath))
            Directory.Delete(RootPath, recursive: true);
    }

    private static void ExtractZipSafely(string zipPath, string destination)
    {
        using var archive = ZipFile.OpenRead(zipPath);
        var root = Path.GetFullPath(destination) + Path.DirectorySeparatorChar;
        foreach (var entry in archive.Entries)
        {
            var relative = entry.FullName.Replace('/', Path.DirectorySeparatorChar);
            TypedDecisionBundleManifest.ValidateRelativePath(relative, "archive entry");
            var target = Path.GetFullPath(Path.Combine(destination, relative));
            if (!target.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"Archive entry '{entry.FullName}' escapes the extraction directory.");
            if (string.IsNullOrEmpty(entry.Name))
            {
                Directory.CreateDirectory(target);
                continue;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            entry.ExtractToFile(target, overwrite: false);
        }
    }

    private static void ValidateFiles(string directory, TypedDecisionBundleManifest manifest)
    {
        var required = new List<string>
        {
            manifest.ModelFile,
            "laya_config.json",
            Path.Combine(manifest.TokenizerDirectory, "tokenizer.json")
        };
        required.AddRange(manifest.ExternalDataFiles);
        foreach (var relative in required)
        {
            TypedDecisionBundleManifest.ValidateRelativePath(relative, "bundle file");
            var path = Path.Combine(directory, relative);
            if (!File.Exists(path))
                throw new FileNotFoundException($"Typed-decision bundle is missing '{relative}'.", path);
            if (manifest.FileSha256.TryGetValue(relative.Replace('\\', '/'), out var expected))
            {
                var actual = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
                if (!string.Equals(expected, actual, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException($"SHA-256 mismatch for bundle file '{relative}'.");
            }
        }
    }

    private static void AddHash(string directory, string relative, IDictionary<string, string> hashes)
    {
        var path = Path.Combine(directory, relative);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Cannot write a bundle manifest; '{relative}' is missing.", path);
        hashes[relative.Replace('\\', '/')] =
            Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter() }
    };

    private static readonly JsonSerializerOptions JsonOptionsIndented = new(JsonOptions)
    {
        WriteIndented = true
    };
}
