using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using MLNet.TextInference.Onnx;

namespace MLNet.TextInference.TypedDecisions;

/// <summary>
/// A local, versioned typed-decision bundle. Opening a bundle never downloads anything.
/// </summary>
internal sealed class TypedDecisionBundle : IDisposable
{
    public const string ManifestFileName = "typed-decision-bundle.json";

    private readonly bool _ownsRoot;
    private readonly Lazy<LayaDecisionProfile> _profile;
    private readonly Lazy<LayaTokenizerMetadata> _tokenizerMetadata;
    private readonly Lazy<LayaTokenizer> _tokenizer;
    private bool _disposed;

    private TypedDecisionBundle(string rootPath, bool ownsRoot, TypedDecisionBundleManifest manifest)
    {
        RootPath = rootPath;
        _ownsRoot = ownsRoot;
        Manifest = manifest;
        _profile = new Lazy<LayaDecisionProfile>(
            () => LayaDecisionProfile.Load(RootPath, Manifest),
            LazyThreadSafetyMode.ExecutionAndPublication);
        _tokenizerMetadata = new Lazy<LayaTokenizerMetadata>(
            () => LayaTokenizer.LoadMetadata(Path.Combine(RootPath, Manifest.TokenizerDirectory)),
            LazyThreadSafetyMode.ExecutionAndPublication);
        _tokenizer = new Lazy<LayaTokenizer>(
            () => LayaTokenizer.Load(Path.Combine(RootPath, Manifest.TokenizerDirectory)),
            LazyThreadSafetyMode.ExecutionAndPublication);
    }

    public string RootPath { get; }
    public TypedDecisionBundleManifest Manifest { get; }
    public LayaDecisionProfile Profile => _profile.Value;
    public LayaTokenizerMetadata TokenizerMetadata => _tokenizerMetadata.Value;
    public LayaTokenizer Tokenizer => _tokenizer.Value;
    public string ModelPath => Path.Combine(RootPath, Manifest.ModelFile);

    public static TypedDecisionBundle Open(
        string path,
        TypedDecisionBundleLoadRequirements requirements =
            TypedDecisionBundleLoadRequirements.All)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        path = Path.GetFullPath(path);
        if (Directory.Exists(path))
            return OpenDirectory(path, ownsRoot: false, requirements);
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
            AssetArchive.ExtractZipSafely(path, extractionRoot);
            return OpenDirectory(extractionRoot, ownsRoot: true, requirements);
        }
        catch
        {
            if (Directory.Exists(extractionRoot))
                Directory.Delete(extractionRoot, recursive: true);
            throw;
        }
    }

    public static TypedDecisionBundle OpenDirectory(
        string directory,
        bool ownsRoot = false,
        TypedDecisionBundleLoadRequirements requirements =
            TypedDecisionBundleLoadRequirements.All)
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
        ValidateFiles(directory, manifest, requirements);
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

    private static void ValidateFiles(
        string directory,
        TypedDecisionBundleManifest manifest,
        TypedDecisionBundleLoadRequirements requirements)
    {
        var required = new List<string>();
        if (requirements.HasFlag(TypedDecisionBundleLoadRequirements.Model))
        {
            required.Add(manifest.ModelFile);
            required.AddRange(manifest.ExternalDataFiles);
        }
        if (requirements.HasFlag(TypedDecisionBundleLoadRequirements.Profile))
            required.Add("laya_config.json");
        if (requirements.HasFlag(TypedDecisionBundleLoadRequirements.Tokenizer))
            required.Add(Path.Combine(manifest.TokenizerDirectory, "tokenizer.json"));
        foreach (var relative in required)
        {
            TypedDecisionBundleManifest.ValidateRelativePath(relative, "bundle file");
            var path = Path.Combine(directory, relative);
            if (!File.Exists(path))
                throw new FileNotFoundException($"Typed-decision bundle is missing '{relative}'.", path);
            if (manifest.FileSha256.TryGetValue(relative.Replace('\\', '/'), out var expected))
            {
                var actual = ComputeSha256(path);
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
        hashes[relative.Replace('\\', '/')] = ComputeSha256(path);
    }

    private static string ComputeSha256(string path)
    {
        using var stream = File.OpenRead(path);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = new byte[1024 * 1024];
        int read;
        while ((read = stream.Read(buffer, 0, buffer.Length)) > 0)
            hash.AppendData(buffer, 0, read);
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
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

[Flags]
internal enum TypedDecisionBundleLoadRequirements
{
    Model = 1,
    Profile = 2,
    Tokenizer = 4,
    All = Model | Profile | Tokenizer
}
