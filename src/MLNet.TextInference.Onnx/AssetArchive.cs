using System.IO.Compression;

namespace MLNet.TextInference.Onnx;

internal static class AssetArchive
{
    internal static void ExtractZipSafely(string zipPath, string destination)
    {
        using var archive = ZipFile.OpenRead(zipPath);
        var fullDestination = Path.GetFullPath(destination);
        var root = fullDestination.TrimEnd(
            Path.DirectorySeparatorChar,
            Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        foreach (var entry in archive.Entries)
        {
            var relative = NormalizeRelativePath(
                entry.FullName,
                "archive entry",
                allowTrailingSeparator: string.IsNullOrEmpty(entry.Name));
            var target = Path.GetFullPath(Path.Combine(fullDestination, relative));
            if (!target.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    $"Archive entry '{entry.FullName}' escapes the extraction directory.");

            if (string.IsNullOrEmpty(entry.Name))
            {
                Directory.CreateDirectory(target);
                continue;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            entry.ExtractToFile(target, overwrite: false);
        }
    }

    internal static string ResolveWithinRoot(
        string root,
        string relativePath,
        string description)
    {
        var relative = NormalizeRelativePath(relativePath, description);
        var fullRoot = Path.GetFullPath(root)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            + Path.DirectorySeparatorChar;
        var fullPath = Path.GetFullPath(Path.Combine(root, relative));
        if (!fullPath.StartsWith(fullRoot, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException(
                $"{description} '{relativePath}' escapes the asset root.");
        return fullPath;
    }

    internal static string NormalizeRelativePath(
        string path,
        string description,
        bool allowTrailingSeparator = false)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new InvalidDataException($"{description} cannot be empty.");

        var normalized = path.Replace('\\', '/');
        if (allowTrailingSeparator && normalized.EndsWith('/'))
            normalized = normalized.TrimEnd('/');
        if (normalized.StartsWith('/') ||
            normalized.Contains('\0') ||
            Path.IsPathRooted(normalized))
            throw new InvalidDataException(
                $"{description} '{path}' must be a relative path.");

        var parts = normalized.Split('/');
        if (parts.Length == 0 ||
            parts.Any(static part => part.Length == 0 || part == "."))
            throw new InvalidDataException(
                $"{description} '{path}' contains an invalid path segment.");
        if (parts.Any(static part => part == ".."))
            throw new InvalidDataException(
                $"{description} '{path}' escapes the asset root.");
        return string.Join('/', parts);
    }

    internal static IReadOnlyList<string> DiscoverOnnxExternalDataFiles(
        string modelPath)
    {
        var bytes = File.ReadAllBytes(modelPath);
        var locations = new HashSet<string>(StringComparer.Ordinal);
        var model = new ProtoReader(bytes);
        while (model.TryReadField(out var field))
        {
            if (field.WireType != 2)
                continue;
            switch (field.Number)
            {
                case 7:
                    ReadGraph(field.Bytes, locations);
                    break;
                case 20:
                    ReadTrainingInfo(field.Bytes, locations);
                    break;
                case 25:
                    ReadFunction(field.Bytes, locations);
                    break;
            }
        }

        var modelDirectory = Path.GetDirectoryName(Path.GetFullPath(modelPath))!;
        var result = new List<string>(locations.Count);
        foreach (var location in locations.Order(StringComparer.Ordinal))
        {
            var relative = NormalizeRelativePath(location, "ONNX external-data location");
            var fullPath = ResolveWithinRoot(modelDirectory, relative, "ONNX external-data location");
            if (!File.Exists(fullPath))
                throw new FileNotFoundException(
                    $"ONNX external-data file '{relative}' referenced by '{modelPath}' was not found.",
                    fullPath);
            result.Add(relative);
        }

        return result;
    }

    private static void ReadGraph(
        ReadOnlySpan<byte> bytes,
        ISet<string> locations)
    {
        var graph = new ProtoReader(bytes);
        while (graph.TryReadField(out var field))
        {
            if (field.WireType != 2)
                continue;
            switch (field.Number)
            {
                case 1:
                    ReadNode(field.Bytes, locations);
                    break;
                case 5:
                    ReadTensor(field.Bytes, locations);
                    break;
                case 15:
                    ReadSparseTensor(field.Bytes, locations);
                    break;
            }
        }
    }

    private static void ReadNode(
        ReadOnlySpan<byte> bytes,
        ISet<string> locations)
    {
        var node = new ProtoReader(bytes);
        while (node.TryReadField(out var field))
        {
            if (field.Number == 5 && field.WireType == 2)
                ReadAttribute(field.Bytes, locations);
        }
    }

    private static void ReadAttribute(
        ReadOnlySpan<byte> bytes,
        ISet<string> locations)
    {
        var attribute = new ProtoReader(bytes);
        while (attribute.TryReadField(out var field))
        {
            if (field.WireType != 2)
                continue;
            switch (field.Number)
            {
                case 5:
                    ReadTensor(field.Bytes, locations);
                    break;
                case 6:
                    ReadGraph(field.Bytes, locations);
                    break;
                case 10:
                    ReadTensor(field.Bytes, locations);
                    break;
                case 11:
                    ReadGraph(field.Bytes, locations);
                    break;
                case 22:
                case 23:
                    ReadSparseTensor(field.Bytes, locations);
                    break;
            }
        }
    }

    private static void ReadSparseTensor(
        ReadOnlySpan<byte> bytes,
        ISet<string> locations)
    {
        var sparse = new ProtoReader(bytes);
        while (sparse.TryReadField(out var field))
        {
            if (field.WireType == 2 && field.Number is 1 or 2)
                ReadTensor(field.Bytes, locations);
        }
    }

    private static void ReadTrainingInfo(
        ReadOnlySpan<byte> bytes,
        ISet<string> locations)
    {
        var training = new ProtoReader(bytes);
        while (training.TryReadField(out var field))
        {
            if (field.WireType == 2 && field.Number is 1 or 2)
                ReadGraph(field.Bytes, locations);
        }
    }

    private static void ReadFunction(
        ReadOnlySpan<byte> bytes,
        ISet<string> locations)
    {
        var function = new ProtoReader(bytes);
        while (function.TryReadField(out var field))
        {
            if (field.WireType != 2)
                continue;
            switch (field.Number)
            {
                case 7:
                    ReadNode(field.Bytes, locations);
                    break;
                case 11:
                    ReadAttribute(field.Bytes, locations);
                    break;
            }
        }
    }

    private static void ReadTensor(
        ReadOnlySpan<byte> bytes,
        ISet<string> locations)
    {
        var tensor = new ProtoReader(bytes);
        var isExternal = false;
        var tensorLocations = new List<string>();
        while (tensor.TryReadField(out var field))
        {
            if (field.Number == 13 && field.WireType == 2)
            {
                var entry = new ProtoReader(field.Bytes);
                string? key = null;
                string? value = null;
                while (entry.TryReadField(out var entryField))
                {
                    if (entryField.Number == 1 && entryField.WireType == 2)
                        key = entryField.Utf8;
                    else if (entryField.Number == 2 && entryField.WireType == 2)
                        value = entryField.Utf8;
                }

                if (string.Equals(key, "location", StringComparison.Ordinal) &&
                    value is not null)
                    tensorLocations.Add(value);
            }
            else if (field.Number == 14 && field.WireType == 0)
            {
                isExternal = field.Varint == 1;
            }
        }

        if (isExternal)
        {
            if (tensorLocations.Count == 0)
                throw new InvalidDataException(
                    "ONNX tensor is marked external but has no location.");
            foreach (var location in tensorLocations)
                locations.Add(location);
        }
    }

    private readonly ref struct ProtoField
    {
        internal ProtoField(int number, int wireType, ulong varint, ReadOnlySpan<byte> bytes)
        {
            Number = number;
            WireType = wireType;
            Varint = varint;
            Bytes = bytes;
        }

        internal int Number { get; }
        internal int WireType { get; }
        internal ulong Varint { get; }
        internal ReadOnlySpan<byte> Bytes { get; }
        internal string Utf8 => System.Text.Encoding.UTF8.GetString(Bytes);
    }

    private ref struct ProtoReader
    {
        private readonly ReadOnlySpan<byte> _bytes;
        private int _offset;

        internal ProtoReader(ReadOnlySpan<byte> bytes)
        {
            _bytes = bytes;
            _offset = 0;
        }

        internal bool TryReadField(out ProtoField field)
        {
            if (_offset >= _bytes.Length)
            {
                field = default;
                return false;
            }

            var tag = ReadVarint();
            var number = checked((int)(tag >> 3));
            var wireType = checked((int)(tag & 7));
            switch (wireType)
            {
                case 0:
                    field = new ProtoField(number, wireType, ReadVarint(), default);
                    return true;
                case 1:
                    Skip(8);
                    field = new ProtoField(number, wireType, 0, default);
                    return true;
                case 2:
                    var length = checked((int)ReadVarint());
                    if (length < 0 || _offset > _bytes.Length - length)
                        throw new InvalidDataException("Malformed ONNX protobuf length.");
                    var bytes = _bytes.Slice(_offset, length);
                    _offset += length;
                    field = new ProtoField(number, wireType, 0, bytes);
                    return true;
                case 5:
                    Skip(4);
                    field = new ProtoField(number, wireType, 0, default);
                    return true;
                default:
                    throw new InvalidDataException(
                        $"Unsupported ONNX protobuf wire type {wireType}.");
            }
        }

        private ulong ReadVarint()
        {
            ulong value = 0;
            for (var shift = 0; shift < 64; shift += 7)
            {
                if (_offset >= _bytes.Length)
                    throw new InvalidDataException("Malformed ONNX protobuf varint.");
                var current = _bytes[_offset++];
                value |= (ulong)(current & 0x7f) << shift;
                if ((current & 0x80) == 0)
                    return value;
            }

            throw new InvalidDataException("Malformed ONNX protobuf varint.");
        }

        private void Skip(int count)
        {
            if (count < 0 || _offset > _bytes.Length - count)
                throw new InvalidDataException("Malformed ONNX protobuf field.");
            _offset += count;
        }
    }
}
