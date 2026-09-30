using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using Apache.Arrow;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DecisionArrowPredictor;

internal sealed class OriginalConsumerOracle : IDisposable
{
    private readonly AssemblyLoadContext context = new("frozen-consumer-oracle", isCollectible: true);
    private readonly Assembly assembly;
    public Func<RecordBatch, System.Array> Project { get; }
    public string ReceiptSha256 { get; }

    public OriginalConsumerOracle(string directory)
    {
        string receipt = Path.Combine(directory, "reference.receipt.json");
        using var document = JsonDocument.Parse(File.ReadAllBytes(receipt));
        var root = document.RootElement;
        if (root.GetProperty("status").GetString() != "FROZEN" ||
            root.GetProperty("sourceCommit").GetString() != "ca1381be58f1d935c31b545aa2e99e4d0a5bddad")
            throw new InvalidDataException("Expected the immutable original consumer reference.");
        foreach (var file in root.GetProperty("files").EnumerateArray())
        {
            string relative = file.GetProperty("file").GetString()!;
            string path = Path.GetFullPath(Path.Combine(directory, relative));
            if (!path.StartsWith(Path.GetFullPath(directory) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) ||
                new FileInfo(path).Length != file.GetProperty("sizeBytes").GetInt64())
                throw new InvalidDataException("Original reference closure path/size mismatch.");
            ArtifactFiles.RequireHash(path, file.GetProperty("sha256").GetString()!);
        }
        ReceiptSha256 = ArtifactFiles.Hash(receipt);
        // Portable oracle dependencies stay original when the consumer later changes adapter pins.
        context.Resolving += (_, name) =>
        {
            string frozenPath = Path.Combine(directory, "runtime", name.Name + ".dll");
            if (name.Name?.StartsWith("DecisionInference.", StringComparison.Ordinal) == true)
                return context.LoadFromAssemblyPath(Path.GetFullPath(frozenPath));
            var shared = AssemblyLoadContext.Default.LoadFromAssemblyName(name);
            if (File.Exists(frozenPath) && ArtifactFiles.Hash(shared.Location) != ArtifactFiles.Hash(frozenPath))
                throw new InvalidDataException($"Shared oracle dependency differs from original frozen bytes: {name.Name}.");
            return shared;
        };
        assembly = context.LoadFromAssemblyPath(Path.GetFullPath(Path.Combine(directory, "runtime", "DecisionArrowPredictor.dll")));
        var method = assembly.GetType("DecisionArrowPredictor.ArrowFeatureReader", true)!.GetMethod("Project")!;
        Project = method.CreateDelegate<Func<RecordBatch, System.Array>>();
    }

    public async Task<object> ImportAsync(string[] paths)
    {
        var method = assembly.GetType("DecisionArrowPredictor.ArrowFeatureReader", true)!.GetMethod("ImportAsync")!;
        var task = (Task)method.Invoke(null, [.. paths.Cast<object>(), CancellationToken.None])!;
        await task;
        return task.GetType().GetProperty("Result")!.GetValue(task)!;
    }

    public object Rows(object imported) => imported.GetType().GetProperty("Rows")!.GetValue(imported)!;

    public System.Array BridgeRows(ImportedStudy imported)
    {
        var type = assembly.GetType("DecisionArrowPredictor.LearningRow", true)!;
        var id = type.GetProperty(nameof(LearningRow.RowId))!;
        var group = type.GetProperty(nameof(LearningRow.GroupId))!;
        var label = type.GetProperty(nameof(LearningRow.Label))!;
        var text = type.GetProperty(nameof(LearningRow.Text))!;
        var semantic = type.GetProperty(nameof(LearningRow.Semantic))!;
        var direct = type.GetProperty(nameof(LearningRow.SpamBaseline))!;
        var rows = System.Array.CreateInstance(type, imported.Rows.Length);
        for (int i = 0; i < rows.Length; i++)
        {
            var source = imported.Rows[i];
            var row = Activator.CreateInstance(type) ??
                throw new InvalidDataException("Original consumer row construction failed.");
            id.SetValue(row, source.RowId); group.SetValue(row, source.GroupId);
            label.SetValue(row, source.Label); text.SetValue(row, source.Text);
            semantic.SetValue(row, source.Semantic.ToArray()); direct.SetValue(row, source.SpamBaseline);
            rows.SetValue(row, i);
        }
        return rows;
    }

    public void RequireContent(object imported, StudyData data) => RequireRowsContent((System.Array)Rows(imported), data);

    public void RequireRowsContent(System.Array rows, StudyData data)
    {
        if (rows.Length != data.Metadata.Count) throw new InvalidDataException("Original/compact content count differs.");
        if (rows.Length == 0) return;
        var rowType = rows.GetValue(0)!.GetType();
        var sourceId = rowType.GetProperty("RowId")!;
        var sourceGroup = rowType.GetProperty("GroupId")!;
        var sourceLabel = rowType.GetProperty("Label")!;
        var sourceText = rowType.GetProperty("Text")!;
        var sourceSemantic = rowType.GetProperty("Semantic")!;
        var sourceDirect = rowType.GetProperty("SpamBaseline")!;
        var view = data.All().View();
        using var cursor = view.GetRowCursor([view.Schema["Semantic"], view.Schema["SpamBaseline"]]);
        var features = cursor.GetGetter<VBuffer<float>>(view.Schema["Semantic"]);
        var direct = cursor.GetGetter<double>(view.Schema["SpamBaseline"]);
        VBuffer<float> values = default;
        double baseline = 0;
        int row = 0;
        while (cursor.MoveNext())
        {
            var original = rows.GetValue(row)!;
            var metadata = data.Metadata[row];
            if ((long)sourceId.GetValue(original)! != metadata.RowId ||
                (long)sourceGroup.GetValue(original)! != metadata.GroupId ||
                (bool)sourceLabel.GetValue(original)! != metadata.Label ||
                !string.Equals((string)sourceText.GetValue(original)!, metadata.Text, StringComparison.Ordinal))
                throw new InvalidDataException("Original/compact source/text/group/label content differs.");
            features(ref values); direct(ref baseline);
            var expected = (float[])sourceSemantic.GetValue(original)!;
            if (values.Length != expected.Length ||
                BitConverter.DoubleToInt64Bits(baseline) != BitConverter.DoubleToInt64Bits((double)sourceDirect.GetValue(original)!))
                throw new InvalidDataException("Original/compact numeric content differs.");
            for (int j = 0; j < expected.Length; j++)
                if (BitConverter.SingleToInt32Bits(values.GetValues()[j]) != BitConverter.SingleToInt32Bits(expected[j]))
                    throw new InvalidDataException("Original/compact semantic feature bits/order differ.");
            row++;
        }
        if (row != rows.Length) throw new InvalidDataException("Original/compact cursor is missing rows.");
    }

    public System.Array Predict(ITransformer model, object rows) =>
        (System.Array)assembly.GetType("DecisionArrowPredictor.PredictorTraining", true)!.GetMethod("Predict")!
            .Invoke(null, [model, rows])!;

    public static Prediction[] Predictions(System.Array values)
    {
        if (values.Length == 0) return [];
        var type = values.GetValue(0)!.GetType();
        var id = type.GetProperty("RowId")!;
        var group = type.GetProperty("GroupId")!;
        var label = type.GetProperty("Label")!;
        var probability = type.GetProperty("Probability")!;
        var result = new Prediction[values.Length];
        for (int i = 0; i < result.Length; i++)
        {
            var value = values.GetValue(i)!;
            result[i] = new((long)id.GetValue(value)!, (long)group.GetValue(value)!,
                (bool)label.GetValue(value)!, (double)probability.GetValue(value)!);
        }
        return result;
    }

    public static FeatureObservation[] Observations(System.Array values)
    {
        if (values.Length == 0) return [];
        var type = values.GetValue(0)!.GetType();
        var id = type.GetProperty(nameof(FeatureObservation.RowId))!;
        var semantic = type.GetProperty(nameof(FeatureObservation.Semantic))!;
        var direct = type.GetProperty(nameof(FeatureObservation.SpamBaseline))!;
        var result = new FeatureObservation[values.Length];
        for (int i = 0; i < result.Length; i++)
        {
            var value = values.GetValue(i)!;
            result[i] = new((long)id.GetValue(value)!, (float[])semantic.GetValue(value)!,
                (double)direct.GetValue(value)!);
        }
        return result;
    }

    public void Dispose() => context.Unload();
}
