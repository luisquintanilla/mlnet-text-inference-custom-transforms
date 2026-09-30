using Microsoft.ML;

namespace DecisionArrowPredictor;

internal readonly record struct PredictionValue(long RowId, long GroupId, bool Label, double Probability);

public sealed class PredictionBuffer : IReadOnlyList<Prediction>
{
    public const int MaximumControlCursors = 16;
    private readonly long[] ids, groups;
    private readonly bool[] labels;
    private readonly double[] probabilities;
    private ITransformer? boundModel;
    private IDataView? boundInput, boundOutput;
    private RowSelection? indexedSelection;
    private Dictionary<long, int>? ranks;
    private int[]? seen;
    public int Count { get; private set; }
    public int Capacity => ids.Length;
    public Prediction this[int index] => (uint)index < (uint)Count ?
        new(ids[index], groups[index], labels[index], probabilities[index]) :
        throw new ArgumentOutOfRangeException(nameof(index));

    public PredictionBuffer(int capacity)
    {
        ids = new long[capacity]; groups = new long[capacity];
        labels = new bool[capacity]; probabilities = new double[capacity];
    }

    public void Fill(ITransformer model, IDataView input, RowSelection expected, Action<string, long>? allocationObserver = null,
        int requestedCursors = 1)
    {
        long stageStart = allocationObserver is null ? 0 : GC.GetAllocatedBytesForCurrentThread();
        Count = 0;
        if (requestedCursors is < 1 or > MaximumControlCursors)
            throw new ArgumentOutOfRangeException(nameof(requestedCursors), "The bounded control profile supports 1..16 output cursors.");
        if (expected.Count > Capacity) throw new InvalidDataException("Prediction buffer capacity is too small.");
        if (!ReferenceEquals(boundModel, model) || !ReferenceEquals(boundInput, input))
        {
            var nextOutput = model.Transform(input);
            boundModel = model;
            boundInput = input;
            boundOutput = nextOutput;
        }
        IDataView output = boundOutput ?? throw new InvalidOperationException("Prediction output binding is incomplete.");
        var schema = output.Schema;
        var idColumn = schema[nameof(LearningRow.RowId)];
        var groupColumn = schema[nameof(LearningRow.GroupId)];
        var labelColumn = schema[nameof(LearningRow.Label)];
        var probabilityColumn = schema[nameof(ScoredRow.Probability)];
        ObserveAllocation(allocationObserver, "output-binding-and-schema", ref stageStart);
        var cursors = output.GetRowCursorSet([idColumn, groupColumn, labelColumn, probabilityColumn], requestedCursors);
        if (cursors.Length == 0 || cursors.Length > requestedCursors)
        {
            foreach (var owned in cursors) owned.Dispose();
            throw new InvalidDataException("Prediction output returned an invalid cursor count.");
        }
        if (requestedCursors > 1)
        {
            try
            {
                PrepareIndex(expected);
                ObserveAllocation(allocationObserver, "output-cursors-and-rank-index", ref stageStart);
                var failures = new Exception?[cursors.Length];
                Parallel.For(0, cursors.Length, new ParallelOptions { MaxDegreeOfParallelism = cursors.Length }, index =>
                {
                    using var ownedCursor = cursors[index];
                    try
                    {
                        var getSourceId = ownedCursor.GetGetter<long>(idColumn);
                        var getGroupId = ownedCursor.GetGetter<long>(groupColumn);
                        var getLabel = ownedCursor.GetGetter<bool>(labelColumn);
                        var getProbability = ownedCursor.GetGetter<float>(probabilityColumn);
                        long sourceId = 0, groupId = 0; bool isSpam = false; float value = 0;
                        while (ownedCursor.MoveNext())
                        {
                            getSourceId(ref sourceId); getGroupId(ref groupId); getLabel(ref isSpam); getProbability(ref value);
                            // Drain rejected rows so a splitter cannot block on an unconsumed output queue.
                            if (failures[index] is not null) continue;
                            try
                            {
                                if (!ranks!.TryGetValue(sourceId, out int rank))
                                    throw new InvalidDataException("Parallel scored output has an extra source ID.");
                                var metadata = expected.Owner.Metadata[expected[rank]];
                                ProbabilityStore.RequireProbability(value);
                                if (sourceId != metadata.RowId || groupId != metadata.GroupId || isSpam != metadata.Label)
                                    throw new InvalidDataException("Parallel scored output source/group/label association changed.");
                                if (Interlocked.CompareExchange(ref seen![rank], 1, 0) != 0)
                                    throw new InvalidDataException("Parallel output cursors are not disjoint: duplicate source ID.");
                                ids[rank] = sourceId; groups[rank] = groupId; labels[rank] = isSpam; probabilities[rank] = value;
                            }
                            catch (InvalidDataException error) { failures[index] = error; }
                        }
                    }
                    catch (Exception error) when (error is InvalidDataException or InvalidOperationException or ArgumentException)
                    {
                        failures[index] = error;
                    }
                });
                var errors = failures.OfType<Exception>().ToArray();
                if (errors.Length != 0)
                    throw new InvalidDataException("Parallel prediction failed row/getter validation.", new AggregateException(errors));
                for (int rank = 0; rank < expected.Count; rank++)
                    if (seen![rank] != 1) throw new InvalidDataException("Parallel scored output is missing source IDs.");
                ObserveAllocation(allocationObserver, "parallel-rows-and-association", ref stageStart);
                Count = expected.Count;
                return;
            }
            catch (AggregateException error)
            {
                throw new InvalidDataException("Parallel prediction cursor consumption failed.", error.Flatten());
            }
            finally
            {
                foreach (var owned in cursors) owned.Dispose();
            }
        }
        using var cursor = cursors[0];
        ObserveAllocation(allocationObserver, "output-cursor", ref stageStart);
        var id = cursor.GetGetter<long>(idColumn);
        var group = cursor.GetGetter<long>(groupColumn);
        var label = cursor.GetGetter<bool>(labelColumn);
        var probability = cursor.GetGetter<float>(probabilityColumn);
        ObserveAllocation(allocationObserver, "getter-binding", ref stageStart);
        int row = 0;
        while (cursor.MoveNext())
        {
            if (row >= expected.Count) throw new InvalidDataException("Scored output has extra rows.");
            id(ref ids[row]); group(ref groups[row]); label(ref labels[row]);
            float value = 0;
            probability(ref value);
            ProbabilityStore.RequireProbability(value);
            probabilities[row] = value;
            var metadata = expected.Owner.Metadata[expected[row]];
            if (ids[row] != metadata.RowId || groups[row] != metadata.GroupId || labels[row] != metadata.Label)
                throw new InvalidDataException("Scored output source/group/label association changed.");
            row++;
        }
        if (row != expected.Count) throw new InvalidDataException("Scored output is missing rows.");
        ObserveAllocation(allocationObserver, "rows-and-association", ref stageStart);
        cursor.Dispose();
        ObserveAllocation(allocationObserver, "cursor-disposal", ref stageStart);
        Count = row;
    }

    private void PrepareIndex(RowSelection expected)
    {
        if (!ReferenceEquals(indexedSelection, expected))
        {
            var next = new Dictionary<long, int>(expected.Count);
            for (int rank = 0; rank < expected.Count; rank++)
                next.Add(expected.Owner.Metadata[expected[rank]].RowId, rank);
            ranks = next;
            indexedSelection = expected;
        }
        seen ??= new int[Capacity];
        Array.Clear(seen, 0, expected.Count);
    }

    private static void ObserveAllocation(Action<string, long>? observer, string stage, ref long start)
    {
        if (observer is null) return;
        observer(stage, GC.GetAllocatedBytesForCurrentThread() - start);
        start = GC.GetAllocatedBytesForCurrentThread();
    }

    public void FillBaseline(RowSelection selection, string arm, double prior)
    {
        Count = 0;
        selection.Owner.RequireOpen();
        if (selection.Count > Capacity || arm is not ("prior" or "direct"))
            throw new InvalidDataException("Invalid baseline buffer capacity/arm.");
        if (arm == "prior" && (!double.IsFinite(prior) || prior <= 0 || prior >= 1))
            throw new InvalidDataException("Frozen training prevalence must lie strictly between zero and one.");
        for (int row = 0; row < selection.Count; row++)
        {
            int ordinal = selection[row];
            var metadata = selection.Owner.Metadata[ordinal];
            ids[row] = metadata.RowId; groups[row] = metadata.GroupId; labels[row] = metadata.Label;
            probabilities[row] = arm == "prior" ? prior : selection.Owner.Probabilities.Direct(ordinal);
        }
        Count = selection.Count;
    }

    public void RequireReplay(IReadOnlyList<Prediction> expected)
    {
        if (Count != expected.Count) throw new InvalidDataException("Saved model replay count differs.");
        for (int i = 0; i < Count; i++)
        {
            var row = expected[i];
            if (ids[i] != row.RowId || groups[i] != row.GroupId || labels[i] != row.Label ||
                !double.IsFinite(row.Probability) || Math.Abs(probabilities[i] - row.Probability) > 1e-6)
                throw new InvalidDataException("Saved model replay differs from fitted predictions.");
        }
    }

    public void RequireReplay(PredictionBuffer expected)
    {
        if (Count != expected.Count) throw new InvalidDataException("Saved model replay count differs.");
        for (int i = 0; i < Count; i++)
            if (ids[i] != expected.ids[i] || groups[i] != expected.groups[i] || labels[i] != expected.labels[i] ||
                !double.IsFinite(expected.probabilities[i]) ||
                Math.Abs(probabilities[i] - expected.probabilities[i]) > 1e-6)
                throw new InvalidDataException("Saved model replay differs from fitted predictions.");
    }

    public Prediction[] Snapshot() => Enumerable.Range(0, Count).Select(i => this[i]).ToArray();
    internal IReadOnlyList<PredictionValue> Values => new ValueView(this);

    internal sealed class ValueView(PredictionBuffer buffer) : IReadOnlyList<PredictionValue>
    {
        public int Count => buffer.Count;
        public PredictionValue this[int index] => (uint)index < (uint)Count ?
            new(buffer.ids[index], buffer.groups[index], buffer.labels[index], buffer.probabilities[index]) :
            throw new ArgumentOutOfRangeException(nameof(index));
        public IEnumerator<PredictionValue> GetEnumerator()
        {
            for (int i = 0; i < Count; i++) yield return this[i];
        }
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    public IEnumerator<Prediction> GetEnumerator()
    {
        for (int i = 0; i < Count; i++) yield return this[i];
    }
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
}
