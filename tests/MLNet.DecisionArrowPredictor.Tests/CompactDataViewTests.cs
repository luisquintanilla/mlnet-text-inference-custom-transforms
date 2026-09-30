using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class CompactDataViewTests
{
    [TestMethod]
    [DataRow(0, 0L)]
    [DataRow(1, 48L)]
    [DataRow(1024, 49152L)]
    [DataRow(1025, 49200L)]
    [DataRow(1048577, 50331696L)]
    public void Store_CountsActualFinalBlockCapacityAndRejectsOneByteUnderCap(int rows, long expected)
    {
        using var store = new ProbabilityStore(rows, expected);
        Assert.AreEqual(rows, store.Count);
        Assert.AreEqual(expected, store.NumericCapacityBytes);
        Assert.AreEqual(expected, store.NumericCapBytes);
        if (expected > 0)
        {
            var error = Assert.ThrowsExactly<InvalidDataException>(() => new ProbabilityStore(rows, expected - 1));
            StringAssert.Contains(error.Message, "no unlimited fallback");
        }
    }

    [TestMethod]
    public void Store_ImpossibleCountIsRejectedBeforeAllocationAndNegativeArgumentsFail()
    {
        Assert.ThrowsExactly<InvalidDataException>(() => new ProbabilityStore(int.MaxValue));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new ProbabilityStore(-1));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new ProbabilityStore(1, -1));
    }

    [TestMethod]
    public void Selection_SubsetsShareCanonicalMetadataAndMatchLegacyGroupedPolicy()
    {
        var legacy = StudyFixture.Authored();
        using var data = StudyData.FromLegacy(legacy);
        var train = data.Partition("train");
        foreach (int target in new[] { 100, 500, 600 })
        {
            var subset = train.Subset(target);
            long[] expected = PredictorTraining.Subset(legacy.Rows.Where(r => r.RowId < 1000).ToArray(), target)
                .Select(r => r.RowId).ToArray();
            CollectionAssert.AreEqual(expected, subset.SourceIds());
            foreach (int ordinal in Enumerable.Range(0, subset.Count).Select(i => subset[i]))
                Assert.AreSame(legacy.Rows[ordinal].Text, data.Metadata[ordinal].Text);
            Assert.AreEqual((double)legacy.Rows.Count(r => expected.Contains(r.RowId) && r.Label) / expected.Length,
                subset.Prevalence());
        }
        Assert.AreSame(train.View().Schema, data.Partition("validation").View().Schema);
    }

    [TestMethod]
    public void Cursor_LocalIdentityRandomAndFeatureBitsMatchLoadFromEnumerableOnGappedNonmonotonicIds()
    {
        var original = StudyFixture.Authored();
        var reordered = original with { Rows = original.Rows.Reverse().ToArray() };
        using var data = StudyData.FromLegacy(reordered);
        var rows = new MLContext(1).Data.LoadFromEnumerable(reordered.Rows);
        var compact = data.All().View();
        Assert.AreEqual(rows.CanShuffle, compact.CanShuffle);
        Assert.IsFalse(compact.CanShuffle);
        for (int pass = 0; pass < 2; pass++)
        {
            using var a = rows.GetRowCursor(rows.Schema, new Random(1729));
            using var b = compact.GetRowCursor(compact.Schema, new Random(1729));
            var sourceA = a.GetGetter<long>(a.Schema["RowId"]);
            var sourceB = b.GetGetter<long>(b.Schema["RowId"]);
            var idA = a.GetIdGetter(); var idB = b.GetIdGetter();
            var featureA = a.GetGetter<VBuffer<float>>(a.Schema["Semantic"]);
            var featureB = b.GetGetter<VBuffer<float>>(b.Schema["Semantic"]);
            VBuffer<float> av = default, bv = default;
            int count = 0;
            while (a.MoveNext())
            {
                Assert.IsTrue(b.MoveNext());
                long ai = 0, bi = 0;
                sourceA(ref ai); sourceB(ref bi);
                Assert.AreEqual(reordered.Rows[count].RowId, ai);
                Assert.AreEqual(ai, bi);
                DataViewRowId ar = default, br = default;
                idA(ref ar); idB(ref br);
                Assert.AreEqual(new DataViewRowId((ulong)count, 0), ar);
                Assert.AreEqual(ar, br);
                featureA(ref av); featureB(ref bv);
                CollectionAssert.AreEqual(av.GetValues().ToArray().Select(BitConverter.SingleToInt32Bits).ToArray(),
                    bv.GetValues().ToArray().Select(BitConverter.SingleToInt32Bits).ToArray());
                count++;
            }
            Assert.IsFalse(b.MoveNext());
            Assert.AreEqual(reordered.Rows.Length, count);
        }
    }

    [TestMethod]
    public void Cursor_IndependentActiveMasksInvalidPositionsAndCursorSetDoNotDuplicateRows()
    {
        using var data = StudyData.FromLegacy(StudyFixture.Authored());
        var view = data.All().View();
        using var a = view.GetRowCursor([view.Schema["Text"]]);
        using var b = view.GetRowCursor([view.Schema["Semantic"]]);
        Assert.IsFalse(a.IsColumnActive(view.Schema["Semantic"]));
        Assert.ThrowsExactly<InvalidOperationException>(() => a.GetGetter<VBuffer<float>>(view.Schema["Semantic"]));
        Assert.ThrowsExactly<InvalidOperationException>(() => b.GetGetter<double>(view.Schema["Semantic"]));
        var text = a.GetGetter<ReadOnlyMemory<char>>(view.Schema["Text"]);
        var semantic = b.GetGetter<VBuffer<float>>(view.Schema["Semantic"]);
        ReadOnlyMemory<char> t = default;
        VBuffer<float> v = default;
        Assert.ThrowsExactly<InvalidOperationException>(() => text(ref t));
        Assert.ThrowsExactly<InvalidOperationException>(() => semantic(ref v));
        Assert.IsTrue(a.MoveNext()); Assert.IsTrue(a.MoveNext()); Assert.IsTrue(b.MoveNext());
        Assert.AreEqual(1L, a.Position); Assert.AreEqual(0L, b.Position);
        text(ref t); semantic(ref v);
        Assert.AreEqual(data.Metadata[1].Text, t.ToString());
        CollectionAssert.AreEqual(StudyFixture.Authored().Rows[0].Semantic, v.GetValues().ToArray());
        var set = view.GetRowCursorSet([view.Schema["RowId"]], 4);
        Assert.AreEqual(1, set.Length);
        using var only = set[0];
        int count = 0;
        while (only.MoveNext()) count++;
        Assert.AreEqual(data.Metadata.Count, count);
        Assert.ThrowsExactly<InvalidOperationException>(() =>
        {
            DataViewRowId id = default;
            only.GetIdGetter()(ref id);
        });
    }

    [TestMethod]
    public void Cursor_CallerBuffersNeverAliasStoreAndSurviveOwnerDisposalWhileActiveCursorFinishes()
    {
        var legacy = StudyFixture.Authored();
        using var data = StudyData.FromLegacy(legacy);
        var view = data.All().View();
        using var cursor = view.GetRowCursor([view.Schema["Semantic"]]);
        Assert.AreEqual(1, data.ActiveCursors);
        var get = cursor.GetGetter<VBuffer<float>>(view.Schema["Semantic"]);
        VBuffer<float> first = default, second = default;
        Assert.IsTrue(cursor.MoveNext()); get(ref first);
        float[] frozen = first.GetValues().ToArray();
        legacy.Rows[0].Semantic[0] = .99f;
        get(ref second);
        CollectionAssert.AreEqual(frozen, second.GetValues().ToArray());
        var editor = VBufferEditor.Create(ref first, 10);
        editor.Values[0] = .77f;
        first = editor.Commit();
        get(ref second);
        CollectionAssert.AreEqual(frozen, second.GetValues().ToArray());
        data.Dispose();
        Assert.ThrowsExactly<ObjectDisposedException>(() => view.GetRowCursor(view.Schema));
        Assert.ThrowsExactly<ObjectDisposedException>(() => data.All());
        Assert.IsTrue(cursor.MoveNext()); get(ref second);
        cursor.Dispose();
        Assert.AreEqual(0, data.ActiveCursors);
        Assert.AreEqual(.77f, first.GetValues()[0]);
        Assert.AreEqual(10, second.Length);
        Assert.IsFalse(cursor.MoveNext());
        Assert.ThrowsExactly<InvalidOperationException>(() => get(ref second));
    }

    [TestMethod]
    public void PredictionBuffer_BaselinesReuseCapacityAndRejectWrongReplayAssociation()
    {
        using var data = StudyData.FromLegacy(StudyFixture.Authored());
        var selected = data.Partition("validation");
        var buffer = new PredictionBuffer(selected.Count);
        buffer.FillBaseline(selected, "direct", 0);
        var saved = buffer.Snapshot();
        buffer.FillBaseline(selected, "prior", .25);
        Assert.AreEqual(selected.Count, buffer.Count);
        Assert.AreEqual(selected.Count, buffer.Capacity);
        for (int i = 0; i < buffer.Count; i++)
        {
            Assert.AreEqual(saved[i].RowId, buffer[i].RowId);
            Assert.AreEqual(saved[i].GroupId, buffer[i].GroupId);
            Assert.AreEqual(saved[i].Label, buffer[i].Label);
            Assert.AreEqual(.25, buffer[i].Probability);
        }
        buffer.FillBaseline(selected, "direct", 0);
        buffer.RequireReplay(saved);
        saved[0] = saved[0] with { GroupId = -1 };
        Assert.ThrowsExactly<InvalidDataException>(() => buffer.RequireReplay(saved));
        Assert.ThrowsExactly<InvalidDataException>(() => buffer.RequireReplay(saved[..^1]));
        Assert.ThrowsExactly<InvalidDataException>(() => buffer.FillBaseline(selected, "prior", 0));
        Assert.ThrowsExactly<InvalidDataException>(() => new PredictionBuffer(1).FillBaseline(selected, "direct", 0));
    }

    [TestMethod]
    [DataRow("Semantic")]
    [DataRow("Features")]
    public void DiagnosticTrace_ForwardsIdentityActiveColumnsRandomAndDenseFeatureBitsWithoutText(string features)
    {
        var legacy = StudyFixture.Authored();
        using var data = StudyData.FromLegacy(legacy);
        var selected = data.Partition("train").Subset(100);
        long[] ids = selected.SourceIds();
        var context = new MLContext(1);
        IDataView a = context.Data.LoadFromEnumerable(legacy.Rows.Where(r => ids.Contains(r.RowId)).OrderBy(r => r.RowId));
        IDataView b = selected.View();
        if (features == "Features")
        {
            var copy = context.Transforms.CopyColumns(features, "Semantic");
            a = copy.Fit(a).Transform(a);
            b = copy.Fit(b).Transform(b);
        }
        var expected = Trace(a);
        var actual = Trace(b);
        CollectionAssert.AreEqual(expected.ActiveColumns, actual.ActiveColumns);
        Assert.AreEqual(expected.Rows, actual.Rows);
        Assert.AreEqual(expected.RowOrderSha256, actual.RowOrderSha256);
        Assert.AreEqual(expected.FeatureGetterCalls, actual.FeatureGetterCalls);
        Assert.AreEqual(expected.FeatureBitsSha256, actual.FeatureBitsSha256);
        Assert.AreEqual(expected.LabelGetterCalls, actual.LabelGetterCalls);
        Assert.AreEqual(expected.LabelBitsSha256, actual.LabelBitsSha256);
        Assert.AreEqual(selected.Count, actual.Rows);
        Assert.IsTrue(actual.Complete);
        Assert.IsTrue(actual.RandomSupplied);
        Assert.IsFalse(actual.ActiveColumns.Contains("Text", StringComparer.Ordinal));

        CursorTrace Trace(IDataView source)
        {
            var trace = new DataViewTrace(source, ids);
            Assert.AreSame(source.Schema, trace.Schema);
            Assert.AreEqual(source.CanShuffle, trace.CanShuffle);
            Assert.AreEqual(source.GetRowCount(), trace.GetRowCount());
            var random = new Random(1729);
            var control = new Random(1729);
            using var cursor = trace.GetRowCursor([source.Schema[features], source.Schema["Label"]], random);
            var get = cursor.GetGetter<VBuffer<float>>(source.Schema[features]);
            var label = cursor.GetGetter<bool>(source.Schema["Label"]);
            VBuffer<float> vector = default; bool value = false;
            while (cursor.MoveNext()) { get(ref vector); label(ref value); }
            Assert.AreEqual(control.Next(), random.Next());
            return trace.Snapshot().Single();
        }
    }

    [TestMethod]
    public void DiagnosticTrace_RecordsPartialSetDisposalAndDoesNotActivateSemanticOrReadText()
    {
        using var data = StudyData.FromLegacy(StudyFixture.Authored());
        var selected = data.All();
        var view = selected.View();
        var trace = new DataViewTrace(view, selected.SourceIds());
        var cursors = trace.GetRowCursorSet([view.Schema["RowId"]], 4);
        Assert.AreEqual(1, cursors.Length);
        var cursor = cursors[0];
        Assert.IsFalse(cursor.IsColumnActive(view.Schema["Semantic"]));
        Assert.IsTrue(cursor.MoveNext());
        cursor.Dispose();
        Assert.AreEqual(0, data.ActiveCursors);
        var entry = trace.Snapshot().Single();
        Assert.AreEqual("set", entry.Method);
        Assert.AreEqual(4, entry.RequestedCount);
        Assert.AreEqual(1L, entry.Rows);
        Assert.IsFalse(entry.Complete);
        Assert.AreEqual(0L, entry.FeatureGetterCalls);
        Assert.AreEqual(0L, entry.LabelGetterCalls);
        CollectionAssert.AreEqual(new[] { "RowId" }, entry.ActiveColumns);
    }

    [TestMethod]
    [DataRow(1)]
    [DataRow(4)]
    [DataRow(16)]
    public void PredictionBuffer_PublicOutputCursorSetRestoresSelectionRankAndReusesColumns(int cursors)
    {
        var legacy = StudyFixture.Authored();
        using var data = StudyData.FromLegacy(legacy with { Rows = legacy.Rows.Reverse().ToArray() });
        var selected = data.All();
        var view = selected.View();
        var context = new MLContext(1);
        var model = context.Transforms.Conversion.ConvertType("Probability", "SpamBaseline", DataKind.Single).Fit(view);
        var buffer = new PredictionBuffer(selected.Count);
        var oracle = selected.SourceIds().Select(id => legacy.Rows.Single(r => r.RowId == id))
            .Select(r => new Prediction(r.RowId, r.GroupId, r.Label, (float)r.SpamBaseline)).ToArray();
        for (int pass = 0; pass < 3; pass++)
        {
            buffer.Fill(model, view, selected, requestedCursors: cursors);
            buffer.RequireReplay(oracle);
            Assert.AreEqual(selected.Count, buffer.Count);
            Assert.AreEqual(0, data.ActiveCursors);
        }
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() =>
            buffer.Fill(model, view, selected, requestedCursors: 17));
        Assert.AreEqual(0, buffer.Count);
    }

    [TestMethod]
    [DataRow("duplicate")]
    [DataRow("missing")]
    [DataRow("extra")]
    [DataRow("group")]
    [DataRow("label")]
    [DataRow("nonfinite")]
    public void PredictionBuffer_ParallelProfileRejectsInvalidUnionWithoutPublishingPartialRows(string mutation)
    {
        var legacy = StudyFixture.Authored();
        using var data = StudyData.FromLegacy(legacy);
        var selected = data.All();
        var rows = legacy.Rows.ToArray();
        if (mutation == "duplicate") rows = [.. rows, rows[0]];
        else if (mutation == "missing") rows = rows[..^1];
        else
        {
            var original = rows[0];
            rows[0] = new LearningRow
            {
                RowId = mutation == "extra" ? -9999 : original.RowId,
                GroupId = mutation == "group" ? -1 : original.GroupId,
                Label = mutation == "label" ? !original.Label : original.Label,
                Text = original.Text, Semantic = original.Semantic,
                SpamBaseline = mutation == "nonfinite" ? double.NaN : original.SpamBaseline
            };
        }
        var context = new MLContext(1);
        var input = context.Data.LoadFromEnumerable(rows);
        var model = context.Transforms.Conversion.ConvertType("Probability", "SpamBaseline", DataKind.Single).Fit(input);
        var buffer = new PredictionBuffer(selected.Count);
        Assert.ThrowsExactly<InvalidDataException>(() =>
            buffer.Fill(model, input, selected, requestedCursors: 4));
        Assert.AreEqual(0, buffer.Count);
        Assert.AreEqual(0, data.ActiveCursors);
    }
}
