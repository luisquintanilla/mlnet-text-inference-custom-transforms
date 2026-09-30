using Apache.Arrow;
using Apache.Arrow.Types;
using System.Runtime.InteropServices;

namespace DecisionArrowPredictor;

// Borrowed batch bindings are used synchronously and never retained by StudyData.
public sealed class ProbabilityBatchAccessor
{
    private readonly Int64Array ids;
    private readonly DoubleArray commercial, contact, spam;
    private readonly StructArray pressure, purpose;
    private readonly ArrayData time, intent, timeValues, purposeValues;
    public int Count { get; }

    public ProbabilityBatchAccessor(RecordBatch batch)
    {
        if (batch.ColumnCount != 6 || batch.Column(0) is not Int64Array idColumn ||
            batch.Column(1) is not DoubleArray commercialColumn || batch.Column(2) is not DoubleArray contactColumn ||
            batch.Column(3) is not StructArray pressureColumn || batch.Column(4) is not StructArray purposeColumn ||
            batch.Column(5) is not DoubleArray spamColumn || pressureColumn.Data.Children.Length != 5 ||
            purposeColumn.Data.Children.Length != 2)
            throw new InvalidDataException("Expected producer-validated v1 five-question probability layout.");
        ids = idColumn; commercial = commercialColumn; contact = contactColumn;
        pressure = pressureColumn; purpose = purposeColumn; spam = spamColumn;
        time = pressure.Data.Children[4]; intent = purpose.Data.Children[1];
        if (time.DataType is not FixedSizeListType { ListSize: 3 } ||
            intent.DataType is not FixedSizeListType { ListSize: 5 } ||
            time.Children.Length != 1 || intent.Children.Length != 1 ||
            time.Children[0].DataType is not DoubleType || intent.Children[0].DataType is not DoubleType ||
            time.Children[0].Buffers.Length != 2 || intent.Children[0].Buffers.Length != 2)
            throw new InvalidDataException("Malformed nested Double probability vector layout.");
        timeValues = time.Children[0]; purposeValues = intent.Children[0];
        Count = batch.Length;
        if (ids.Length != Count || commercial.Length != Count || contact.Length != Count ||
            pressure.Length != Count || purpose.Length != Count || spam.Length != Count ||
            checked(pressure.Offset + Count) > time.Length || checked(purpose.Offset + Count) > intent.Length)
            throw new InvalidDataException("Probability batch column lengths differ.");
    }

    public long RowId(int row)
    {
        CheckRow(row);
        return ids.GetValue(row) ?? throw new InvalidDataException("Null Arrow source ID.");
    }

    public double CopyRow(int row, Span<float> destination)
    {
        CheckRow(row);
        if (destination.Length != FeatureContract.Width) throw new ArgumentException("Expected ten destination coordinates.", nameof(destination));
        int timeRow = checked(pressure.Offset + row);
        int purposeRow = checked(purpose.Offset + row);
        if (pressure.IsNull(row) || purpose.IsNull(row) || !IsValid(time, timeRow) || !IsValid(intent, purposeRow))
            throw new InvalidDataException("Null decision struct/probability list.");
        destination[0] = Convert(Number(commercial, row));
        destination[1] = Convert(Number(contact, row));
        // Match Struct.Fields' parent slicing without constructing its unrelated field wrappers.
        int timeStart = checked((time.Offset + timeRow) * 3);
        int purposeStart = checked((intent.Offset + purposeRow) * 5);
        for (int j = 0; j < 3; j++) destination[2 + j] = Convert(Number(timeValues, checked(timeStart + j)));
        for (int j = 0; j < 5; j++) destination[5 + j] = Convert(Number(purposeValues, checked(purposeStart + j)));
        double baseline = Number(spam, row);
        ProbabilityStore.RequireProbability(baseline);
        return baseline;
    }

    private void CheckRow(int row)
    {
        if ((uint)row >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(row));
    }

    private static double Number(DoubleArray array, int row) => array.GetValue(row) ??
        throw new InvalidDataException("Null probability in completed decision dataset.");

    private static bool IsValid(ArrayData data, int row)
    {
        if ((uint)row >= (uint)data.Length) throw new InvalidDataException("Nested Arrow row offset is out of bounds.");
        var bitmap = data.Buffers[0].Span;
        if (data.NullCount == 0 || bitmap.IsEmpty) return true;
        int index = checked(data.Offset + row);
        if ((uint)(index >> 3) >= (uint)bitmap.Length)
            throw new InvalidDataException("Nested Arrow validity offset is out of bounds.");
        return (bitmap[index >> 3] & (1 << (index & 7))) != 0;
    }

    private static double Number(ArrayData data, int row)
    {
        if (!IsValid(data, row)) throw new InvalidDataException("Null nested probability in completed decision dataset.");
        var values = MemoryMarshal.Cast<byte, double>(data.Buffers[1].Span);
        int index = checked(data.Offset + row);
        if ((uint)index >= (uint)values.Length)
            throw new InvalidDataException("Nested Arrow probability offset is out of bounds.");
        return values[index];
    }

    private static float Convert(double value)
    {
        ProbabilityStore.RequireProbability(value);
        return checked((float)value);
    }
}
