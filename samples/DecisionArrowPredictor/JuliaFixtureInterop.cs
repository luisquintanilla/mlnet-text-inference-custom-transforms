using DecisionInference.Arrow;
using System.Runtime.CompilerServices;

namespace DecisionArrowPredictor;

public static class JuliaFixtureInterop
{
    public const string ProducerReceiptSha256 = "1f237da39bfe74542d74dfbdb82f744efb3ba1bc96fb60e1426f311721f849fd";
    public const string HighPrecisionFingerprint = "78ed3dc0344f71f1d8515f5923dc2c1de3604e59e6b4a1c9cb89851e2b3cf05d";
    public const string FictionalMode = "julia-offline-fixture-not-real";

    private sealed record Fixture(string Precision, string Fingerprint, long ContractBytes, long ManifestBytes,
        string ManifestSha256, long DataBytes, string DataSha256);
    private static readonly Fixture[] Fixtures =
    [
        new("HighPrecision", HighPrecisionFingerprint, 2448, 4538,
            "dec32a01975d2a007ef4727e9bc586b22ce2435259e87c2c8f4658666c4254c7", 41936,
            "f2f0371297618f09f252fc3ff47c5aad5feb0dd2a7dfb68962b54acfe5748633"),
        new("FourDecimalPlaces", "622832ce70b7928e5cc5b015acd1f9e8b1ea3b621a78e6976d62cb4d2af32d62", 2452, 4540,
            "ceb6b192c028649ff37496bee82e7d0145cdd7f7784883261f9c39365ff586af", 41960,
            "a948845f156c781d5cf303cdd650f55f6bb554ad51cd01487ca00fc8261395e1"),
        new("TwoDecimalPlaces", "2a19f1e2e6933924c4d4cb7b8df99bd8d5d58ce6689d8b1e9fa1ce727e737723", 2451, 4540,
            "44aa0ec2c466ca3c7718ca3b99db5266040f4ae9f2cd68652edb2d9aa538e30e", 41952,
            "9a78ff7d1214f6aca5e59f2f9b275ac83b7b09501ad883fe5778387519db601d")
    ];

    public static async Task VerifyAsync(string root, string producerReceipt, string questions, string output,
        CancellationToken token = default)
    {
        if (File.Exists(output)) throw new IOException("Interop receipts are immutable.");
        ArtifactFiles.RequireHash(producerReceipt, ProducerReceiptSha256);
        RequireFile(Path.Combine(root, "julia-contract.json"), 2448, HighPrecisionFingerprint);
        RequireFile(Path.Combine(root, "profile.json"), 207,
            "0fd0708c71579bb79f659a23c3ac7808247ddc58c08e3525074dc209647c028b");
        var checks = new List<object>();
        foreach (var fixture in Fixtures)
        {
            string directory = Path.Combine(root, fixture.Precision);
            string contractPath = Path.Combine(directory, "contract.json");
            string manifestPath = Path.Combine(directory, "manifest.json");
            RequireFile(contractPath, fixture.ContractBytes, fixture.Fingerprint);
            RequireFile(manifestPath, fixture.ManifestBytes, fixture.ManifestSha256);
            RequireFile(Path.Combine(directory, "decisions.arrow"), fixture.DataBytes, fixture.DataSha256);
            var contract = ArrowFeatureReader.ExpectedContract(contractPath, fixture.Fingerprint, questions);
            int rows = 0, batches = 0;
            using (var store = new ProbabilityStore(257))
            using (var reader = await DecisionArrowDatasetReader.OpenAsync(manifestPath, contract,
                Enumerable.Range(1, 257).Select(i => (long)i), token))
            {
                if (reader.Manifest.Provenance.ExecutionMode != FictionalMode ||
                    reader.Manifest.Provenance.QuestionsSha256 != FeatureContract.QuestionsV1Sha256 ||
                    reader.Manifest.RowCount != 257 || reader.Manifest.OutputBatchSize != 256)
                    throw new InvalidDataException("Expected independently pinned fictional Julia fixture, not real data.");
                Apache.Arrow.RecordBatch? batch;
                while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
                {
                    using (batch)
                    {
                        if (batch.Length != (batches == 0 ? 256 : 1) || batches > 1)
                            throw new InvalidDataException("Julia fixture batch boundary differs from independent expectation.");
                        var legacy = ArrowFeatureReader.Project(batch);
                        var accessor = new ProbabilityBatchAccessor(batch);
                        for (int row = 0; row < batch.Length; row++)
                        {
                            int ordinal = checked(rows + row);
                            if (accessor.RowId(row) != ordinal + 1L || legacy[row].RowId != ordinal + 1L)
                                throw new InvalidDataException("Julia fixture IDs/order differ from independently fixed 1..257.");
                            var destination = store.WritableSemantic(ordinal);
                            double direct = accessor.CopyRow(row, destination);
                            store.SetDirect(ordinal, direct);
                            if (BitConverter.DoubleToInt64Bits(direct) != BitConverter.DoubleToInt64Bits(legacy[row].SpamBaseline))
                                throw new InvalidDataException("Julia direct Double bits differ from legacy projection.");
                            for (int column = 0; column < FeatureContract.Width; column++)
                                if (BitConverter.SingleToInt32Bits(destination[column]) !=
                                    BitConverter.SingleToInt32Bits(legacy[row].Semantic[column]))
                                    throw new InvalidDataException("Julia feature Single bits/order differ from legacy projection.");
                        }
                        rows = checked(rows + batch.Length);
                        batches++;
                    }
                }
                if (rows != 257 || batches != 2)
                    throw new InvalidDataException("Julia fixture full EOS/count validation failed.");
            }
            string roundtripDirectory = Path.Combine(output + ".roundtrip", fixture.Precision);
            var incoming = ArtifactFiles.Read<DecisionArrowManifest>(manifestPath);
            var written = await DecisionArrowWriter.WriteBatchesAsync(roundtripDirectory, contract,
                OwnedBatches(manifestPath, contract, token), incoming.Provenance with { Measurements = null }, 256, token);
            RequireFile(Path.Combine(roundtripDirectory, "decisions.arrow"), fixture.DataBytes, fixture.DataSha256);
            if (written.DataSha256 != fixture.DataSha256 || written.FeatureFingerprint != fixture.Fingerprint ||
                written.RowCount != 257)
                throw new InvalidDataException("Package-only standard-batch roundtrip changed fixture bytes/identity/count.");
            int rereadRows = 0;
            using (var reread = await DecisionArrowDatasetReader.OpenAsync(
                Path.Combine(roundtripDirectory, "manifest.json"), contract, Enumerable.Range(1, 257).Select(i => (long)i), token))
            {
                Apache.Arrow.RecordBatch? batch;
                while ((batch = await reread.ReadNextRecordBatchAsync(token)) is not null)
                    using (batch) rereadRows = checked(rereadRows + batch.Length);
            }
            if (rereadRows != 257) throw new InvalidDataException("Roundtrip public-reader full EOS validation failed.");
            checks.Add(new { fixture.Precision, fixture.Fingerprint, rows, batches, exactProjectionBits = true,
                fullPublicReaderEos = true, arrowLeasesAfterImport = 0, ownedStandardBatchWriterRoundtripByteExact = true });
        }
        token.ThrowIfCancellationRequested();
        ArtifactFiles.Write(output, new
        {
            schemaVersion = 1, status = "FICTIONAL_JULIA_INTEROP_PASS",
            producerReceiptSha256 = ProducerReceiptSha256,
            adapterAssemblySha256 = ArtifactFiles.Hash(typeof(DecisionArrowDatasetReader).Assembly.Location),
            sourceAssemblySha256 = ArtifactFiles.Hash(typeof(JuliaFixtureInterop).Assembly.Location),
            expectedIds = "Independently producer-fixed 1..257, not inferred from incoming batches.",
            mode = FictionalMode, inferenceOccurred = false, realJuliaImportEnabled = false,
            finalPackageGate = "Reader and owned standard-batch writer are exercised through package-only public APIs; dependency closure verification remains a separate receipt.",
            checks
        });
    }

    private static async IAsyncEnumerable<Apache.Arrow.RecordBatch> OwnedBatches(string manifest,
        DecisionArrowSchema contract, [EnumeratorCancellation] CancellationToken token)
    {
        using var reader = await DecisionArrowDatasetReader.OpenAsync(manifest, contract,
            Enumerable.Range(1, 257).Select(i => (long)i), token);
        Apache.Arrow.RecordBatch? batch;
        while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
            yield return batch; // Public writer owns each yielded batch; the iterator owns only its reader.
    }

    private static void RequireFile(string path, long bytes, string sha256)
    {
        if (new FileInfo(path).Length != bytes) throw new InvalidDataException("Julia fixture size differs from independent pin.");
        ArtifactFiles.RequireHash(path, sha256);
    }
}
