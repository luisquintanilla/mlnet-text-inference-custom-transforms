using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Tokenizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLNet.TextInference.Onnx;

namespace MLNet.TextInference.TypedDecisions.Tests;

[TestClass]
public sealed class ConventionalRowMapperTests
{
    [TestMethod]
    public void MappedDataViewRegistersGettersBeforeMoveNextAndDisposesInputOnce()
    {
        var ml = new MLContext(seed: 1);
        var source = new DisposeCountingDataView(
            ml.Data.LoadFromEnumerable(
            [
                new PoolInputRow
                {
                    RawOutput = new VBuffer<float>(4, [1, 1, 3, 3]),
                    AttentionMask = new VBuffer<long>(2, [1, 1])
                },
                new PoolInputRow
                {
                    RawOutput = new VBuffer<float>(4, [2, 2, 4, 4]),
                    AttentionMask = new VBuffer<long>(2, [1, 1])
                },
                new PoolInputRow
                {
                    RawOutput = new VBuffer<float>(4, [3, 3, 5, 5]),
                    AttentionMask = new VBuffer<long>(2, [1, 1])
                }
            ]));
        var pooler = new EmbeddingPoolingTransformer(
            ml,
            new EmbeddingPoolingOptions
            {
                InputColumnName = nameof(PoolInputRow.RawOutput),
                AttentionMaskColumnName = nameof(PoolInputRow.AttentionMask),
                OutputColumnName = "Embedding",
                HiddenDim = 2,
                SequenceLength = 2,
                Pooling = PoolingStrategy.MeanPooling,
                Normalize = false
            });

        var output = pooler.Transform(source);
        var embedding = output.Schema["Embedding"];
        using (var cursor = output.GetRowCursor([embedding]))
        {
            var getter = cursor.GetGetter<VBuffer<float>>(embedding);
            var values = new List<float[]>();
            for (var index = 0; index < 3; index++)
            {
                Assert.IsTrue(cursor.MoveNext());
                VBuffer<float> value = default;
                getter(ref value);
                values.Add(value.DenseValues().ToArray());
            }

            CollectionAssert.AreEqual(new[] { 2f, 2f }, values[0]);
            CollectionAssert.AreEqual(new[] { 3f, 3f }, values[1]);
            CollectionAssert.AreEqual(new[] { 4f, 4f }, values[2]);
        }

        Assert.AreEqual(1, source.CursorDisposeCount);
    }

    [TestMethod]
    public void TextTokenizerMapperComputesLazyOutputsAndPreservesPassthrough()
    {
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n"));
        var tokenizer = BertTokenizer.Create(stream);
        var ml = new MLContext(seed: 1);
        var source = ml.Data.LoadFromEnumerable(
            [new InputRow { Text = "hello world", Label = "row-1" }]);
        var transformer = new TextTokenizerEstimator(
            ml,
            new TextTokenizerOptions
            {
                Tokenizer = tokenizer,
                MaxTokenLength = 6,
                OutputTokenTypeIds = false
            }).Fit(source);
        var mapper = transformer.GetRowToRowMapper(source.Schema);
        var tokenColumn = mapper.OutputSchema["TokenIds"];
        var labelColumn = mapper.OutputSchema["Label"];

        using var cursor = source.GetRowCursor(mapper.GetDependencies(
            [tokenColumn, labelColumn]));
        Assert.IsTrue(cursor.MoveNext());
        using var row = mapper.GetRow(cursor, [tokenColumn, labelColumn]);

        var tokenGetter = row.GetGetter<VBuffer<long>>(tokenColumn);
        VBuffer<long> first = default;
        tokenGetter(ref first);
        VBuffer<long> second = default;
        tokenGetter(ref second);

        Assert.AreEqual(6, first.Length);
        CollectionAssert.AreEqual(first.DenseValues().ToArray(), second.DenseValues().ToArray());

        var labelGetter = row.GetGetter<ReadOnlyMemory<char>>(labelColumn);
        ReadOnlyMemory<char> label = default;
        labelGetter(ref label);
        Assert.AreEqual("row-1", label.ToString());
        Assert.ThrowsExactly<InvalidOperationException>(() =>
            row.GetGetter<VBuffer<long>>(mapper.OutputSchema["AttentionMask"]));
    }

    [TestMethod]
    public void OnnxScorerMapperReadsConfiguredOutputsLazilyAndCachesGetters()
    {
        const string modelBase64 =
            "CAk66AEKJAoJaW5wdXRfaWRzEgZvdXRwdXQiBENhc3QqCQoCdG8YAaABAgouCg5hdHRlbnRpb25fbWFzaxILbWFza19vdXRwdXQiBENhc3QqCQoCdG8YAaABAhIEdGlueVogCglpbnB1dF9pZHMSEwoRCAcSDQoHEgViYXRjaAoCCARaJQoOYXR0ZW50aW9uX21hc2sSEwoRCAcSDQoHEgViYXRjaAoCCARiHQoGb3V0cHV0EhMKEQgBEg0KBxIFYmF0Y2gKAggEYiIKC21hc2tfb3V0cHV0EhMKEQgBEg0KBxIFYmF0Y2gKAggEQgQKABAN";
        var root = Directory.CreateTempSubdirectory("onnx-scorer-mapper-");
        try
        {
            var modelPath = Path.Combine(root.FullName, "model.onnx");
            File.WriteAllBytes(modelPath, Convert.FromBase64String(modelBase64));

            var ml = new MLContext(seed: 1);
            var rows = new[]
            {
                new TokenRow
                {
                    TokenIds = [1, 2, 3, 4],
                    AttentionMask = [1, 1, 0, 0],
                    Label = "first",
                    Features = new VBuffer<float>(5, 2, [10, 20], [1, 4])
                },
                new TokenRow
                {
                    TokenIds = [5, 6, 7, 8],
                    AttentionMask = [1, 1, 1, 0],
                    Label = "second",
                    Features = new VBuffer<float>(5, 2, [30, 40], [0, 3])
                }
            };
            var fitSource = ml.Data.LoadFromEnumerable(rows);
            var source = new TokenRowDataView(rows);
            using var transformer = new OnnxTextModelScorerEstimator(
                ml,
                new OnnxTextModelScorerOptions
                {
                    ModelPath = modelPath,
                    TokenTypeIdsColumnName = null,
                    MaxTokenLength = 4,
                    AdditionalOutputTensorNames = ["mask_output"],
                    AdditionalOutputColumnNames = ["MaskOutput"]
                }).Fit(fitSource);

            var mapper = transformer.GetRowToRowMapper(source.Schema);
            var raw = mapper.OutputSchema["RawOutput"];
            var mask = mapper.OutputSchema["MaskOutput"];
            using var cursor = source.GetRowCursor(mapper.GetDependencies(
                [raw, mask]));
            Assert.AreEqual(2L, source.GetRowCount());
            var dependencies = mapper.GetDependencies([raw, mask]).ToArray();
            CollectionAssert.AreEquivalent(
                new[] { "TokenIds", "AttentionMask" },
                dependencies.Select(column => column.Name).ToArray());
            CollectionAssert.AreEquivalent(new[] { 0, 1 }, dependencies.Select(column => column.Index).ToArray());
            Assert.IsTrue(ReferenceEquals(source.Schema, mapper.InputSchema));
            var expectedRaw = new[] { new[] { 1f, 2f, 3f, 4f }, new[] { 5f, 6f, 7f, 8f } };
            var expectedMask = new[] { new[] { 1f, 1f, 0f, 0f }, new[] { 1f, 1f, 1f, 0f } };

            using var mappedRow = mapper.GetRow(cursor, [raw, mask]);
            var rawGetter = mappedRow.GetGetter<VBuffer<float>>(raw);
            var maskGetter = mappedRow.GetGetter<VBuffer<float>>(mask);
            for (var index = 0; index < 2; index++)
            {
                Assert.IsTrue(cursor.MoveNext());
                VBuffer<float> rawValue = default;
                VBuffer<float> maskValue = default;
                rawGetter(ref rawValue);
                maskGetter(ref maskValue);
                CollectionAssert.AreEqual(expectedRaw[index], rawValue.DenseValues().ToArray());
                CollectionAssert.AreEqual(expectedMask[index], maskValue.DenseValues().ToArray());

                VBuffer<float> repeated = default;
                rawGetter(ref repeated);
                CollectionAssert.AreEqual(expectedRaw[index], repeated.DenseValues().ToArray());
            }
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    [TestMethod]
    public void OnnxScorerCursorPassesThroughRequestedColumnsWithoutInference()
    {
        const string modelBase64 =
            "CAk66AEKJAoJaW5wdXRfaWRzEgZvdXRwdXQiBENhc3QqCQoCdG8YAaABAgouCg5hdHRlbnRpb25fbWFzaxILbWFza19vdXRwdXQiBENhc3QqCQoCdG8YAaABAhIEdGlueVogCglpbnB1dF9pZHMSEwoRCAcSDQoHEgViYXRjaAoCCARaJQoOYXR0ZW50aW9uX21hc2sSEwoRCAcSDQoHEgViYXRjaAoCCARiHQoGb3V0cHV0EhMKEQgBEg0KBxIFYmF0Y2gKAggEYiIKC21hc2tfb3V0cHV0EhMKEQgBEg0KBxIFYmF0Y2gKAggEQgQKABAN";
        var root = Directory.CreateTempSubdirectory("onnx-scorer-passthrough-");
        try
        {
            var modelPath = Path.Combine(root.FullName, "model.onnx");
            File.WriteAllBytes(modelPath, Convert.FromBase64String(modelBase64));
            var ml = new MLContext(seed: 1);
            var rows = new[]
            {
                new TokenRow
                {
                    TokenIds = [1, 2, 3, 4],
                    AttentionMask = [1, 1, 0, 0],
                    Label = "first",
                    Features = new VBuffer<float>(5, 2, [10, 20], [1, 4])
                },
                new TokenRow
                {
                    TokenIds = [5, 6, 7, 8],
                    AttentionMask = [1, 1, 1, 0],
                    Label = "second",
                    Features = new VBuffer<float>(5, 2, [30, 40], [0, 3])
                }
            };
            var fitSource = ml.Data.LoadFromEnumerable(rows);
            var source = new TokenRowDataView(rows);
            using var transformer = new OnnxTextModelScorerEstimator(
                ml,
                new OnnxTextModelScorerOptions
                {
                    ModelPath = modelPath,
                    TokenTypeIdsColumnName = null,
                    MaxTokenLength = 4,
                    BatchSize = 2
                }).Fit(fitSource);

            var output = transformer.Transform(source);
            var label = output.Schema[nameof(TokenRow.Label)];
            var features = output.Schema[nameof(TokenRow.Features)];
            using var cursor = output.GetRowCursor([label, features]);
            var labelGetter = cursor.GetGetter<ReadOnlyMemory<char>>(label);
            var featuresGetter = cursor.GetGetter<VBuffer<float>>(features);

            for (int rowIndex = 0; rowIndex < rows.Length; rowIndex++)
            {
                Assert.IsTrue(cursor.MoveNext());
                ReadOnlyMemory<char> labelValue = default;
                VBuffer<float> featureValue = default;
                labelGetter(ref labelValue);
                featuresGetter(ref featureValue);
                Assert.AreEqual(rows[rowIndex].Label, labelValue.ToString());
                Assert.IsFalse(featureValue.IsDense);
                CollectionAssert.AreEqual(
                    rows[rowIndex].Features.GetIndices().ToArray(),
                    featureValue.GetIndices().ToArray());
                CollectionAssert.AreEqual(
                    rows[rowIndex].Features.GetValues().ToArray(),
                    featureValue.GetValues().ToArray());
            }
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    [TestMethod]
    public void OnnxScorerCursorPreservesIdsAndRejectsInactiveOrWrongTypeGetters()
        {
            const string modelBase64 =
                "CAk66AEKJAoJaW5wdXRfaWRzEgZvdXRwdXQiBENhc3QqCQoCdG8YAaABAgouCg5hdHRlbnRpb25fbWFzaxILbWFza19vdXRwdXQiBENhc3QqCQoCdG8YAaABAhIEdGlueVogCglpbnB1dF9pZHMSEwoRCAcSDQoHEgViYXRjaAoCCARaJQoOYXR0ZW50aW9uX21hc2sSEwoRCAcSDQoHEgViYXRjaAoCCARiHQoGb3V0cHV0EhMKEQgBEg0KBxIFYmF0Y2gKAggEYiIKC21hc2tfb3V0cHV0EhMKEQgBEg0KBxIFYmF0Y2gKAggEQgQKABAN";
            var root = Directory.CreateTempSubdirectory("onnx-scorer-cursor-contract-");
            try
            {
                var modelPath = Path.Combine(root.FullName, "model.onnx");
                File.WriteAllBytes(modelPath, Convert.FromBase64String(modelBase64));
                var ml = new MLContext(seed: 1);
                var rows = new[]
                {
                    new TokenRow { TokenIds = [1, 2, 3, 4], AttentionMask = [1, 1, 0, 0], Label = "first" },
                    new TokenRow { TokenIds = [5, 6, 7, 8], AttentionMask = [1, 1, 1, 0], Label = "second" },
                    new TokenRow { TokenIds = [9, 10, 11, 12], AttentionMask = [1, 1, 0, 0], Label = "third" },
                    new TokenRow { TokenIds = [13, 14, 15, 16], AttentionMask = [1, 1, 1, 0], Label = "fourth" },
                    new TokenRow { TokenIds = [17, 18, 19, 20], AttentionMask = [1, 1, 0, 0], Label = "fifth" }
                };
                var fitSource = ml.Data.LoadFromEnumerable(rows);
                using var transformer = new OnnxTextModelScorerEstimator(
                    ml,
                    new OnnxTextModelScorerOptions
                    {
                        ModelPath = modelPath,
                        TokenTypeIdsColumnName = null,
                        MaxTokenLength = 4,
                        BatchSize = 2
                    }).Fit(fitSource);

                var source = new TokenRowDataView(rows);
                var output = transformer.Transform(source);
                var label = output.Schema[nameof(TokenRow.Label)];
                var raw = output.Schema["RawOutput"];
                using var cursor = output.GetRowCursor([label]);

                Assert.ThrowsExactly<InvalidOperationException>(
                    () => cursor.GetGetter<VBuffer<float>>(raw));
                Assert.ThrowsExactly<InvalidOperationException>(
                    () => cursor.GetGetter<float>(label));

                var labelGetter = cursor.GetGetter<ReadOnlyMemory<char>>(label);
                var idGetter = cursor.GetIdGetter();
                for (var index = 0; index < rows.Length; index++)
                {
                    Assert.IsTrue(cursor.MoveNext());
                    Assert.AreEqual(index / 2, cursor.Batch);
                    DataViewRowId id = default;
                    idGetter(ref id);
                    Assert.AreEqual((ulong)index, id.Low);
                    ReadOnlyMemory<char> value = default;
                    labelGetter(ref value);
                    Assert.AreEqual(rows[index].Label, value.ToString());
                }
            }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    [TestMethod]
    public void ConventionalPostProcessingMappersExposeDirectResultsAndPreservePassthrough()
    {
        var ml = new MLContext(seed: 1);
        var source = ml.Data.LoadFromEnumerable(
            new[]
            {
                new PostProcessingRow
                {
                    RawOutput = new VBuffer<float>(6, [1, 2, 3, 4, 5, 6]),
                    AttentionMask = new VBuffer<long>(3, [1, 1, 0]),
                    NerOutput = new VBuffer<float>(
                        12,
                        [
                            0, 0, 0,
                            0, 5, 0,
                            0, 0, 5,
                            0, 0, 0
                        ]),
                    StartLogits = new VBuffer<float>(4, [0, 5, 0, 0]),
                    EndLogits = new VBuffer<float>(4, [0, 0, 5, 0]),
                    NerMask = new VBuffer<long>(4, [1, 1, 1, 1]),
                    QaMask = new VBuffer<long>(4, [1, 1, 1, 0]),
                    Starts = new VBuffer<long>(4, [0, 0, 0, 0]),
                    Ends = new VBuffer<long>(4, [0, 1, 3, 0]),
                    Text = "Bob",
                    Label = "preserved"
                }
            });

        var pooling = new EmbeddingPoolingTransformer(
            ml,
            new EmbeddingPoolingOptions
            {
                InputColumnName = nameof(PostProcessingRow.RawOutput),
                AttentionMaskColumnName = nameof(PostProcessingRow.AttentionMask),
                OutputColumnName = "Embedding",
                HiddenDim = 2,
                SequenceLength = 3,
                Pooling = PoolingStrategy.MeanPooling,
                Normalize = false
            });
        var poolingMapper = pooling.GetRowToRowMapper(source.Schema);
        var embedding = poolingMapper.OutputSchema["Embedding"];
        using (var cursor = source.GetRowCursor(poolingMapper.GetDependencies([embedding])))
        {
            Assert.IsTrue(cursor.MoveNext());
            using var row = poolingMapper.GetRow(cursor, [embedding]);
            var getter = row.GetGetter<VBuffer<float>>(embedding);
            VBuffer<float> value = default;
            getter(ref value);
            CollectionAssert.AreEqual(new[] { 2f, 3f }, value.DenseValues().ToArray());
        }
        Assert.IsFalse(poolingMapper.GetDependencies([embedding]).Any(
            column => column.Name == nameof(PostProcessingRow.Label)));

        var sigmoid = new SigmoidScorerTransformer(
            ml,
            new SigmoidScorerOptions
            {
                InputColumnName = nameof(PostProcessingRow.StartLogits),
                OutputColumnName = "Score"
            });
        var sigmoidMapper = sigmoid.GetRowToRowMapper(source.Schema);
        var score = sigmoidMapper.OutputSchema["Score"];
        using (var cursor = source.GetRowCursor(sigmoidMapper.GetDependencies([score])))
        {
            Assert.IsTrue(cursor.MoveNext());
            using var row = sigmoidMapper.GetRow(cursor, [score]);
            var getter = row.GetGetter<float>(score);
            float value = 0;
            getter(ref value);
            Assert.AreEqual(0.5f, value, 0.000001f);
        }

        var ner = new NerDecodingTransformer(
            ml,
            new NerDecodingOptions
            {
                InputColumnName = nameof(PostProcessingRow.NerOutput),
                AttentionMaskColumnName = nameof(PostProcessingRow.NerMask),
                TextColumnName = nameof(PostProcessingRow.Text),
                TokenStartOffsetsColumnName = nameof(PostProcessingRow.Starts),
                TokenEndOffsetsColumnName = nameof(PostProcessingRow.Ends),
                Labels = ["O", "B-PER", "I-PER"],
                NumLabels = 3,
                OutputColumnName = "Entities"
            });
        var nerMapper = ner.GetRowToRowMapper(source.Schema);
        var entities = nerMapper.OutputSchema["Entities"];
        using (var cursor = source.GetRowCursor(nerMapper.GetDependencies([entities])))
        {
            Assert.IsTrue(cursor.MoveNext());
            using var row = nerMapper.GetRow(cursor, [entities]);
            var getter = row.GetGetter<ReadOnlyMemory<char>>(entities);
            ReadOnlyMemory<char> value = default;
            getter(ref value);
            StringAssert.Contains(value.ToString(), "\"entity\":\"PER\"");
            StringAssert.Contains(value.ToString(), "\"word\":\"Bob\"");
        }

        var qa = new QaSpanExtractionTransformer(
            ml,
            new QaSpanExtractionOptions
            {
                StartLogitsColumnName = nameof(PostProcessingRow.StartLogits),
                EndLogitsColumnName = nameof(PostProcessingRow.EndLogits),
                AttentionMaskColumnName = nameof(PostProcessingRow.QaMask),
                TextColumnName = nameof(PostProcessingRow.Text),
                TokenStartOffsetsColumnName = nameof(PostProcessingRow.Starts),
                TokenEndOffsetsColumnName = nameof(PostProcessingRow.Ends),
                OutputColumnName = "Answer",
                ScoreColumnName = "AnswerScore"
            });
        var qaMapper = qa.GetRowToRowMapper(source.Schema);
        var answer = qaMapper.OutputSchema["Answer"];
        var answerScore = qaMapper.OutputSchema["AnswerScore"];
        using (var cursor = source.GetRowCursor(qaMapper.GetDependencies([answer, answerScore])))
        {
            Assert.IsTrue(cursor.MoveNext());
            using var row = qaMapper.GetRow(cursor, [answer, answerScore]);
            var answerGetter = row.GetGetter<ReadOnlyMemory<char>>(answer);
            var scoreGetter = row.GetGetter<float>(answerScore);
            ReadOnlyMemory<char> answerValue = default;
            float scoreValue = 0;
            answerGetter(ref answerValue);
            scoreGetter(ref scoreValue);
            Assert.AreEqual("Bob", answerValue.ToString());
            Assert.AreEqual(10f, scoreValue);
        }
    }

    [TestMethod]
    public void RankThreeScorerUsesSequenceTimesHiddenWidthAndFacadeMapperPreservesIntermediates()
    {
        const string modelBase64 =
            "CAgSFHR5cGVkLWRlY2lzaW9uLXRlc3RzOtECCikKDmF0dGVudGlvbl9tYXNrEg1tYXNrX2lkZW50aXR5IghJZGVudGl0eQoiCglpbnB1dF9pZHMSBGNhc3QiBENhc3QqCQoCdG8YAaABAgohCgRjYXN0CgRheGVzEghleHBhbmRlZCIJVW5zcXVlZXplCiwKCGV4cGFuZGVkCgdyZXBlYXRzEhFsYXN0X2hpZGRlbl9zdGF0ZSIEVGlsZRIFcmFuazMqFAgBEAdCBGF4ZXNKCAIAAAAAAAAAKicIAxAHQgdyZXBlYXRzShgBAAAAAAAAAAEAAAAAAAAAAwAAAAAAAABaHAoJaW5wdXRfaWRzEg8KDQgHEgkKAxIBQgoCCARaIQoOYXR0ZW50aW9uX21hc2sSDwoNCAcSCQoDEgFCCgIIBGIoChFsYXN0X2hpZGRlbl9zdGF0ZRITChEIARINCgMSAUIKAggECgIIA0IECgAQDQ==";
        var root = Directory.CreateTempSubdirectory("onnx-rank3-");
        try
        {
            var modelPath = Path.Combine(root.FullName, "rank3.onnx");
            File.WriteAllBytes(modelPath, Convert.FromBase64String(modelBase64));

            var ml = new MLContext(seed: 1);
            var rows = new[]
            {
                new TokenRow { TokenIds = [1, 2, 3, 4], AttentionMask = [1, 1, 1, 1], Label = "first" },
                new TokenRow { TokenIds = [5, 6, 7, 8], AttentionMask = [1, 1, 1, 1], Label = "second" }
            };
            var source = ml.Data.LoadFromEnumerable(rows);
            using var scorer = new OnnxTextModelScorerEstimator(
                ml,
                new OnnxTextModelScorerOptions
                {
                    ModelPath = modelPath,
                    TokenTypeIdsColumnName = null,
                    MaxTokenLength = 4,
                    BatchSize = 2,
                    OutputTensorName = "last_hidden_state"
                }).Fit(source);

            Assert.AreEqual(3, scorer.HiddenDim);
            Assert.IsFalse(scorer.HasPooledOutput);
            var scored = scorer.Transform(source);
            var raw = scored.Schema["RawOutput"];
            using (var cursor = scored.GetRowCursor([raw]))
            {
                var getter = cursor.GetGetter<VBuffer<float>>(raw);
                var expected = new[]
                {
                    new[] { 1f, 1f, 1f, 2f, 2f, 2f, 3f, 3f, 3f, 4f, 4f, 4f },
                    new[] { 5f, 5f, 5f, 6f, 6f, 6f, 7f, 7f, 7f, 8f, 8f, 8f }
                };
                for (var rowIndex = 0; rowIndex < expected.Length; rowIndex++)
                {
                    Assert.IsTrue(cursor.MoveNext());
                    VBuffer<float> value = default;
                    getter(ref value);
                    CollectionAssert.AreEqual(expected[rowIndex], value.DenseValues().ToArray());
                }
            }

            var vocabPath = Path.Combine(root.FullName, "vocab.txt");
            File.WriteAllText(vocabPath, "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n");
            using var embedding = new OnnxTextEmbeddingEstimator(
                ml,
                new OnnxTextEmbeddingOptions
                {
                    ModelPath = modelPath,
                    TokenizerPath = vocabPath,
                    InputColumnName = nameof(EmbeddingInput.Text),
                    OutputColumnName = "Embedding",
                    MaxTokenLength = 4,
                    Pooling = PoolingStrategy.MeanPooling,
                    Normalize = false,
                    BatchSize = 2
                }).Fit(ml.Data.LoadFromEnumerable(
                    [new EmbeddingInput { Text = "hello world" }]));

            var embeddingOutput = embedding.Transform(
                ml.Data.LoadFromEnumerable(
                [
                    new EmbeddingInput { Text = "hello world" },
                    new EmbeddingInput { Text = "hello" },
                    new EmbeddingInput { Text = "world" }
                ]));
            CollectionAssert.Contains(
                embeddingOutput.Schema.Select(column => column.Name).ToArray(),
                "TokenIds");
            CollectionAssert.Contains(
                embeddingOutput.Schema.Select(column => column.Name).ToArray(),
                "RawOutput");
            var embeddingColumn = embeddingOutput.Schema["Embedding"];
            using (var cursor = embeddingOutput.GetRowCursor([embeddingColumn]))
            {
                var getter = cursor.GetGetter<VBuffer<float>>(embeddingColumn);
                var values = new List<float[]>();
                for (var rowIndex = 0; rowIndex < 3; rowIndex++)
                {
                    Assert.IsTrue(cursor.MoveNext());
                    VBuffer<float> value = default;
                    getter(ref value);
                    values.Add(value.DenseValues().ToArray());
                }

                Assert.AreEqual(3, values.Count);
                Assert.AreEqual(3, values[0].Length);
                Assert.AreEqual(3, values[1].Length);
                Assert.AreEqual(3, values[2].Length);
            }
            var enumerableRows = ml.Data.CreateEnumerable<EmbeddingOutput>(
                embeddingOutput,
                reuseRowObject: false).ToArray();
            Assert.AreEqual(3, enumerableRows.Length);
            Assert.IsTrue(enumerableRows.All(row => row.Embedding.Length == 3));
            using var engine = ml.Model.CreatePredictionEngine<EmbeddingInput, EmbeddingOutput>(
                embedding,
                new PredictionEngineOptions { OwnsTransformer = false });
            var result = engine.Predict(new EmbeddingInput { Text = "hello world" });
            Assert.AreEqual(3, result.Embedding.Length);
            Assert.AreEqual(12, result.RawOutput.Length);
            CollectionAssert.AreEqual(
                new[] { 5.5f, 5.5f, 5.5f },
                result.Embedding);
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    private sealed class InputRow
    {
        public string Text { get; set; } = string.Empty;
        public string Label { get; set; } = string.Empty;
    }

    private sealed class TokenRow
    {
        public long[] TokenIds { get; set; } = [];
        public long[] AttentionMask { get; set; } = [];
        public string Label { get; set; } = string.Empty;
        public VBuffer<float> Features { get; set; }
    }

    private sealed class EmbeddingInput
    {
        public string Text { get; set; } = string.Empty;
    }

    private sealed class EmbeddingOutput
    {
        public float[] Embedding { get; set; } = [];
        public float[] RawOutput { get; set; } = [];
    }

    private sealed class PoolInputRow
    {
        public VBuffer<float> RawOutput { get; set; }
        public VBuffer<long> AttentionMask { get; set; }
    }

    private sealed class DisposeCountingDataView : IDataView
    {
        private readonly IDataView _inner;

        internal DisposeCountingDataView(IDataView inner)
        {
            _inner = inner;
        }

        internal int CursorDisposeCount { get; private set; }
        public DataViewSchema Schema => _inner.Schema;
        public bool CanShuffle => _inner.CanShuffle;
        public long? GetRowCount() => _inner.GetRowCount();

        public DataViewRowCursor GetRowCursor(
            IEnumerable<DataViewSchema.Column> columnsNeeded,
            Random? rand = null)
            => new CountingCursor(this, _inner.GetRowCursor(columnsNeeded, rand));

        public DataViewRowCursor[] GetRowCursorSet(
            IEnumerable<DataViewSchema.Column> columnsNeeded,
            int n,
            Random? rand = null)
            => [GetRowCursor(columnsNeeded, rand)];

        private sealed class CountingCursor : DataViewRowCursor
        {
            private readonly DisposeCountingDataView _parent;
            private readonly DataViewRowCursor _inner;
            private bool _disposed;

            internal CountingCursor(
                DisposeCountingDataView parent,
                DataViewRowCursor inner)
            {
                _parent = parent;
                _inner = inner;
            }

            public override DataViewSchema Schema => _inner.Schema;
            public override long Position => _inner.Position;
            public override long Batch => _inner.Batch;
            public override bool MoveNext() => _inner.MoveNext();
            public override ValueGetter<TValue> GetGetter<TValue>(
                DataViewSchema.Column column)
                => _inner.GetGetter<TValue>(column);
            public override ValueGetter<DataViewRowId> GetIdGetter()
                => _inner.GetIdGetter();
            public override bool IsColumnActive(DataViewSchema.Column column)
                => _inner.IsColumnActive(column);

            protected override void Dispose(bool disposing)
            {
                if (disposing && !_disposed)
                {
                    _disposed = true;
                    _parent.CursorDisposeCount++;
                    _inner.Dispose();
                }

                base.Dispose(disposing);
            }
        }
    }

    private sealed class TokenRowDataView : IDataView
    {
        private readonly TokenRow[] _rows;

        public TokenRowDataView(TokenRow[] rows)
        {
            _rows = rows;
            var builder = new DataViewSchema.Builder();
            builder.AddColumn(
                nameof(TokenRow.TokenIds),
                new VectorDataViewType(NumberDataViewType.Int64, 4));
            builder.AddColumn(
                nameof(TokenRow.AttentionMask),
                new VectorDataViewType(NumberDataViewType.Int64, 4));
            builder.AddColumn(nameof(TokenRow.Label), TextDataViewType.Instance);
            builder.AddColumn(
                nameof(TokenRow.Features),
                new VectorDataViewType(NumberDataViewType.Single));
            Schema = builder.ToSchema();
        }

        public DataViewSchema Schema { get; }
        public bool CanShuffle => false;
        public long? GetRowCount() => _rows.LongLength;

        public DataViewRowCursor GetRowCursor(
            IEnumerable<DataViewSchema.Column> columnsNeeded,
            Random? rand = null)
            => new Cursor(this, columnsNeeded);

        public DataViewRowCursor[] GetRowCursorSet(
            IEnumerable<DataViewSchema.Column> columnsNeeded,
            int n,
            Random? rand = null)
            => [GetRowCursor(columnsNeeded, rand)];

        private sealed class Cursor : DataViewRowCursor
        {
            private readonly TokenRowDataView _parent;
            private readonly HashSet<string> _activeColumns;
            private int _index = -1;

            public Cursor(
                TokenRowDataView parent,
                IEnumerable<DataViewSchema.Column> columnsNeeded)
            {
                _parent = parent;
                _activeColumns = columnsNeeded
                    .Select(static column => column.Name)
                    .ToHashSet(StringComparer.Ordinal);
            }

            public override DataViewSchema Schema => _parent.Schema;
            public override long Position => _index;
            public override long Batch => Math.Max(0, _index / 2);

            public override bool MoveNext()
            {
                _index++;
                return _index < _parent._rows.Length;
            }

            public override ValueGetter<TValue> GetGetter<TValue>(
                DataViewSchema.Column column)
                => column.Name switch
                {
                    nameof(TokenRow.TokenIds) => Cast<VBuffer<long>, TValue>(
                        (ref VBuffer<long> value) =>
                            value = new VBuffer<long>(
                                _parent._rows[_index].TokenIds.Length,
                                _parent._rows[_index].TokenIds)),
                    nameof(TokenRow.AttentionMask) => Cast<VBuffer<long>, TValue>(
                        (ref VBuffer<long> value) =>
                            value = new VBuffer<long>(
                                _parent._rows[_index].AttentionMask.Length,
                                _parent._rows[_index].AttentionMask)),
                    nameof(TokenRow.Label) => Cast<ReadOnlyMemory<char>, TValue>(
                        (ref ReadOnlyMemory<char> value) =>
                            value = _parent._rows[_index].Label.AsMemory()),
                    nameof(TokenRow.Features) => Cast<VBuffer<float>, TValue>(
                        (ref VBuffer<float> value) =>
                            value = _parent._rows[_index].Features),
                    _ => throw new InvalidOperationException(
                        $"Unknown column '{column.Name}'.")
                };

            public override ValueGetter<DataViewRowId> GetIdGetter()
                => (ref DataViewRowId value) =>
                    value = new DataViewRowId((ulong)_index, 0);

            public override bool IsColumnActive(DataViewSchema.Column column)
                => _activeColumns.Contains(column.Name);

            private static ValueGetter<TValue> Cast<TSource, TValue>(
                ValueGetter<TSource> getter)
                => (ValueGetter<TValue>)(object)getter;
        }
    }

    private sealed class PostProcessingRow
    {
        public VBuffer<float> RawOutput { get; set; }
        public VBuffer<long> AttentionMask { get; set; }
        public VBuffer<float> NerOutput { get; set; }
        public VBuffer<float> StartLogits { get; set; }
        public VBuffer<float> EndLogits { get; set; }
        public VBuffer<long> NerMask { get; set; }
        public VBuffer<long> QaMask { get; set; }
        public VBuffer<long> Starts { get; set; }
        public VBuffer<long> Ends { get; set; }
        public string Text { get; set; } = string.Empty;
        public string Label { get; set; } = string.Empty;
    }
}
