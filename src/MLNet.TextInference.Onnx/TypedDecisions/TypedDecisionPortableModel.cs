using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.TextInference.TypedDecisions;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Saves and loads typed-decision transformers and the explicitly supported
/// typed-decision pipeline compositions as portable local ZIP artifacts.
/// </summary>
/// <remarks>
/// This is intentionally separate from <see cref="MLContext.Model"/> native
/// persistence. The archive contains configuration and local model assets;
/// the load context creates new tokenizer and ONNX Runtime resources.
/// </remarks>
public static class TypedDecisionPortableModel
{
    internal const int CurrentFormatVersion = 1;
    internal const string PortableManifestFileName = "typed-decision-portable.json";

    /// <summary>Saves one supported typed-decision transformer to a portable ZIP artifact.</summary>
    public static void Save(ITransformer transformer, string path)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        if (transformer is TransformerChain<ITransformer> chain)
        {
            SavePipelineCore(
                ValidatePipelineScopes(chain),
                path);
            return;
        }

        var descriptor = CreateDescriptor(transformer);
        SaveArtifact(
            CreateArtifact(descriptor),
            [GetAssetsPath(transformer)],
            GetAssetRequirements(descriptor.Kind),
            path);
    }

    /// <summary>Loads a supported typed-decision transformer from a portable ZIP artifact.</summary>
    public static ITransformer Load(MLContext mlContext, string path)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        var artifact = ReadArtifact(path, expectedKind: null, out var extractedRoot);
        try
        {
            if (artifact.Kind == TypedDecisionPortableKinds.Pipeline)
                throw new InvalidDataException(
                    "The portable artifact contains a pipeline. Use LoadPipeline for pipeline artifacts.");

            var lease = new TypedDecisionRootLease(extractedRoot!);
            ITransformer transformer;
            try
            {
                transformer = CreateTransformer(mlContext, artifact, extractedRoot!, lease);
            }
            catch
            {
                lease.Release();
                throw;
            }
            lease.Release();
            extractedRoot = null;
            return transformer;
        }
        finally
        {
            DeleteOwnedRoot(extractedRoot);
        }
    }

    /// <summary>
    /// Saves the demonstrated typed-decision flat chain compositions without
    /// changing ML.NET transformer scopes.
    /// </summary>
    public static void SavePipeline<TLastTransformer>(
        TransformerChain<TLastTransformer> pipeline,
        string path)
        where TLastTransformer : class, ITransformer
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        var transformers = ValidatePipelineScopes(pipeline);
        if (transformers.Count == 0)
            throw new InvalidDataException("An empty typed-decision pipeline is not supported.");

        SavePipelineCore(transformers, path);
    }

    /// <summary>
    /// Saves a supported typed-decision chain when the concrete final transformer
    /// type is known at the call site.
    /// </summary>
    public static void Save<TLastTransformer>(
        TransformerChain<TLastTransformer> pipeline,
        string path)
        where TLastTransformer : class, ITransformer
        => SavePipeline(pipeline, path);

    private static void SavePipelineCore(
        IReadOnlyList<ITransformer> transformers,
        string path)
    {
        var descriptors = new List<PortableTransformerDescriptor>(transformers.Count);
        var assetPaths = new List<string>(transformers.Count);
        var requirements = TypedDecisionBundleLoadRequirements.None;
        foreach (var transformer in transformers)
        {
            var descriptor = CreateDescriptor(transformer);
            descriptors.Add(descriptor);
            assetPaths.Add(GetAssetsPath(transformer));
            requirements |= GetAssetRequirements(descriptor.Kind);
        }

        SaveArtifact(
            new PortableArtifact
            {
                Kind = TypedDecisionPortableKinds.Pipeline,
                Pipeline = new PortablePipeline
                {
                    Transformers = descriptors.ToArray()
                }
            },
            assetPaths,
            requirements,
            path);
    }

    /// <summary>Loads a supported typed-decision flat chain from a portable artifact.</summary>
    public static TransformerChain<ITransformer> LoadPipeline(MLContext mlContext, string path)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        var artifact = ReadArtifact(
            path,
            TypedDecisionPortableKinds.Pipeline,
            out var extractedRoot);
        try
        {
            var descriptors = artifact.Pipeline?.Transformers
                ?? throw new InvalidDataException("The portable pipeline has no transformer descriptors.");
            if (descriptors.Length == 0)
                throw new InvalidDataException("The portable pipeline must contain at least one transformer.");

            var lease = new TypedDecisionRootLease(extractedRoot!);
            var loaded = new ITransformer[descriptors.Length];
            try
            {
                for (var index = 0; index < descriptors.Length; index++)
                {
                    var descriptor = descriptors[index]
                        ?? throw new InvalidDataException(
                            $"Portable pipeline transformer descriptor {index} is null.");
                    var componentArtifact = artifact with
                    {
                        Kind = descriptor.Kind,
                        Facade = descriptor.Facade,
                        Preparation = descriptor.Preparation,
                        Scoring = descriptor.Scoring,
                        Decoding = descriptor.Decoding,
                        Pipeline = null
                    };
                    loaded[index] = CreateTransformer(
                        mlContext,
                        componentArtifact,
                        extractedRoot!,
                        lease);
                }

                lease.Release();
                extractedRoot = null;
                return new TransformerChain<ITransformer>(loaded);
            }
            catch
            {
                foreach (var transformer in loaded)
                    (transformer as IDisposable)?.Dispose();
                lease.Release();
                extractedRoot = null;
                throw;
            }
        }
        finally
        {
            DeleteOwnedRoot(extractedRoot);
        }
    }

    internal static void Save(OnnxTypedDecisionsTransformer transformer, string path)
        => Save((ITransformer)transformer, path);

    internal static void Save(DecisionInputPreparationTransformer transformer, string path)
        => Save((ITransformer)transformer, path);

    internal static void Save(OnnxDecisionModelScorerTransformer transformer, string path)
        => Save((ITransformer)transformer, path);

    internal static void Save(DecisionDecodingTransformer transformer, string path)
        => Save((ITransformer)transformer, path);

    internal static OnnxTypedDecisionsTransformer LoadFacade(MLContext mlContext, string path)
        => LoadTyped<OnnxTypedDecisionsTransformer>(
            mlContext,
            path,
            TypedDecisionPortableKinds.Facade);

    internal static DecisionInputPreparationTransformer LoadPreparation(
        MLContext mlContext,
        string path)
        => LoadTyped<DecisionInputPreparationTransformer>(
            mlContext,
            path,
            TypedDecisionPortableKinds.Preparation);

    internal static OnnxDecisionModelScorerTransformer LoadScoring(
        MLContext mlContext,
        string path)
        => LoadTyped<OnnxDecisionModelScorerTransformer>(
            mlContext,
            path,
            TypedDecisionPortableKinds.Scoring);

    internal static DecisionDecodingTransformer LoadDecoding(
        MLContext mlContext,
        string path)
        => LoadTyped<DecisionDecodingTransformer>(
            mlContext,
            path,
            TypedDecisionPortableKinds.Decoding);

    private static TTransformer LoadTyped<TTransformer>(
        MLContext mlContext,
        string path,
        string expectedKind)
        where TTransformer : class, ITransformer
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        var artifact = ReadArtifact(path, expectedKind, out var extractedRoot);
        try
        {
            var lease = new TypedDecisionRootLease(extractedRoot!);
            ITransformer transformer;
            try
            {
                transformer = CreateTransformer(mlContext, artifact, extractedRoot!, lease);
            }
            catch
            {
                lease.Release();
                throw;
            }
            lease.Release();
            extractedRoot = null;
            return (TTransformer)transformer;
        }
        finally
        {
            DeleteOwnedRoot(extractedRoot);
        }
    }

    private static ITransformer CreateTransformer(
        MLContext mlContext,
        PortableArtifact artifact,
        string extractedRoot,
        TypedDecisionRootLease lease)
    {
        ITransformer? transformer = null;
        try
        {
            transformer = artifact.Kind switch
            {
                TypedDecisionPortableKinds.Facade => CreateFacade(mlContext, artifact, extractedRoot, lease),
                TypedDecisionPortableKinds.Preparation => CreatePreparation(mlContext, artifact, extractedRoot, lease),
                TypedDecisionPortableKinds.Scoring => CreateScoring(mlContext, artifact, extractedRoot, lease),
                TypedDecisionPortableKinds.Decoding => CreateDecoding(mlContext, artifact, extractedRoot, lease),
                _ => throw new InvalidDataException(
                    $"Portable artifact kind '{artifact.Kind}' cannot be loaded as a transformer.")
            };
            return transformer;
        }
        catch
        {
            (transformer as IDisposable)?.Dispose();
            throw;
        }
    }

    private static OnnxTypedDecisionsTransformer CreateFacade(
        MLContext mlContext,
        PortableArtifact artifact,
        string extractedRoot,
        TypedDecisionRootLease lease)
    {
        var config = artifact.Facade
            ?? throw new InvalidDataException("The facade portable artifact has no configuration.");
        var options = new OnnxTypedDecisionsOptions
        {
            ModelAssetsPath = extractedRoot,
            Questions = FromQuestions(config.Questions),
            StateColumnName = config.StateColumnName,
            OutputPrefix = config.OutputPrefix,
            ResultsColumnName = config.ResultsColumnName,
            BatchSize = config.BatchSize
        };
        options.Validate();
        var transformer = new OnnxTypedDecisionsTransformer(mlContext, options);
        transformer.AttachOwnedAssetDirectory(extractedRoot, lease);
        return transformer;
    }

    private static DecisionInputPreparationTransformer CreatePreparation(
        MLContext mlContext,
        PortableArtifact artifact,
        string extractedRoot,
        TypedDecisionRootLease lease)
    {
        var config = artifact.Preparation
            ?? throw new InvalidDataException(
                "The preparation portable artifact has no configuration.");
        var options = new DecisionInputPreparationOptions
        {
            ModelAssetsPath = extractedRoot,
            Questions = FromQuestions(config.Questions),
            StateColumnName = config.StateColumnName,
            InputIdsColumnName = config.InputIdsColumnName,
            AttentionMaskColumnName = config.AttentionMaskColumnName,
            MarkerPositionsColumnName = config.MarkerPositionsColumnName,
            MarkerMaskColumnName = config.MarkerMaskColumnName,
            QuestionTypesColumnName = config.QuestionTypesColumnName,
            BatchSizeColumnName = config.BatchSizeColumnName,
            SequenceLengthColumnName = config.SequenceLengthColumnName,
            MarkerWidthColumnName = config.MarkerWidthColumnName
        };
        options.Validate();
        var transformer = new DecisionInputPreparationTransformer(mlContext, options);
        transformer.AttachOwnedAssetDirectory(extractedRoot, lease);
        return transformer;
    }

    private static OnnxDecisionModelScorerTransformer CreateScoring(
        MLContext mlContext,
        PortableArtifact artifact,
        string extractedRoot,
        TypedDecisionRootLease lease)
    {
        var config = artifact.Scoring
            ?? throw new InvalidDataException("The scoring portable artifact has no configuration.");
        var options = new OnnxDecisionModelScorerOptions
        {
            ModelAssetsPath = extractedRoot,
            InputIdsColumnName = config.InputIdsColumnName,
            AttentionMaskColumnName = config.AttentionMaskColumnName,
            MarkerPositionsColumnName = config.MarkerPositionsColumnName,
            MarkerMaskColumnName = config.MarkerMaskColumnName,
            QuestionTypesColumnName = config.QuestionTypesColumnName,
            BatchSizeColumnName = config.BatchSizeColumnName,
            SequenceLengthColumnName = config.SequenceLengthColumnName,
            MarkerWidthColumnName = config.MarkerWidthColumnName,
            LogitsColumnName = config.LogitsColumnName,
            ActionProbabilitiesColumnName = config.ActionProbabilitiesColumnName,
            BatchSize = config.BatchSize
        };
        options.Validate();
        var transformer = new OnnxDecisionModelScorerTransformer(mlContext, options);
        transformer.AttachOwnedAssetDirectory(extractedRoot, lease);
        return transformer;
    }

    private static DecisionDecodingTransformer CreateDecoding(
        MLContext mlContext,
        PortableArtifact artifact,
        string extractedRoot,
        TypedDecisionRootLease lease)
    {
        var config = artifact.Decoding
            ?? throw new InvalidDataException("The decoding portable artifact has no configuration.");
        var options = new DecisionDecodingOptions
        {
            ModelAssetsPath = extractedRoot,
            Questions = FromQuestions(config.Questions),
            InputIdsColumnName = config.InputIdsColumnName,
            AttentionMaskColumnName = config.AttentionMaskColumnName,
            MarkerPositionsColumnName = config.MarkerPositionsColumnName,
            MarkerMaskColumnName = config.MarkerMaskColumnName,
            QuestionTypesColumnName = config.QuestionTypesColumnName,
            BatchSizeColumnName = config.BatchSizeColumnName,
            SequenceLengthColumnName = config.SequenceLengthColumnName,
            MarkerWidthColumnName = config.MarkerWidthColumnName,
            LogitsColumnName = config.LogitsColumnName,
            ActionProbabilitiesColumnName = config.ActionProbabilitiesColumnName,
            OutputPrefix = config.OutputPrefix,
            ResultsColumnName = config.ResultsColumnName
        };
        options.Validate();
        var transformer = new DecisionDecodingTransformer(mlContext, options);
        transformer.AttachOwnedAssetDirectory(extractedRoot, lease);
        return transformer;
    }

    private static PortableTransformerDescriptor CreateDescriptor(ITransformer transformer)
    {
        return transformer switch
        {
            OnnxTypedDecisionsTransformer facade => new PortableTransformerDescriptor
            {
                Kind = TypedDecisionPortableKinds.Facade,
                Facade = ToFacade(facade.Options)
            },
            DecisionInputPreparationTransformer preparation => new PortableTransformerDescriptor
            {
                Kind = TypedDecisionPortableKinds.Preparation,
                Preparation = ToPreparation(preparation.Options)
            },
            OnnxDecisionModelScorerTransformer scoring => new PortableTransformerDescriptor
            {
                Kind = TypedDecisionPortableKinds.Scoring,
                Scoring = ToScoring(scoring.Options)
            },
            DecisionDecodingTransformer decoding => new PortableTransformerDescriptor
            {
                Kind = TypedDecisionPortableKinds.Decoding,
                Decoding = ToDecoding(decoding.Options)
            },
            _ => throw new NotSupportedException(
                $"Portable typed-decision persistence does not support transformer type " +
                $"'{transformer.GetType().FullName}'.")
        };
    }

    private static PortableArtifact CreateArtifact(
        PortableTransformerDescriptor descriptor)
        => new()
        {
            Kind = descriptor.Kind,
            Facade = descriptor.Facade,
            Preparation = descriptor.Preparation,
            Scoring = descriptor.Scoring,
            Decoding = descriptor.Decoding
        };

    private static TypedDecisionBundleLoadRequirements GetAssetRequirements(
        string kind)
        => kind switch
        {
            TypedDecisionPortableKinds.Facade =>
                TypedDecisionBundleLoadRequirements.All,
            TypedDecisionPortableKinds.Preparation =>
                TypedDecisionBundleLoadRequirements.Model |
                TypedDecisionBundleLoadRequirements.Profile |
                TypedDecisionBundleLoadRequirements.Tokenizer,
            TypedDecisionPortableKinds.Scoring =>
                TypedDecisionBundleLoadRequirements.Model |
                TypedDecisionBundleLoadRequirements.Tokenizer,
            TypedDecisionPortableKinds.Decoding =>
                TypedDecisionBundleLoadRequirements.Profile,
            _ => throw new NotSupportedException(
                $"Portable typed-decision persistence does not support asset requirements for '{kind}'.")
        };

    private static string GetAssetsPath(ITransformer transformer)
        => transformer switch
        {
            OnnxTypedDecisionsTransformer facade => facade.AssetsRootPath,
            DecisionInputPreparationTransformer preparation => preparation.AssetsRootPath,
            OnnxDecisionModelScorerTransformer scoring => scoring.AssetsRootPath,
            DecisionDecodingTransformer decoding => decoding.AssetsRootPath,
            _ => throw new NotSupportedException(
                $"Portable typed-decision persistence does not support transformer type " +
                $"'{transformer.GetType().FullName}'.")
        };

    private static List<ITransformer> ValidatePipelineScopes<TLastTransformer>(
        TransformerChain<TLastTransformer> pipeline)
        where TLastTransformer : class, ITransformer
    {
        ValidateUnsupportedScopeBits(pipeline);
        return ValidatePipelineScopes(
            pipeline,
            pipeline.GetModelFor(TransformerScope.Training),
            pipeline.GetModelFor(TransformerScope.Testing),
            pipeline.GetModelFor(TransformerScope.Scoring),
            pipeline.GetModelFor(
                TransformerScope.Training | TransformerScope.Scoring));
    }

    private static List<ITransformer> ValidatePipelineScopes<TLastTransformer>(
        TransformerChain<TLastTransformer> pipeline,
        TransformerChain<ITransformer> training,
        TransformerChain<ITransformer> testing,
        TransformerChain<ITransformer> scoring,
        TransformerChain<ITransformer> trainingAndScoring)
        where TLastTransformer : class, ITransformer
    {
        return ValidatePipelineScopes(
            pipeline.ToArray(),
            training.ToArray(),
            testing.ToArray(),
            scoring.ToArray(),
            trainingAndScoring.ToArray());
    }

    private static List<ITransformer> ValidatePipelineScopes(
        IReadOnlyList<ITransformer> all,
        IReadOnlyList<ITransformer> training,
        IReadOnlyList<ITransformer> testing,
        IReadOnlyList<ITransformer> scoring,
        IReadOnlyList<ITransformer> trainingAndScoring)
    {
        if (all.Any(static transformer =>
                transformer.GetType().IsGenericType &&
                transformer.GetType().GetGenericTypeDefinition() == typeof(TransformerChain<>)))
        {
            throw new NotSupportedException(
                "Nested TransformerChain values are not supported by typed-decision portable persistence.");
        }

        if (!SameSequence(all, training) ||
            !SameSequence(all, testing) ||
            !SameSequence(all, scoring) ||
            !SameSequence(all, trainingAndScoring))
        {
            throw new NotSupportedException(
                "Portable typed-decision persistence supports only flat TransformerChain values " +
                "whose Training, Testing, Scoring, and Training|Scoring scopes all contain " +
                "the complete sequence.");
        }

        return all.ToList();
    }

    private static void ValidateUnsupportedScopeBits<TLastTransformer>(
        TransformerChain<TLastTransformer> pipeline)
        where TLastTransformer : class, ITransformer
    {
        var unsupportedBits = ~TransformerScope.Everything;
        try
        {
            var unsupported = pipeline.GetModelFor(unsupportedBits);
            if (unsupported.ToArray().Length > 0)
            {
                throw new NotSupportedException(
                    "Portable typed-decision persistence rejects transformer scopes " +
                    "outside TransformerScope.Everything.");
            }
        }
        catch (NotSupportedException)
        {
            throw;
        }
        catch (ArgumentException ex)
        {
            throw new NotSupportedException(
                "Portable typed-decision persistence rejects transformer scopes " +
                "outside TransformerScope.Everything.",
                ex);
        }
    }

    private static bool SameSequence(
        IReadOnlyList<ITransformer> expected,
        IReadOnlyList<ITransformer> actual)
        => expected.Count == actual.Count &&
           expected.Zip(actual).All(static pair => ReferenceEquals(pair.First, pair.Second));

    private static PortableFacade ToFacade(OnnxTypedDecisionsOptions options)
        => new()
        {
            Questions = ToQuestions(options.Questions),
            StateColumnName = options.StateColumnName,
            OutputPrefix = options.OutputPrefix,
            ResultsColumnName = options.ResultsColumnName,
            BatchSize = options.BatchSize
        };

    private static PortablePreparation ToPreparation(DecisionInputPreparationOptions options)
        => new()
        {
            Questions = ToQuestions(options.Questions),
            StateColumnName = options.StateColumnName,
            InputIdsColumnName = options.InputIdsColumnName,
            AttentionMaskColumnName = options.AttentionMaskColumnName,
            MarkerPositionsColumnName = options.MarkerPositionsColumnName,
            MarkerMaskColumnName = options.MarkerMaskColumnName,
            QuestionTypesColumnName = options.QuestionTypesColumnName,
            BatchSizeColumnName = options.BatchSizeColumnName,
            SequenceLengthColumnName = options.SequenceLengthColumnName,
            MarkerWidthColumnName = options.MarkerWidthColumnName
        };

    private static PortableScoring ToScoring(OnnxDecisionModelScorerOptions options)
        => new()
        {
            InputIdsColumnName = options.InputIdsColumnName,
            AttentionMaskColumnName = options.AttentionMaskColumnName,
            MarkerPositionsColumnName = options.MarkerPositionsColumnName,
            MarkerMaskColumnName = options.MarkerMaskColumnName,
            QuestionTypesColumnName = options.QuestionTypesColumnName,
            BatchSizeColumnName = options.BatchSizeColumnName,
            SequenceLengthColumnName = options.SequenceLengthColumnName,
            MarkerWidthColumnName = options.MarkerWidthColumnName,
            LogitsColumnName = options.LogitsColumnName,
            ActionProbabilitiesColumnName = options.ActionProbabilitiesColumnName,
            BatchSize = options.BatchSize
        };

    private static PortableDecoding ToDecoding(DecisionDecodingOptions options)
        => new()
        {
            Questions = ToQuestions(options.Questions),
            InputIdsColumnName = options.InputIdsColumnName,
            AttentionMaskColumnName = options.AttentionMaskColumnName,
            MarkerPositionsColumnName = options.MarkerPositionsColumnName,
            MarkerMaskColumnName = options.MarkerMaskColumnName,
            QuestionTypesColumnName = options.QuestionTypesColumnName,
            BatchSizeColumnName = options.BatchSizeColumnName,
            SequenceLengthColumnName = options.SequenceLengthColumnName,
            MarkerWidthColumnName = options.MarkerWidthColumnName,
            LogitsColumnName = options.LogitsColumnName,
            ActionProbabilitiesColumnName = options.ActionProbabilitiesColumnName,
            OutputPrefix = options.OutputPrefix,
            ResultsColumnName = options.ResultsColumnName
        };

    private static PortableQuestion[] ToQuestions(IReadOnlyList<DecisionQuestion> questions)
        => questions.Select(static question => new PortableQuestion
        {
            Id = question.Id,
            Type = question.Type,
            Instructions = question.Instructions,
            Choices = question.Choices?.Select(static choice => new PortableChoice
            {
                Label = choice.Key,
                Description = choice.Value
            }).ToArray(),
            ScoreLevels = question.ScoreLevels?.ToArray(),
            NoulCriteria = question.NoulCriteria is null
                ? null
                : new PortableNoulCriteria
                {
                    True = question.NoulCriteria.True,
                    False = question.NoulCriteria.False
                }
        }).ToArray();

    private static IReadOnlyList<DecisionQuestion> FromQuestions(
        IReadOnlyList<PortableQuestion>? questions)
    {
        ArgumentNullException.ThrowIfNull(questions);
        return questions.Select((question, index) =>
        {
            if (question is null)
                throw new InvalidDataException(
                    $"Portable question entry {index} is null.");
            return question.Type switch
            {
                DecisionQuestionType.Choice => DecisionQuestion.Choice(
                    question.Id,
                    question.Instructions,
                    question.Choices is null
                        ? throw new InvalidDataException(
                            $"Choice question '{question.Id}' is missing Choices.")
                        : question.Choices.ToDictionary(
                            static choice => choice.Label,
                            static choice => choice.Description,
                            StringComparer.Ordinal)),
                DecisionQuestionType.Score => DecisionQuestion.Score(
                    question.Id,
                    question.Instructions,
                    question.ScoreLevels
                        ?? throw new InvalidDataException(
                            $"Score question '{question.Id}' is missing ScoreLevels.")),
                DecisionQuestionType.Noul => DecisionQuestion.Noul(
                    question.Id,
                    question.Instructions,
                    question.NoulCriteria is null
                        ? null
                        : new NoulCriteria(
                            question.NoulCriteria.True,
                            question.NoulCriteria.False)),
                _ => throw new InvalidDataException(
                    $"Question '{question.Id}' has no choices or supported type.")
            };
        }).ToArray();
    }

    private static void SaveArtifact(
        PortableArtifact artifact,
        IReadOnlyList<string> assetPaths,
        TypedDecisionBundleLoadRequirements requirements,
        string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (assetPaths.Count == 0)
            throw new InvalidDataException("A portable typed-decision artifact requires model assets.");

        var destination = Path.GetFullPath(path);
        var destinationDirectory = Path.GetDirectoryName(destination)!;
        Directory.CreateDirectory(destinationDirectory);
        var temporary = Path.Combine(
            destinationDirectory,
            $".{Path.GetFileName(destination)}.{Guid.NewGuid():N}.tmp");

        try
        {
            var asset = ReadAssetSource(assetPaths[0], requirements);
            try
            {
                for (var index = 1; index < assetPaths.Count; index++)
                {
                    var other = ReadAssetSource(assetPaths[index], requirements);
                    try
                    {
                        if (!HaveSameHashes(asset.Hashes, other.Hashes))
                            throw new InvalidDataException(
                                "All transformers in a supported portable pipeline must reference " +
                                "the same model/tokenizer asset payload.");
                    }
                    finally
                    {
                        other.Bundle.Dispose();
                    }
                }

                artifact = artifact with
                {
                    ExecutionPolicy = "UseLoadContext",
                    FormatVersion = CurrentFormatVersion,
                    AssetHashes = new Dictionary<string, string>(
                        asset.Hashes,
                        StringComparer.Ordinal)
                };
                var portableJson = JsonSerializer.SerializeToUtf8Bytes(artifact, JsonOptions);
                using (var stream = new FileStream(
                    temporary,
                    FileMode.CreateNew,
                    FileAccess.ReadWrite,
                    FileShare.None,
                    1024 * 1024,
                    FileOptions.SequentialScan))
                using (var archive = new ZipArchive(stream, ZipArchiveMode.Create))
                {
                    WriteBytes(archive, PortableManifestFileName, portableJson);
                    foreach (var pair in asset.Files.OrderBy(static pair => pair.Key, StringComparer.Ordinal))
                        WriteFile(archive, pair.Key, pair.Value);
                    WriteBytes(
                        archive,
                        TypedDecisionBundle.ManifestFileName,
                        asset.BundleManifestJson);
                    stream.Flush(flushToDisk: true);
                }

                ReplaceAtomically(temporary, destination);
            }
            finally
            {
                asset.Bundle.Dispose();
            }
        }
        catch
        {
            DeleteFile(temporary);
            throw;
        }
    }

    private static AssetSource ReadAssetSource(
        string assetsPath,
        TypedDecisionBundleLoadRequirements requirements)
    {
        var bundle = TypedDecisionBundle.Open(assetsPath, requirements);
        try
        {
            var files = new Dictionary<string, string>(StringComparer.Ordinal);
            var hashes = new Dictionary<string, string>(StringComparer.Ordinal);
            var root = bundle.RootPath;
            var manifest = CreatePortableBundleManifest(bundle, requirements);

            if (requirements.HasFlag(TypedDecisionBundleLoadRequirements.Model))
            {
                AddAssetFile(root, manifest.ModelFile, files, hashes);
                foreach (var external in manifest.ExternalDataFiles)
                    AddAssetFile(root, external, files, hashes);
            }
            if (requirements.HasFlag(TypedDecisionBundleLoadRequirements.Profile))
                AddAssetFile(root, "laya_config.json", files, hashes);

            if (requirements.HasFlag(TypedDecisionBundleLoadRequirements.Tokenizer))
            {
                var tokenizerRoot = AssetArchive.ResolveWithinRoot(
                    root,
                    manifest.TokenizerDirectory,
                    "tokenizer directory");
                foreach (var fileName in new[] { "tokenizer.json", "tokenizer_config.json" })
                {
                    var file = Path.Combine(tokenizerRoot, fileName);
                    if (File.Exists(file))
                    {
                        var relative = Path.GetRelativePath(root, file).Replace('\\', '/');
                        AddAssetFile(root, relative, files, hashes);
                    }
                }
            }

            var bundleManifestJson = JsonSerializer.SerializeToUtf8Bytes(
                manifest,
                BundleJsonOptions);
            hashes[TypedDecisionBundle.ManifestFileName] =
                ComputeSha256(bundleManifestJson);
            return new AssetSource(
                bundle,
                files,
                hashes,
                bundleManifestJson);
        }
        catch
        {
            bundle.Dispose();
            throw;
        }
    }

    private static TypedDecisionBundleManifest CreatePortableBundleManifest(
        TypedDecisionBundle bundle,
        TypedDecisionBundleLoadRequirements requirements)
    {
        var source = bundle.Manifest;
        var external = requirements.HasFlag(TypedDecisionBundleLoadRequirements.Model)
            ? DiscoverExternalDataFiles(bundle)
                .Concat(source.ExternalDataFiles.Select(
                    relative => ResolveExternalRelativePath(bundle, relative, preferModelDirectory: false)))
                .Distinct(StringComparer.Ordinal)
                .Order(StringComparer.Ordinal)
                .ToArray()
            : Array.Empty<string>();
        var manifest = new TypedDecisionBundleManifest
        {
            FormatVersion = source.FormatVersion,
            ModelFile = AssetArchive.NormalizeRelativePath(source.ModelFile, "model file"),
            ExternalDataFiles = external,
            TokenizerDirectory = AssetArchive.NormalizeRelativePath(
                source.TokenizerDirectory,
                "tokenizer directory"),
            Profile = source.Profile,
            Decoder = source.Decoder
        };
        return manifest;
    }

    private static string ResolveExternalRelativePath(
        TypedDecisionBundle bundle,
        string relative,
        bool preferModelDirectory)
    {
        var modelDirectory = Path.GetDirectoryName(bundle.ModelPath)!;
        if (preferModelDirectory)
        {
            var modelRelative = AssetArchive.ResolveWithinRoot(
                modelDirectory,
                relative,
                "ONNX external-data location");
            if (File.Exists(modelRelative))
            {
                return Path.GetRelativePath(bundle.RootPath, modelRelative)
                    .Replace('\\', '/');
            }
        }

        var rootRelative = AssetArchive.ResolveWithinRoot(
            bundle.RootPath,
            relative,
            "bundle external-data location");
        if (!File.Exists(rootRelative))
            throw new FileNotFoundException(
                $"Typed-decision bundle is missing external-data file '{relative}'.",
                rootRelative);
        return Path.GetRelativePath(bundle.RootPath, rootRelative)
            .Replace('\\', '/');
    }

    private static IReadOnlyList<string> DiscoverExternalDataFiles(
        TypedDecisionBundle bundle)
        => AssetArchive.DiscoverOnnxExternalDataFiles(bundle.ModelPath)
            .Select(relative =>
            {
                var modelDirectory = Path.GetDirectoryName(bundle.ModelPath)!;
                var modelRelative = AssetArchive.ResolveWithinRoot(
                    modelDirectory,
                    relative,
                    "ONNX external-data location");
                return Path.GetRelativePath(bundle.RootPath, modelRelative)
                    .Replace('\\', '/');
            })
            .ToArray();

    private static void AddAssetFile(
        string root,
        string relative,
        IDictionary<string, string> files,
        IDictionary<string, string> hashes)
    {
        var normalized = AssetArchive.NormalizeRelativePath(relative, "asset path");
        var path = AssetArchive.ResolveWithinRoot(root, normalized, "asset path");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Typed-decision portable save is missing asset '{normalized}'.",
                path);
        files[normalized] = path;
        hashes[normalized] = ComputeSha256(path);
    }

    private static bool HaveSameHashes(
        IReadOnlyDictionary<string, string> left,
        IReadOnlyDictionary<string, string> right)
        => left.Count == right.Count &&
           left.All(pair =>
               right.TryGetValue(pair.Key, out var value) &&
               string.Equals(pair.Value, value, StringComparison.OrdinalIgnoreCase));

    private static PortableArtifact ReadArtifact(
        string path,
        string? expectedKind,
        out string? extractedRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
            throw new FileNotFoundException("Portable typed-decision artifact was not found.", fullPath);

        extractedRoot = Path.Combine(
            Path.GetTempPath(),
            "mlnet-typed-decision-portable",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(extractedRoot);
        try
        {
            AssetArchive.ExtractZipSafely(fullPath, extractedRoot);
            var manifestPath = AssetArchive.ResolveWithinRoot(
                extractedRoot,
                PortableManifestFileName,
                "portable manifest");
            var artifact = JsonSerializer.Deserialize<PortableArtifact>(
                File.ReadAllText(manifestPath),
                JsonOptions)
                ?? throw new InvalidDataException(
                    "Portable typed-decision manifest is empty.");
            artifact.Validate(expectedKind);
            var requirements = artifact.Kind == TypedDecisionPortableKinds.Pipeline
                ? GetPipelineAssetRequirements(artifact.Pipeline!)
                : GetAssetRequirements(artifact.Kind);
            using var bundle = TypedDecisionBundle.OpenDirectory(
                extractedRoot,
                requirements: requirements);
            ValidateAssetHashes(extractedRoot, artifact, bundle, requirements);
            return artifact;
        }
        catch (JsonException ex)
        {
            DeleteOwnedRoot(extractedRoot);
            extractedRoot = null;
            throw new InvalidDataException(
                "Portable typed-decision artifact contains malformed JSON.",
                ex);
        }
        catch
        {
            DeleteOwnedRoot(extractedRoot);
            extractedRoot = null;
            throw;
        }
    }

    private static TypedDecisionBundleLoadRequirements GetPipelineAssetRequirements(
        PortablePipeline pipeline)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        if (pipeline.Transformers is null)
            throw new InvalidDataException("Portable pipeline has no transformer descriptors.");
        var requirements = TypedDecisionBundleLoadRequirements.None;
        foreach (var descriptor in pipeline.Transformers)
        {
            if (descriptor is null)
                throw new InvalidDataException("Portable pipeline contains a null transformer descriptor.");
            requirements |= GetAssetRequirements(descriptor.Kind);
        }
        return requirements;
    }

    private static void ValidateAssetHashes(
        string root,
        PortableArtifact artifact,
        TypedDecisionBundle bundle,
        TypedDecisionBundleLoadRequirements requirements)
    {
        var bundleManifest = bundle.Manifest;
        if (artifact.AssetHashes is null || artifact.AssetHashes.Count == 0)
            throw new InvalidDataException("Portable typed-decision manifest has no asset hashes.");

        var required = new HashSet<string>(StringComparer.Ordinal)
        {
            TypedDecisionBundle.ManifestFileName
        };
        if (requirements.HasFlag(TypedDecisionBundleLoadRequirements.Model))
        {
            required.Add(AssetArchive.NormalizeRelativePath(
                bundleManifest.ModelFile,
                "model file"));
            var externalDataFiles = (bundleManifest.ExternalDataFiles ?? [])
                .Concat(DiscoverExternalDataFiles(bundle))
                .Distinct(StringComparer.Ordinal);
            foreach (var external in externalDataFiles)
            {
                required.Add(AssetArchive.NormalizeRelativePath(
                    external,
                    "external-data file"));
            }
        }
        if (requirements.HasFlag(TypedDecisionBundleLoadRequirements.Profile))
            required.Add("laya_config.json");
        if (requirements.HasFlag(TypedDecisionBundleLoadRequirements.Tokenizer))
        {
            var tokenizerDirectory = AssetArchive.NormalizeRelativePath(
                bundleManifest.TokenizerDirectory,
                "tokenizer directory");
            required.Add($"{tokenizerDirectory}/tokenizer.json");
            var optionalConfig = $"{tokenizerDirectory}/tokenizer_config.json";
            if (File.Exists(AssetArchive.ResolveWithinRoot(root, optionalConfig, "tokenizer config")))
                required.Add(optionalConfig);
        }

        foreach (var requiredAsset in required)
        {
            if (!artifact.AssetHashes.ContainsKey(requiredAsset))
                throw new InvalidDataException(
                    $"Portable typed-decision manifest is missing the SHA-256 hash for required asset '{requiredAsset}'.");
        }

        foreach (var pair in artifact.AssetHashes)
        {
            var relative = AssetArchive.NormalizeRelativePath(pair.Key, "portable asset");
            var path = AssetArchive.ResolveWithinRoot(root, relative, "portable asset");
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"Portable typed-decision artifact is missing '{relative}'.",
                    path);
            var actual = ComputeSha256(path);
            if (!string.Equals(actual, pair.Value, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    $"SHA-256 mismatch for portable asset '{relative}'.");
        }
    }

    private static void WriteFile(ZipArchive archive, string relative, string source)
    {
        var compression = Path.GetExtension(relative).ToLowerInvariant() is ".data" or ".bin"
            ? CompressionLevel.NoCompression
            : CompressionLevel.Optimal;
        var entry = archive.CreateEntry(relative, compression);
        using var input = new FileStream(
            source,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            1024 * 1024,
            FileOptions.SequentialScan);
        using var output = entry.Open();
        input.CopyTo(output);
    }

    private static void WriteBytes(ZipArchive archive, string relative, byte[] bytes)
    {
        var entry = archive.CreateEntry(relative, CompressionLevel.Optimal);
        using var output = entry.Open();
        output.Write(bytes);
    }

    private static void ReplaceAtomically(string temporary, string destination)
    {
        if (File.Exists(destination))
        {
            try
            {
                File.Replace(temporary, destination, null);
                return;
            }
            catch (PlatformNotSupportedException)
            {
                // File.Move with overwrite is the portable fallback.
            }
        }

        File.Move(temporary, destination, overwrite: true);
    }

    private static void DeleteOwnedRoot(string? root)
    {
        if (string.IsNullOrWhiteSpace(root))
            return;
        if (Directory.Exists(root))
            Directory.Delete(root, recursive: true);
    }

    private static void DeleteFile(string path)
    {
        if (File.Exists(path))
            File.Delete(path);
    }

    private static string ComputeSha256(byte[] bytes)
        => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

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

    private static readonly JsonSerializerOptions BundleJsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() }
    };

    private sealed record AssetSource(
        TypedDecisionBundle Bundle,
        IReadOnlyDictionary<string, string> Files,
        IReadOnlyDictionary<string, string> Hashes,
        byte[] BundleManifestJson);
}

internal static class TypedDecisionPortableKinds
{
    internal const string Facade = "facade";
    internal const string Preparation = "preparation";
    internal const string Scoring = "scoring";
    internal const string Decoding = "decoding";
    internal const string Pipeline = "pipeline";
}

internal sealed record PortableArtifact
{
    public int FormatVersion { get; init; }
    public string Kind { get; init; } = "";
    public string ExecutionPolicy { get; init; } = "";
    public Dictionary<string, string>? AssetHashes { get; init; } = new(StringComparer.Ordinal);
    public PortableFacade? Facade { get; init; }
    public PortablePreparation? Preparation { get; init; }
    public PortableScoring? Scoring { get; init; }
    public PortableDecoding? Decoding { get; init; }
    public PortablePipeline? Pipeline { get; init; }

    internal void Validate(string? expectedKind)
    {
        if (FormatVersion != TypedDecisionPortableModel.CurrentFormatVersion)
            throw new InvalidDataException(
                $"Unsupported typed-decision portable format version {FormatVersion}.");
        if (ExecutionPolicy != "UseLoadContext")
            throw new InvalidDataException(
                $"Unsupported typed-decision execution policy '{ExecutionPolicy}'.");
        if (Kind is not TypedDecisionPortableKinds.Facade and
            not TypedDecisionPortableKinds.Preparation and
            not TypedDecisionPortableKinds.Scoring and
            not TypedDecisionPortableKinds.Decoding and
            not TypedDecisionPortableKinds.Pipeline)
        {
            throw new InvalidDataException($"Unsupported typed-decision portable kind '{Kind}'.");
        }
        if (expectedKind is not null &&
            !string.Equals(expectedKind, Kind, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Expected a '{expectedKind}' portable artifact but found '{Kind}'.");
        }
        if (Kind == TypedDecisionPortableKinds.Pipeline && Pipeline is null)
            throw new InvalidDataException("Portable pipeline artifact is missing its descriptor.");
        if (Kind == TypedDecisionPortableKinds.Pipeline &&
            (Pipeline!.Transformers is null || Pipeline.Transformers.Length == 0))
            throw new InvalidDataException(
                "Portable pipeline artifact must contain at least one transformer descriptor.");
        if (Kind == TypedDecisionPortableKinds.Facade && Facade is null)
            throw new InvalidDataException("Portable facade artifact is missing its configuration.");
        if (Kind == TypedDecisionPortableKinds.Preparation && Preparation is null)
            throw new InvalidDataException(
                "Portable preparation artifact is missing its configuration.");
        if (Kind == TypedDecisionPortableKinds.Scoring && Scoring is null)
            throw new InvalidDataException("Portable scoring artifact is missing its configuration.");
        if (Kind == TypedDecisionPortableKinds.Decoding && Decoding is null)
            throw new InvalidDataException(
                "Portable decoding artifact is missing its configuration.");
        if (Kind == TypedDecisionPortableKinds.Pipeline)
        {
            for (var index = 0; index < Pipeline!.Transformers!.Length; index++)
            {
                var descriptor = Pipeline.Transformers[index]
                    ?? throw new InvalidDataException(
                        $"Portable pipeline transformer descriptor {index} is null.");
                descriptor.Validate();
            }
        }
        Facade?.Validate();
        Preparation?.Validate();
        Scoring?.Validate();
        Decoding?.Validate();
    }
}

internal sealed record PortableTransformerDescriptor
{
    public string Kind { get; init; } = "";
    public PortableFacade? Facade { get; init; }
    public PortablePreparation? Preparation { get; init; }
    public PortableScoring? Scoring { get; init; }
    public PortableDecoding? Decoding { get; init; }

    internal void Validate()
    {
        if (Kind is not TypedDecisionPortableKinds.Facade and
            not TypedDecisionPortableKinds.Preparation and
            not TypedDecisionPortableKinds.Scoring and
            not TypedDecisionPortableKinds.Decoding)
        {
            throw new InvalidDataException(
                $"Unsupported typed-decision transformer kind '{Kind}'.");
        }

        switch (Kind)
        {
            case TypedDecisionPortableKinds.Facade when Facade is null:
                throw new InvalidDataException(
                    "Portable facade descriptor is missing its configuration.");
            case TypedDecisionPortableKinds.Preparation when Preparation is null:
                throw new InvalidDataException(
                    "Portable preparation descriptor is missing its configuration.");
            case TypedDecisionPortableKinds.Scoring when Scoring is null:
                throw new InvalidDataException(
                    "Portable scoring descriptor is missing its configuration.");
            case TypedDecisionPortableKinds.Decoding when Decoding is null:
                throw new InvalidDataException(
                    "Portable decoding descriptor is missing its configuration.");
        }

        Facade?.Validate();
        Preparation?.Validate();
        Scoring?.Validate();
        Decoding?.Validate();
    }
}

internal sealed record PortablePipeline
{
    public PortableTransformerDescriptor[]? Transformers { get; init; } = [];
}

internal sealed record PortableFacade
{
    public PortableQuestion[]? Questions { get; init; } = [];
    public string StateColumnName { get; init; } = "State";
    public string OutputPrefix { get; init; } = "Decision_";
    public string ResultsColumnName { get; init; } = "DecisionResults";
    public int BatchSize { get; init; } = 32;

    internal void Validate()
        => PortableValidation.ValidateQuestions(Questions, "facade");
}

internal sealed record PortablePreparation
{
    public PortableQuestion[]? Questions { get; init; } = [];
    public string StateColumnName { get; init; } = "State";
    public string InputIdsColumnName { get; init; } = "DecisionInputIds";
    public string AttentionMaskColumnName { get; init; } = "DecisionAttentionMask";
    public string MarkerPositionsColumnName { get; init; } = "DecisionMarkerPositions";
    public string MarkerMaskColumnName { get; init; } = "DecisionMarkerMask";
    public string QuestionTypesColumnName { get; init; } = "DecisionQuestionTypes";
    public string BatchSizeColumnName { get; init; } = "DecisionBatchSize";
    public string SequenceLengthColumnName { get; init; } = "DecisionSequenceLength";
    public string MarkerWidthColumnName { get; init; } = "DecisionMarkerWidth";

    internal void Validate()
        => PortableValidation.ValidateQuestions(Questions, "preparation");
}

internal sealed record PortableScoring
{
    public string InputIdsColumnName { get; init; } = "DecisionInputIds";
    public string AttentionMaskColumnName { get; init; } = "DecisionAttentionMask";
    public string MarkerPositionsColumnName { get; init; } = "DecisionMarkerPositions";
    public string MarkerMaskColumnName { get; init; } = "DecisionMarkerMask";
    public string QuestionTypesColumnName { get; init; } = "DecisionQuestionTypes";
    public string BatchSizeColumnName { get; init; } = "DecisionBatchSize";
    public string SequenceLengthColumnName { get; init; } = "DecisionSequenceLength";
    public string MarkerWidthColumnName { get; init; } = "DecisionMarkerWidth";
    public string LogitsColumnName { get; init; } = "DecisionLogits";
    public string ActionProbabilitiesColumnName { get; init; } = "DecisionActionProbabilities";
    public int BatchSize { get; init; } = 32;

    internal void Validate()
    {
        if (BatchSize <= 0)
            throw new InvalidDataException("Portable scoring BatchSize must be positive.");
    }
}

internal sealed record PortableDecoding
{
    public PortableQuestion[]? Questions { get; init; } = [];
    public string InputIdsColumnName { get; init; } = "DecisionInputIds";
    public string AttentionMaskColumnName { get; init; } = "DecisionAttentionMask";
    public string MarkerPositionsColumnName { get; init; } = "DecisionMarkerPositions";
    public string MarkerMaskColumnName { get; init; } = "DecisionMarkerMask";
    public string QuestionTypesColumnName { get; init; } = "DecisionQuestionTypes";
    public string BatchSizeColumnName { get; init; } = "DecisionBatchSize";
    public string SequenceLengthColumnName { get; init; } = "DecisionSequenceLength";
    public string MarkerWidthColumnName { get; init; } = "DecisionMarkerWidth";
    public string LogitsColumnName { get; init; } = "DecisionLogits";
    public string ActionProbabilitiesColumnName { get; init; } = "DecisionActionProbabilities";
    public string OutputPrefix { get; init; } = "Decision_";
    public string ResultsColumnName { get; init; } = "DecisionResults";

    internal void Validate()
        => PortableValidation.ValidateQuestions(Questions, "decoding");
}

internal static class PortableValidation
{
    internal static void ValidateQuestions(
        IReadOnlyList<PortableQuestion>? questions,
        string kind)
    {
        if (questions is null)
            throw new InvalidDataException(
                $"Portable {kind} configuration is missing Questions.");
        for (var index = 0; index < questions.Count; index++)
        {
            if (questions[index] is null)
                throw new InvalidDataException(
                    $"Portable {kind} question entry {index} is null.");
        }
    }
}

internal sealed record PortableQuestion
{
    public string Id { get; init; } = "";
    public DecisionQuestionType Type { get; init; }
    public string Instructions { get; init; } = "";
    public PortableChoice[]? Choices { get; init; }
    public string[]? ScoreLevels { get; init; }
    public PortableNoulCriteria? NoulCriteria { get; init; }
}

internal sealed record PortableChoice
{
    public string Label { get; init; } = "";
    public string? Description { get; init; }
}

internal sealed record PortableNoulCriteria
{
    public string? True { get; init; }
    public string? False { get; init; }
}
