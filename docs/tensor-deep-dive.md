# Tensor Deep Dive: System.Numerics.Tensors for AI Workloads

This document is a standalone tutorial on using `System.Numerics.Tensors` for AI workloads in .NET, grounded in the real implementation of this embedding transform. It covers the three tensor worlds, when to use each, and walks through the SIMD-accelerated math line by line.

## The Three Tensor Worlds

Three distinct numeric/tensor systems meet in this transform. Each owns a specific stage of the pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    System.Numerics.Tensors                         │
│                                                                     │
│  ┌─────────────────────────┐  ┌──────────────────────────────────┐ │
│  │   TensorPrimitives      │  │   Tensor<T>                      │ │
│  │   (Flat Span<T>)        │  │   (Shape-aware, multi-dim)       │ │
│  │                         │  │                                  │ │
│  │   • SIMD-accelerated    │  │   • Create, Reshape, Slice       │ │
│  │   • Add, Multiply, Div  │  │   • Broadcast                    │ │
│  │   • Norm, Sum, Dot      │  │   • Indexing: t[b, s, d]         │ │
│  │   • SoftMax, CosineSim  │  │   • Wraps array (managed view)   │ │
│  │   • No shape awareness  │  │   • Delegates math to Primitives │ │
│  └─────────────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────┐
│   Microsoft.ML.OnnxRuntime         │
│                                    │
│   • OrtValue (native tensor)       │
│   • CreateTensorValueFromMemory    │
│   • GetTensorDataAsSpan<T>()       │
│   • InferenceSession.Run()         │
└────────────────────────────────────┘
```

### `Tensor<T>` — Shape and Construction

`Tensor<T>` is a shape-aware multi-dimensional array type in `System.Numerics.Tensors`. It supports creation from existing arrays, reshaping, slicing, broadcasting, and multi-dimensional indexing.

**Key property:** `Tensor.Create<T>(array, shape)` **wraps the existing array without copying**. This means:
- Writes through `Tensor<T>` indexing are reflected in the original array
- The backing array can be offered to other APIs (like OrtValue) without an intermediate tensor copy. This is a managed-memory view, not a promise that the complete ONNX call is zero-copy.

```csharp
// Create a [2, 3] tensor backed by an existing array
var data = new long[6];
var tensor = Tensor.Create<long>(data, [2, 3]);

tensor[0, 2] = 42;     // sets data[2] = 42
tensor[1, 0] = 99;     // sets data[3] = 99
// data is now: [0, 0, 42, 99, 0, 0]
```

### `TensorPrimitives` — SIMD Math

`TensorPrimitives` provides static methods that operate on flat `Span<T>` / `ReadOnlySpan<T>`. They are SIMD-accelerated by the .NET runtime — on x64 with AVX2, operations process 8 floats simultaneously.

**No shape awareness.** These methods don't know about dimensions or axes. They operate on contiguous memory. This makes them perfect for inner-loop math where you've already computed the correct offsets.

Key operations we use:

| Method | What it does | Our use case |
|--------|-------------|--------------|
| `TensorPrimitives.Add(x, y, dest)` | `dest[i] = x[i] + y[i]` | Accumulate token embeddings |
| `TensorPrimitives.Divide(x, scalar, dest)` | `dest[i] = x[i] / scalar` | Mean (divide by count) and L2 norm |
| `TensorPrimitives.Norm(x)` | `√(Σ x[i]²)` | L2 norm for normalization |
| `TensorPrimitives.Max(x, y, dest)` | `dest[i] = max(x[i], y[i])` | Max pooling |
| `TensorPrimitives.CosineSimilarity(x, y)` | `dot(x,y) / (‖x‖ × ‖y‖)` | Similarity testing |

### `OrtValue` — The ONNX Bridge

`OrtValue` is OnnxRuntime's native tensor type. For CPU inputs it can bind managed memory for the duration of inference:

```csharp
// CPU input: binds/pins the managed array for the native call
var ortValue = OrtValue.CreateTensorValueFromMemory(managedArray, shape);

// Borrowed read: returns a span pointing into native output memory
ReadOnlySpan<float> output = results[0].GetTensorDataAsSpan<float>();
```

The managed input may be pinned rather than copied on the CPU path, but an execution provider can transfer it to device memory. Output spans borrow native storage and are valid only while the result values remain alive; production code snapshots them into managed arrays before disposing results or returning them from a batch. Tensor construction, padding, batch repacking, VBuffer snapshots, and final embedding objects can therefore allocate even when an individual bridge is a view.

## Where Each Tool Is Used

| Pipeline Stage | Tensor\<T\> | TensorPrimitives | OrtValue | Why |
|---------------|:-----------:|:----------------:|:--------:|-----|
| Input construction | ✅ | | | Shape-safe `[b, s]` indexing |
| ONNX I/O bridge | | | ✅ | CPU input view/pinning when supported; providers may copy to device memory |
| Mean pooling | | ✅ | | SIMD-accelerated inner loop |
| L2 normalization | | ✅ | | `Norm` + `Divide` |
| Cosine similarity | | ✅ | | Built-in `CosineSimilarity` |
| Shape documentation | ✅ | | | Explicit `[batch, seqLen, hiddenDim]` |

## Input Construction Walkthrough

The tokenizer produces `List<int>` token IDs. We need to build two tensors for ONNX: `input_ids[batch, seqLen]` and `attention_mask[batch, seqLen]`.

```csharp
// 1. Allocate flat backing arrays
var idsArray = new long[batchSize * seqLen];    // initialized to 0 (padding)
var maskArray = new long[batchSize * seqLen];   // initialized to 0 (no attention)

// 2. Wrap with Tensor<T> for shape-safe indexing
var idsTensor = Tensor.Create<long>(idsArray, [batchSize, seqLen]);
var maskTensor = Tensor.Create<long>(maskArray, [batchSize, seqLen]);

// 3. Fill using clean multi-dim indexing
for (int b = 0; b < batchSize; b++)
{
    var tokens = tokenizer.EncodeToIds(texts[b], seqLen, out _, out _);
    for (int s = 0; s < tokens.Count && s < seqLen; s++)
    {
        idsTensor[b, s] = tokens[s];   // writes to idsArray[b * seqLen + s]
        maskTensor[b, s] = 1;          // writes to maskArray[b * seqLen + s]
    }
    // positions beyond tokens.Count remain 0 → padding with no attention
}

// 4. Pass the SAME flat arrays to OnnxRuntime; CPU binding may avoid an
//    intermediate managed copy, while providers may transfer to a device.
var ortIds = OrtValue.CreateTensorValueFromMemory(idsArray, [batchSize, seqLen]);
var ortMask = OrtValue.CreateTensorValueFromMemory(maskArray, [batchSize, seqLen]);
```

**Why Tensor\<T\> here and not manual indexing?**
`idsTensor[b, s]` is clearer and less error-prone than `idsArray[b * seqLen + s]`. Since `Tensor.Create` wraps without copying, we get the ergonomics of multi-dimensional indexing and can offer the same backing array to the CPU ORT bridge; this is not a guarantee that execution avoids native or GPU transfers.

**Why not Tensor\<T\> for the math stages?**
`Tensor<T>` in .NET 9/10 does NOT have axis-wise reduction — there's no `Tensor.Sum(tensor, dim: 1)`. Mean pooling requires summing across the sequence dimension while preserving the hidden dimension. We'd need to implement the axis reduction manually regardless, so `TensorPrimitives` on flat spans with explicit offset calculation is both simpler and guaranteed SIMD.

## Mean Pooling: SIMD Walkthrough

Mean pooling averages all non-padding token embeddings into a single vector. Given:
- `hiddenStates`: shape `[batch, seqLen, hiddenDim]` — output from ONNX
- `attentionMask`: shape `[batch, seqLen]` — 1 for real tokens, 0 for padding

The formula for batch item `b`, dimension `d`:

```
                Σ  hiddenStates[b, s, d] × attentionMask[b, s]
                s
embedding[d] = ─────────────────────────────────────────────────
                         Σ  attentionMask[b, s]
                         s
```

Implementation from `EmbeddingPooling.cs`:

```csharp
private static float[] MeanPool(
    ReadOnlySpan<float> hiddenStates,    // flat [batch, seqLen, hiddenDim]
    ReadOnlySpan<long> attentionMask,    // flat [batch, seqLen]
    int batchIdx, int seqLen, int hiddenDim)
{
    var embedding = new float[hiddenDim];  // accumulator, initialized to 0
    float tokenCount = 0;

    for (int s = 0; s < seqLen; s++)
    {
        // Check attention mask — skip padding tokens
        if (attentionMask[batchIdx * seqLen + s] > 0)
        {
            // Compute offset into the flat hiddenStates array
            int offset = (batchIdx * seqLen + s) * hiddenDim;

            // Get a span view of this token's hidden state (hiddenDim floats)
            ReadOnlySpan<float> tokenEmbed = hiddenStates.Slice(offset, hiddenDim);

            // SIMD vectorized addition: embedding[i] += tokenEmbed[i]
            TensorPrimitives.Add(embedding, tokenEmbed, embedding);
            tokenCount++;
        }
    }

    // SIMD vectorized division: embedding[i] /= tokenCount
    if (tokenCount > 0)
        TensorPrimitives.Divide(embedding, tokenCount, embedding);

    return embedding;
}
```

### How SIMD Acceleration Works

When you call `TensorPrimitives.Add(embedding, tokenEmbed, embedding)`:

1. The runtime detects the CPU's SIMD capability (AVX2 on modern x64)
2. AVX2 processes **8 floats per instruction** using 256-bit SIMD registers
3. For a 384-dim embedding: 384 / 8 = **48 SIMD additions** instead of 384 scalar additions
4. This is roughly **6-8x faster** than a scalar loop

## Stable Softmax for Finite Logits

Classification, NER, and typed-decision decoding share a finite-logit softmax
kernel rather than calling `TensorPrimitives.SoftMax` directly. The shared
policy validates the temperature, restricts the operation to the valid option
slice, subtracts the maximum logit before exponentiation, and clears padded
output slots:

```csharp
var max = logits[..validCount].Max();
for (int i = 0; i < validCount; i++)
    probabilities[i] = MathF.Exp((logits[i] / temperature) - max);
TensorPrimitives.Divide(probabilities[..validCount], probabilities[..validCount].Sum(), probabilities[..validCount]);
probabilities[validCount..].Clear();
```

The production helper also rejects non-finite logits and invalid temperatures
instead of silently returning a uniform distribution. Typed-decision action
probabilities are already normalized by the graph and are not softmaxed again;
Noul decoding uses the true-class probability from that two-element output.

The developer writes simple, readable code — the SIMD optimization is automatic.

### The Offset Calculation

The flat `hiddenStates` array has shape `[batch, seqLen, hiddenDim]` stored in row-major order. The offset formula:

```
offset = (batchIdx × seqLen + seqPos) × hiddenDim
```

For batch item 1, sequence position 3, hiddenDim 384:
```
offset = (1 × 128 + 3) × 384 = 131 × 384 = 50,304
```

The `Slice(offset, hiddenDim)` call creates a `ReadOnlySpan<float>` view of exactly 384 contiguous floats at that position — no allocation or copy while the native result remains alive. Production code must snapshot values before disposing the result.

## L2 Normalization

After pooling, embeddings are optionally L2-normalized to unit length. This is standard for sentence-transformers and ensures cosine similarity equals dot product.

```csharp
private static void L2Normalize(Span<float> embedding)
{
    // Compute L2 norm: √(Σ embedding[i]²)
    float norm = TensorPrimitives.Norm(embedding);

    // Divide each element: embedding[i] /= norm
    if (norm > 0)
        TensorPrimitives.Divide(embedding, norm, embedding);
}
```

After normalization: `‖embedding‖₂ = 1.0`, which means:
```
cosine_similarity(a, b) = dot(a, b)    // when both are unit vectors
```

Both `Norm` and `Divide` are SIMD-accelerated.

## Cosine Similarity (Testing / Validation)

`TensorPrimitives.CosineSimilarity` provides a one-call similarity computation:

```csharp
float similarity = TensorPrimitives.CosineSimilarity(
    embedding1.AsSpan(),
    embedding2.AsSpan());
// Returns a float in [-1, 1]
// 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite
```

This is useful for validating embedding quality. In the sample app:
- "What is machine learning?" vs "ML.NET is a machine learning framework": **0.49** (related)
- "What is machine learning?" vs "How to cook pasta": **0.09** (unrelated)

## Summary: Zero-Copy Data Flow

The pipeline minimizes avoidable managed copies, but it is not universally zero-copy:

```
Tokenizer output (List<int>)
    │
    ▼ write through Tensor<T> indexing
flat long[] arrays (idsArray, maskArray)
    │
    ▼ OrtValue.CreateTensorValueFromMemory (CPU bind/pin where supported)
OrtValue (native input; providers may transfer to device)
    │
    ▼ InferenceSession.Run()
OrtValue output (native memory)
    │
    ▼ GetTensorDataAsSpan<float>() (borrowed span into native memory)
ReadOnlySpan<float>
    │
    ▼ Slice() views + TensorPrimitives SIMD math
float[] embedding (final result)
    │
    ▼ wrap (no copy)
Embedding<float> (MEAI) or VBuffer<float> (ML.NET)
```

Typical allocations include input arrays, padding/batch repacking buffers, output snapshots, pooled embeddings, and ML.NET VBuffer/row snapshots. The spans and tensor views avoid additional copies within their lifetime, but they do not outlive the native results they borrow, and GPU execution necessarily involves provider-managed transfers.
