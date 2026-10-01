//
//  GPUFusedAttention.swift
//  DL4S
//
//  Created by Palle Klewitz on 29.09.26.
//  Copyright (c) 2026 - Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#if canImport(Metal) && canImport(MetalPerformanceShaders)
import Foundation
import Metal

// Attention runs the kernels of `attention.metal`, which keep the scores in registers, so no matrix of scores is in memory.
// The backward pass computes the result of the attention again, because the backward requirement does not receive it.
// Multi-head attention needs this result for the gradient of the output projection, so its backward pass gets it
// from the same kernel.
//
// Throughput with batch size 16, 8 heads, 512 queries and keys, and the head size 64 (M3 Max), against the default
// implementation, which writes the scores to memory:
// - Forward pass: 1.27 ms (6.8 TFLOPS), the default 3.0 ms. With a causal mask, which skips the blocks in which every
//   score is masked, 0.90 ms, the default 4.0 ms.
// - Backward pass: 4.8 ms, the default 7.3 ms. The kernels compute 7 products of the size of the scores, the default
//   computes 5, because the forward kernel computes the weights for the statistics of the rows and the key and value
//   kernel computes them again. The second kernel for the query gradient, which the key and value kernel replaces with
//   atomics, took 1.9 ms. With a causal mask 3.4 ms, the default 8.4 ms. With the head size 128, the default is faster,
//   see `backwardRunsKernels`.
// - Multi-head attention with 512 hidden elements: the forward pass 3.4 ms (86 percent of the throughput of the
//   matrix products of Metal Performance Shaders), the backward pass 11.0 ms. Splitting and joining the heads with
//   permutations took 0.8 ms in the forward pass and 1.5 ms in the backward pass.

public extension GPUFusedOperations {
    static func scaledDotProductAttention<N: NumericType>(queries: ShapedBuffer<N, GPU>, keys: ShapedBuffer<N, GPU>, values: ShapedBuffer<N, GPU>, mask: ShapedBuffer<N, GPU>?, temperature: N, result: MutableShapedBuffer<N, GPU>) {
        guard let attention = GPUAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature) else {
            matrixProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature, result: result)
            return
        }
        precondition(result.shape == attention.shape.resultShape, "The result must have the shape [batchSize, heads, queryCount, valueDim].")
        attention.forward(result: result.gpuBuffer)
    }

    static func scaledDotProductAttentionBackward<N: NumericType>(
        queries: ShapedBuffer<N, GPU>,
        keys: ShapedBuffer<N, GPU>,
        values: ShapedBuffer<N, GPU>,
        mask: ShapedBuffer<N, GPU>?,
        outputGradient: ShapedBuffer<N, GPU>,
        temperature: N,
        queryGradient: GradientBuffer<N, GPU>?,
        keyGradient: GradientBuffer<N, GPU>?,
        valueGradient: GradientBuffer<N, GPU>?,
    ) {
        guard let attention = GPUAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature), attention.backwardRunsKernels else {
            matrixProductAttentionBackward(queries: queries, keys: keys, values: values, mask: mask, outputGradient: outputGradient, temperature: temperature, queryGradient: queryGradient, keyGradient: keyGradient, valueGradient: valueGradient)
            return
        }
        precondition(outputGradient.shape == attention.shape.resultShape, "The gradient of the result must have the shape of the result.")
        attention.backward(outputGradient: outputGradient.gpuBuffer, output: nil, queryGradient: queryGradient, keyGradient: keyGradient, valueGradient: valueGradient)
    }

    // Multi-head attention with the kernels passes the projections to the attention in their layout, so that no heads are
    // split or joined: [batchSize, count, heads, size].

    static func multiHeadAttention<N: NumericType>(
        queries: ShapedBuffer<N, GPU>,
        keys: ShapedBuffer<N, GPU>,
        values: ShapedBuffer<N, GPU>,
        mask: ShapedBuffer<N, GPU>?,
        queryWeights: ShapedBuffer<N, GPU>,
        keyWeights: ShapedBuffer<N, GPU>,
        valueWeights: ShapedBuffer<N, GPU>,
        outputWeights: ShapedBuffer<N, GPU>,
        heads: Int,
        temperature: N,
        result: MutableShapedBuffer<N, GPU>,
    ) {
        let shape = kernelAttentionShape(queries: queries, keys: keys, values: values, mask: mask, queryWeights: queryWeights, keyWeights: keyWeights, valueWeights: valueWeights, outputWeights: outputWeights, heads: heads)
        projectedAttention(
            queries: queries,
            keys: keys,
            values: values,
            queryWeights: queryWeights,
            keyWeights: keyWeights,
            valueWeights: valueWeights,
            outputWeights: outputWeights,
            heads: heads,
            layout: shape == nil ? .split : .interleaved,
            result: result,
        ) { queries, keys, values, result in
            guard let shape else {
                scaledDotProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature, result: result)
                return
            }
            let attention = GPUAttention(shape: shape, layout: .interleaved(shape), queries: queries.gpuBuffer, keys: keys.gpuBuffer, values: values.gpuBuffer, mask: mask, temperature: temperature)
            attention.forward(result: result.gpuBuffer)
        }
    }

    static func multiHeadAttentionBackward<N: NumericType>(
        queries: ShapedBuffer<N, GPU>,
        keys: ShapedBuffer<N, GPU>,
        values: ShapedBuffer<N, GPU>,
        mask: ShapedBuffer<N, GPU>?,
        queryWeights: ShapedBuffer<N, GPU>,
        keyWeights: ShapedBuffer<N, GPU>,
        valueWeights: ShapedBuffer<N, GPU>,
        outputWeights: ShapedBuffer<N, GPU>,
        outputGradient: ShapedBuffer<N, GPU>,
        heads: Int,
        temperature: N,
        gradients: MultiHeadAttentionGradients<GradientBuffer<N, GPU>?>,
    ) {
        let shape = kernelAttentionShape(queries: queries, keys: keys, values: values, mask: mask, queryWeights: queryWeights, keyWeights: keyWeights, valueWeights: valueWeights, outputWeights: outputWeights, heads: heads)
            .flatMap { GPUAttention<N>.backwardRunsKernels(for: $0) ? $0 : nil }
        projectedAttentionBackward(
            queries: queries,
            keys: keys,
            values: values,
            queryWeights: queryWeights,
            keyWeights: keyWeights,
            valueWeights: valueWeights,
            outputWeights: outputWeights,
            outputGradient: outputGradient,
            heads: heads,
            layout: shape == nil ? .split : .interleaved,
            gradients: gradients,
        ) { queries, keys, values, outputGradient, output, queryGradient, keyGradient, valueGradient in
            if let shape {
                // The backward kernels compute the result of the attention, which the gradient of the output projection needs.
                let attention = GPUAttention(shape: shape, layout: .interleaved(shape), queries: queries.gpuBuffer, keys: keys.gpuBuffer, values: values.gpuBuffer, mask: mask, temperature: temperature)
                attention.backward(outputGradient: outputGradient.gpuBuffer, output: output, queryGradient: queryGradient, keyGradient: keyGradient, valueGradient: valueGradient)
                return
            }
            if let output {
                scaledDotProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature, result: output)
            }
            scaledDotProductAttentionBackward(queries: queries, keys: keys, values: values, mask: mask, outputGradient: outputGradient, temperature: temperature, queryGradient: queryGradient, keyGradient: keyGradient, valueGradient: valueGradient)
        }
    }
}

extension GPUFusedOperations {
    /// Shapes of the attention of the heads of multi-head attention, or nil when its kernels do not support them or the
    /// operation runs on the host.
    private static func kernelAttentionShape<N: NumericType>(
        queries: ShapedBuffer<N, GPU>,
        keys: ShapedBuffer<N, GPU>,
        values: ShapedBuffer<N, GPU>,
        mask: ShapedBuffer<N, GPU>?,
        queryWeights: ShapedBuffer<N, GPU>,
        keyWeights: ShapedBuffer<N, GPU>,
        valueWeights: ShapedBuffer<N, GPU>,
        outputWeights: ShapedBuffer<N, GPU>,
        heads: Int,
    ) -> AttentionShape? {
        let shape = MultiHeadAttentionShape(queries: queries, keys: keys, values: values, queryWeights: queryWeights, keyWeights: keyWeights, valueWeights: valueWeights, outputWeights: outputWeights, heads: heads).attention
        let reading = [queries, keys, values, queryWeights, keyWeights, valueWeights, outputWeights].map(\.gpuBuffer) + (mask.map { [$0.gpuBuffer] } ?? [])
        guard GPUAttention<N>.supports(shape),
              GPUFused.runsKernel(N.self, elements: shape.scoreShape.reduce(1, *), reading: reading)
        else {
            return nil
        }
        return shape
    }
}

/// Scaled dot product attention with the kernels of `attention.metal`.
private struct GPUAttention<N: NumericType> {
    /// Threadgroups of the kernels for a head size: the rows of queries or keys and the threads of a threadgroup.
    /// The values follow from the template arguments of the kernels in `attention.metal`, `ATTENTION_KERNELS`.
    private struct Configuration {
        let forward: AttentionThreadgroup
        let keyValueGradient: AttentionThreadgroup
        let queryGradient: AttentionThreadgroup
        /// Whether `attention_gradients` exists for the head size, which adds the query gradient in the key and value kernel.
        let fusesQueryGradient: Bool

        /// Number of queries of a block of the key and value kernels, `KB` in `ATTENTION_KERNELS`
        static var keyValueQueryBlock: Int {
            16
        }

        static func forHeadSize(_ size: Int) -> Configuration? {
            switch size {
            case 32:
                Configuration(forward: AttentionThreadgroup(rowsPerSIMDGroup: 16, simdGroups: 4), keyValueGradient: AttentionThreadgroup(rowsPerSIMDGroup: 16, simdGroups: 4), queryGradient: AttentionThreadgroup(rowsPerSIMDGroup: 16, simdGroups: 4), fusesQueryGradient: true)
            case 64:
                Configuration(forward: AttentionThreadgroup(rowsPerSIMDGroup: 16, simdGroups: 4), keyValueGradient: AttentionThreadgroup(rowsPerSIMDGroup: 8, simdGroups: 8), queryGradient: AttentionThreadgroup(rowsPerSIMDGroup: 8, simdGroups: 8), fusesQueryGradient: true)
            case 128:
                Configuration(forward: AttentionThreadgroup(rowsPerSIMDGroup: 16, simdGroups: 4), keyValueGradient: AttentionThreadgroup(rowsPerSIMDGroup: 8, simdGroups: 8), queryGradient: AttentionThreadgroup(rowsPerSIMDGroup: 8, simdGroups: 8), fusesQueryGradient: false)
            default:
                nil
            }
        }
    }

    /// The tiles of 8 x 8 scores of a mask, for which `attention_mask_tiles` writes whether every score is masked. The tiles
    /// have the batch and head axes of the mask, and one tile row or column along a query or key axis that broadcasts.
    private struct MaskTiles {
        let batchSize: Int
        let heads: Int
        let queryTiles: Int
        let keyTiles: Int

        init(maskShape: [Int], attention: AttentionShape) {
            let shape = Array(repeating: 1, count: 4 - maskShape.count) + maskShape
            (batchSize, heads) = (shape[0], shape[1])
            queryTiles = shape[2] == 1 ? 1 : (attention.queryCount + 7) / 8
            keyTiles = shape[3] == 1 ? 1 : (attention.keyCount + 7) / 8
        }

        var count: Int {
            batchSize * heads * queryTiles * keyTiles
        }

        /// Strides along the batch, head, query tile, and key tile axes, 0 for axes along which the tiles broadcast.
        var strides: [Int32] {
            ShapeUtil.broadcastStrides([batchSize, heads, queryTiles, keyTiles]).map { Int32($0) }
        }
    }

    /// Strides of the batch, head, and row axes of the operands in elements: of the queries, the result, and their gradients,
    /// and of the keys, the values, and their gradients.
    struct Layout {
        let query: [Int]
        let key: [Int]

        /// The layout of the requirement: [batchSize, heads, count, size]. Keys that broadcast along the batch have the batch stride 0.
        static func heads(_ shape: AttentionShape) -> Layout {
            let keyBatchStride = shape.keyBatchSize == 1 ? 0 : shape.keyHeads * shape.keyCount * shape.keyDim
            return Layout(query: [shape.heads * shape.queryCount * shape.keyDim, shape.queryCount * shape.keyDim, shape.keyDim], key: [keyBatchStride, shape.keyCount * shape.keyDim, shape.keyDim])
        }

        /// The layout of the projections of multi-head attention: [batchSize, count, heads, size].
        static func interleaved(_ shape: AttentionShape) -> Layout {
            let (queryRow, keyRow) = (shape.heads * shape.keyDim, shape.keyHeads * shape.keyDim)
            return Layout(query: [shape.queryCount * queryRow, shape.keyDim, queryRow], key: [shape.keyCount * keyRow, shape.keyDim, keyRow])
        }
    }

    /// Buffers of the statistics that the forward kernel writes for the backward kernels, shape [batchSize, heads, queryCount].
    private struct Statistics {
        /// Largest score of every row in the base 2 domain
        let maximum: GPUBuffer
        /// Inverse of the sum of the exponentials of every row
        let inverseSum: GPUBuffer
        /// Sum of the products of every row of the result with its gradient
        let gradientDot: GPUBuffer
        /// Gradient of the result, which the forward kernel reads for the sums of the products
        let outputGradient: GPUBuffer
    }

    let shape: AttentionShape
    let queries: GPUBuffer
    let keys: GPUBuffer
    let values: GPUBuffer
    let mask: GPUBuffer?
    private let maskTiles: MaskTiles?
    private let configuration: Configuration
    private let parameters: AttentionParameters

    /// The attention of contiguous operands with the shapes of the requirement, or nil when the kernels do not support them
    /// or the operation runs on the host.
    init?(queries: ShapedBuffer<N, GPU>, keys: ShapedBuffer<N, GPU>, values: ShapedBuffer<N, GPU>, mask: ShapedBuffer<N, GPU>?, temperature: N) {
        let shape = AttentionShape(queries: queries, keys: keys, values: values)
        guard Self.supports(shape), GPUFused.runsKernel(N.self, elements: shape.scoreShape.reduce(1, *), reading: [queries.gpuBuffer, keys.gpuBuffer, values.gpuBuffer] + (mask.map { [$0.gpuBuffer] } ?? []))
        else {
            return nil
        }
        self.init(shape: shape, layout: .heads(shape), queries: queries.gpuBuffer, keys: keys.gpuBuffer, values: values.gpuBuffer, mask: mask, temperature: temperature)
    }

    /// The attention of operands with the given shapes and layout, which the kernels support.
    init(shape: AttentionShape, layout: Layout, queries: GPUBuffer, keys: GPUBuffer, values: GPUBuffer, mask: ShapedBuffer<N, GPU>?, temperature: N) {
        precondition(Self.supports(shape), "The attention kernels do not support the shapes.")
        if let mask {
            precondition(ShapeUtil.broadcasts(mask.shape, to: shape.scoreShape), "The mask must be broadcastable to the shape of the scores.")
        }
        self.shape = shape
        configuration = Configuration.forHeadSize(shape.keyDim)!
        (self.queries, self.keys, self.values, self.mask) = (queries, keys, values, mask?.gpuBuffer)
        maskTiles = mask.map { MaskTiles(maskShape: $0.shape, attention: shape) }

        let maskStrides = (mask.map { ShapeUtil.broadcastStrides(Array(repeating: 1, count: 4 - $0.dim) + $0.shape) } ?? [0, 0, 0, 0]).map { Int32($0) }
        let tileStrides = maskTiles?.strides ?? [0, 0, 0, 0]
        let (queryStrides, keyStrides) = (layout.query.map(Int32.init), layout.key.map(Int32.init))
        let log2e = Float(M_LOG2E)
        parameters = AttentionParameters(
            queryCount: Int32(shape.queryCount),
            keyCount: Int32(shape.keyCount),
            heads: Int32(shape.heads),
            keyGroup: Int32(shape.heads / shape.keyHeads),
            batchSize: Int32(shape.batchSize),
            scale: log2e / temperature.floatValue,
            maskScale: 1e9 * log2e,
            gradientScale: 1 / temperature.floatValue,
            queryStrides: (queryStrides[0], queryStrides[1], queryStrides[2]),
            keyStrides: (keyStrides[0], keyStrides[1], keyStrides[2]),
            maskStrides: (maskStrides[0], maskStrides[1], maskStrides[2], maskStrides[3]),
            tileStrides: (tileStrides[0], tileStrides[1], tileStrides[2], tileStrides[3]),
            hasMask: mask == nil ? 0 : 1,
            accumulate: (0, 0, 0),
            computes: (0, 0, 0),
            splits: 1,
        )
    }

    /// Whether the kernels support the shapes: equal key and value sizes of 32, 64, or 128, keys and values with the same
    /// number of heads and batch size, and queries with the batch size of the result.
    static func supports(_ shape: AttentionShape) -> Bool {
        Configuration.forHeadSize(shape.keyDim) != nil && shape.valueDim == shape.keyDim && shape.valueHeads == shape.keyHeads
            && shape.queryBatchSize == shape.batchSize && shape.valueBatchSize == shape.keyBatchSize
    }

    /// Whether the backward pass runs the kernels: with head sizes below 128, and with larger matrices of scores than 2^26
    /// elements.
    var backwardRunsKernels: Bool {
        Self.backwardRunsKernels(for: shape)
    }

    // With the head size 128, the backward kernels keep four rows of 128 elements for every key in registers and reach about
    // 5 TFLOPS. The default implementation, whose matrix products are efficient at this size, is faster: 10.4 ms against
    // 14.3 ms with batch size 16, 8 heads, and 512 queries and keys. It allocates three matrices of scores, so the kernels
    // run for large matrices.
    static func backwardRunsKernels(for shape: AttentionShape) -> Bool {
        shape.keyDim < 128 || shape.scoreShape.reduce(1, *) > 1 << 26
    }

    /// Records the forward pass.
    func forward(result: GPUBuffer) {
        encodeForward(result: result, statistics: nil, zeros: nil, tiles: encodeMaskTiles())
    }

    /// Records the backward pass, and writes the result of the attention into `output` when it is not nil.
    func backward(outputGradient gradient: GPUBuffer, output: MutableShapedBuffer<N, GPU>?, queryGradient: GradientBuffer<N, GPU>?, keyGradient: GradientBuffer<N, GPU>?, valueGradient: GradientBuffer<N, GPU>?) {
        let rows = shape.batchSize * shape.heads * shape.queryCount
        let statistics = Statistics(maximum: GPUKernels.temporary(count: rows), inverseSum: GPUKernels.temporary(count: rows), gradientDot: GPUKernels.temporary(count: rows), outputGradient: gradient)
        // With the head sizes 32 and 64, the key and value kernel also adds the query gradient with atomics, so that no second
        // kernel computes the weights again. The forward kernel sets the query gradient to zero when it does not accumulate.
        let fusedQueryGradient = configuration.fusesQueryGradient ? queryGradient : nil
        let zeros = fusedQueryGradient?.adds == false ? fusedQueryGradient?.gpuBuffer : nil
        let tiles = encodeMaskTiles()
        encodeForward(result: output?.gpuBuffer, statistics: statistics, zeros: zeros, tiles: tiles)

        var parameters = parameters
        parameters.accumulate = (Int32(queryGradient?.accumulateFlag ?? 0), Int32(keyGradient?.accumulateFlag ?? 0), Int32(valueGradient?.accumulateFlag ?? 0))
        parameters.computes = (queryGradient == nil ? 0 : 1, keyGradient == nil ? 0 : 1, valueGradient == nil ? 0 : 1)
        let context = GPUContext.current
        let size = shape.keyDim
        let common = [queries, keys, values, statistics.maximum, statistics.inverseSum, statistics.gradientDot, gradient] + ([mask, tiles].compactMap(\.self))

        let kernelGradients = [keyGradient, valueGradient, fusedQueryGradient]
        let kernel = configuration.keyValueGradient
        let keyBlocks = (shape.keyCount + kernel.rows - 1) / kernel.rows
        parameters.splits = Int32(keyValueSplits(keyBlocks: keyBlocks))
        if parameters.splits > 1 {
            // The threadgroups that share the queries add the key and value gradients with atomics.
            for gradient in [keyGradient, valueGradient].compactMap(\.self) where !gradient.adds {
                GPUEngine.fill(value: N.zero, result: gradient.values.values, count: gradient.values.count)
            }
        }
        if kernelGradients.contains(where: { $0 != nil }) {
            // A gradient that is not requested is not written, so the kernel receives another gradient in its place.
            let targets = kernelGradients.compactMap { $0?.gpuBuffer }
            let (dk, dv) = (keyGradient?.gpuBuffer ?? targets[0], valueGradient?.gpuBuffer ?? targets[0])
            let dq = fusedQueryGradient?.gpuBuffer ?? targets[0]
            // The atomics read the gradients that they add to.
            let atomics = parameters.splits > 1
            let accumulated = [keyGradient, valueGradient].compactMap { $0?.adds == true || atomics ? $0?.gpuBuffer : nil } + (fusedQueryGradient == nil ? [] : [dq])
            let threadgroups = MTLSize(width: keyBlocks, height: shape.keyHeads, depth: shape.keyBatchSize * Int(parameters.splits))
            let name = fusedQueryGradient == nil ? "attention_key_value_gradient_\(size)" : "attention_gradients_\(size)"
            context.compute(GPUKernels.pipeline(name, in: .attention), reading: common + accumulated, writing: targets) { arguments in
                encodeInputs(&arguments, statistics: statistics)
                arguments.buffer(dk)
                arguments.buffer(dv)
                arguments.value(parameters)
                arguments.buffer(tiles ?? queries)
                arguments.buffer(dq)
                arguments.dispatch(threadgroups: threadgroups, threadgroup: MTLSize(width: kernel.threads, height: 1, depth: 1))
            }
        }
        if let queryGradient, fusedQueryGradient == nil {
            let dq = queryGradient.gpuBuffer
            let kernel = configuration.queryGradient
            let threadgroups = MTLSize(width: (shape.queryCount + kernel.rows - 1) / kernel.rows, height: shape.heads, depth: shape.batchSize)
            context.compute(GPUKernels.pipeline("attention_query_gradient_\(size)", in: .attention), reading: common + (queryGradient.adds ? [dq] : []), writing: [dq]) { arguments in
                encodeInputs(&arguments, statistics: statistics)
                arguments.buffer(dq)
                arguments.value(parameters)
                arguments.buffer(tiles ?? queries)
                arguments.dispatch(threadgroups: threadgroups, threadgroup: MTLSize(width: kernel.threads, height: 1, depth: 1))
            }
        }
    }

    // Keys that many query heads share, as in grouped-query attention, give few threadgroups, of which the last do not occupy
    // all GPU cores. With fewer than 16 query blocks, loading the keys and adding the gradients take longer than the products.
    /// Number of threadgroups that share the query blocks of a block of keys in the key and value kernel, so that about 1024
    /// threadgroups run and every threadgroup keeps at least 16 query blocks.
    private func keyValueSplits(keyBlocks: Int) -> Int {
        let threadgroups = keyBlocks * shape.keyHeads * shape.keyBatchSize
        let queryBlocks = (shape.queryCount + Configuration.keyValueQueryBlock - 1) / Configuration.keyValueQueryBlock
        // Keys that broadcast along the batch serve the queries of every batch.
        let batches = shape.keyBatchSize == 1 ? shape.batchSize : 1
        let items = batches * (shape.heads / shape.keyHeads) * queryBlocks
        return max(1, min(1024 / threadgroups, items / 16))
    }

    /// Records the forward kernel, which writes the result when it is not nil, the statistics of every row when they are not
    /// nil, and zeros into `zeros`, which has the layout of the result, when it is not nil.
    private func encodeForward(result: GPUBuffer?, statistics: Statistics?, zeros: GPUBuffer?, tiles: GPUBuffer?) {
        let reading = [queries, keys, values] + [mask, tiles, statistics?.outputGradient].compactMap(\.self)
        let writing = [result, zeros].compactMap(\.self) + (statistics.map { [$0.maximum, $0.inverseSum, $0.gradientDot] } ?? [])
        let outputs = (result == nil ? 0 : 1) | (statistics == nil ? 0 : 2) | (zeros == nil ? 0 : 4)
        let kernel = configuration.forward
        let threadgroups = MTLSize(width: (shape.queryCount + kernel.rows - 1) / kernel.rows, height: shape.heads, depth: shape.batchSize)
        GPUContext.current.compute(GPUKernels.pipeline("attention_forward_\(shape.keyDim)", in: .attention), reading: reading, writing: writing) { arguments in
            arguments.buffer(queries)
            arguments.buffer(keys)
            arguments.buffer(values)
            // The kernel does not write the buffers of outputs that it does not compute.
            arguments.buffer(mask ?? queries)
            arguments.buffer(result ?? queries)
            arguments.buffer(statistics?.maximum ?? queries)
            arguments.buffer(statistics?.inverseSum ?? queries)
            arguments.buffer(statistics?.gradientDot ?? queries)
            arguments.buffer(statistics?.outputGradient ?? queries)
            arguments.value(parameters)
            arguments.value(Int32(outputs))
            arguments.buffer(tiles ?? queries)
            arguments.buffer(zeros ?? queries)
            arguments.dispatch(threadgroups: threadgroups, threadgroup: MTLSize(width: kernel.threads, height: 1, depth: 1))
        }
    }

    /// Records `attention_mask_tiles` and returns its tiles, or returns nil without a mask.
    private func encodeMaskTiles() -> GPUBuffer? {
        guard let mask, let maskTiles else {
            return nil
        }
        let tiles = GPUKernels.temporary(byteCount: maskTiles.count)
        let maskShape = SIMD2<Int32>(Int32(maskTiles.batchSize), Int32(maskTiles.heads))
        GPUContext.current.compute(GPUKernels.pipeline("attention_mask_tiles", in: .attention), reading: [mask], writing: [tiles]) { arguments in
            arguments.buffer(mask)
            arguments.buffer(tiles)
            arguments.value(parameters)
            arguments.value(maskShape)
            arguments.dispatch(threads: MTLSize(width: maskTiles.keyTiles, height: maskTiles.queryTiles, depth: maskTiles.batchSize * maskTiles.heads), threadgroup: MTLSize(width: 8, height: 8, depth: 1))
        }
        return tiles
    }

    /// Sets the queries, keys, values, mask, the statistics of the forward pass, and the gradient of the result, the first
    /// arguments of both backward kernels.
    private func encodeInputs(_ arguments: inout GPUArguments, statistics: Statistics) {
        arguments.buffer(queries)
        arguments.buffer(keys)
        arguments.buffer(values)
        arguments.buffer(mask ?? queries)
        arguments.buffer(statistics.maximum)
        arguments.buffer(statistics.inverseSum)
        arguments.buffer(statistics.gradientDot)
        arguments.buffer(statistics.outputGradient)
    }
}

/// The threadgroup of an attention kernel: the rows of queries or keys that it computes and its threads.
private struct AttentionThreadgroup {
    let rows: Int
    let threads: Int

    init(rowsPerSIMDGroup: Int, simdGroups: Int) {
        rows = rowsPerSIMDGroup * simdGroups
        threads = 32 * simdGroups
    }
}

/// The parameters of the kernels, with the layout of `AttentionParameters` in `attention.metal`.
private struct AttentionParameters {
    var queryCount: Int32
    var keyCount: Int32
    var heads: Int32
    var keyGroup: Int32
    var batchSize: Int32
    var scale: Float
    var maskScale: Float
    var gradientScale: Float
    var queryStrides: (Int32, Int32, Int32)
    var keyStrides: (Int32, Int32, Int32)
    var maskStrides: (Int32, Int32, Int32, Int32)
    var tileStrides: (Int32, Int32, Int32, Int32)
    var hasMask: Int32
    var accumulate: (Int32, Int32, Int32)
    var computes: (Int32, Int32, Int32)
    var splits: Int32
}
#endif
