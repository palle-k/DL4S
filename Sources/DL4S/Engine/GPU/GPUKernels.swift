//
//  GPUKernels.swift
//  DL4S
//
//  Created by Palle Klewitz on 24.09.26.
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

/// Records the kernels of the GPU. The functions do not check whether the host could compute the result in less time.
enum GPUKernels {
    static func pipeline(_ name: String, in source: GPUShaderSource) -> any MTLComputePipelineState {
        GPUContext.current.kernels.pipeline(name, in: source)
    }

    /// Allocates a GPU buffer for the given number of temporary 32-bit values.
    static func temporary(count: Int) -> GPUBuffer {
        let byteCount = count * 4
        return GPUBuffer(storage: GPUStorage(byteCount: byteCount), byteOffset: 0, byteCount: byteCount)
    }

    // MARK: Element-wise

    static func fill(_ result: GPUBuffer, word: UInt32, count: Int) {
        let pipeline = pipeline("fill_u32", in: .elementwise)
        GPUContext.current.compute(pipeline, reading: [], writing: [result]) { arguments in
            arguments.buffer(result)
            arguments.value(word)
            arguments.value(UInt32(count))
            arguments.dispatch(count: count)
        }
    }

    /// Maximum number of bytes that a command can contain, see ``write(_:to:)``.
    static let maximumImmediateByteCount = 4096

    /// Records a command that contains the given bytes and writes them to the result.
    static func write(_ bytes: UnsafeRawBufferPointer, to result: GPUBuffer) {
        precondition(bytes.count % 4 == 0 && bytes.count <= maximumImmediateByteCount, "A command contains at most 4096 bytes in 32-bit words.")
        let count = bytes.count / 4
        let pipeline = pipeline("copy_u32", in: .elementwise)
        GPUContext.current.compute(pipeline, reading: [], writing: [result]) { arguments in
            arguments.bytes(bytes)
            arguments.buffer(result)
            arguments.value(UInt32(count))
            arguments.dispatch(count: count)
        }
    }

    static func copyWords(from source: GPUBuffer, to result: GPUBuffer, byteCount: Int) {
        precondition(byteCount % 4 == 0, "Copies on the GPU move 32-bit words.")
        let count = byteCount / 4
        let pipeline = pipeline("copy_u32", in: .elementwise)
        GPUContext.current.compute(pipeline, reading: [source], writing: [result]) { arguments in
            arguments.buffer(source)
            arguments.buffer(result)
            arguments.value(UInt32(count))
            arguments.dispatch(count: count)
        }
    }

    static func unary(_ name: String, _ element: GPUElement, values: GPUBuffer, result: GPUBuffer, count: Int) {
        let pipeline = pipeline("\(name)_\(element.rawValue)", in: .elementwise)
        GPUContext.current.compute(pipeline, reading: [values], writing: [result]) { arguments in
            arguments.buffer(values)
            arguments.buffer(result)
            arguments.value(UInt32(count))
            arguments.dispatch(count: count)
        }
    }

    /// Form of the operands of a binary operator.
    enum BinaryForm: String {
        case vectorVector = "vv"
        case vectorScalar = "vs"
        case scalarVector = "sv"
    }

    static func binary(_ name: String, _ element: GPUElement, _ form: BinaryForm, lhs: GPUBuffer, rhs: GPUBuffer, result: GPUBuffer, count: Int) {
        let pipeline = pipeline("\(name)_\(form.rawValue)_\(element.rawValue)", in: .elementwise)
        GPUContext.current.compute(pipeline, reading: [lhs, rhs], writing: [result]) { arguments in
            arguments.buffer(lhs)
            arguments.buffer(rhs)
            arguments.buffer(result)
            arguments.value(UInt32(count))
            arguments.dispatch(count: count)
        }
    }

    /// Applies a binary operator to operands with broadcast strides. Operand 0 of the layout is the left operand, operand 1 the right operand.
    static func broadcast(_ name: String, _ element: GPUElement, lhs: GPUBuffer, rhs: GPUBuffer, result: GPUBuffer, layout: GPULayout, count: Int) {
        let pipeline = pipeline("\(name)_bc_\(element.rawValue)", in: .elementwise)
        GPUContext.current.compute(pipeline, reading: [lhs, rhs], writing: [result]) { arguments in
            arguments.buffer(lhs)
            arguments.buffer(rhs)
            arguments.buffer(result)
            arguments.value(UInt32(count))
            arguments.values(layout.arguments)
            arguments.dispatch(count: count)
        }
    }

    static func select(_ name: String, _ element: GPUElement, lhs: GPUBuffer, rhs: GPUBuffer, result: GPUBuffer, context: GPUBuffer, count: Int) {
        let pipeline = pipeline("\(name)_context_\(element.rawValue)", in: .elementwise)
        GPUContext.current.compute(pipeline, reading: [lhs, rhs], writing: [result, context]) { arguments in
            arguments.buffer(lhs)
            arguments.buffer(rhs)
            arguments.buffer(result)
            arguments.buffer(context)
            arguments.value(UInt32(count))
            arguments.dispatch(count: count)
        }
    }

    // MARK: Copies

    /// Copies a strided region of 32-bit elements, and adds a strided summand when one is given.
    ///
    /// - Parameters:
    ///   - source: Source, with the offset of the first element of the region.
    ///   - result: Result, with the offset of the first element of the region.
    ///   - summand: Summand, with the offset of its first element, and the element type for the addition.
    ///   - layout: Layout with the strides of the source, the result, and the summand.
    static func stridedCopy(source: GPUBuffer, result: GPUBuffer, summand: (buffer: GPUBuffer, element: GPUElement)? = nil, layout: GPULayout) {
        let count = layout.shape.reduce(1, *)
        guard count > 0 else {
            return
        }
        if let summand {
            let pipeline = pipeline("strided_copy_add_\(summand.element.rawValue)", in: .copy)
            GPUContext.current.compute(pipeline, reading: [source, summand.buffer], writing: [result]) { arguments in
                arguments.buffer(source)
                arguments.buffer(result)
                arguments.buffer(summand.buffer)
                arguments.values(layout.arguments)
                arguments.value(UInt32(count))
                arguments.dispatch(count: count)
            }
        } else {
            let pipeline = pipeline("strided_copy_u32", in: .copy)
            GPUContext.current.compute(pipeline, reading: [source], writing: [result]) { arguments in
                arguments.buffer(source)
                arguments.buffer(result)
                arguments.values(layout.arguments)
                arguments.value(UInt32(count))
                arguments.dispatch(count: count)
            }
        }
    }

    /// Transposes a batch of matrices of 32-bit elements with the given number of rows and columns.
    static func transpose(source: GPUBuffer, result: GPUBuffer, batch: Int, rows: Int, columns: Int) {
        let pipeline = pipeline("transpose_u32", in: .copy)
        GPUContext.current.compute(pipeline, reading: [source], writing: [result]) { arguments in
            arguments.buffer(source)
            arguments.buffer(result)
            arguments.value(SIMD2<UInt32>(UInt32(rows), UInt32(columns)))
            arguments.dispatch(
                threadgroups: MTLSize(width: (columns + 31) / 32, height: (rows + 31) / 32, depth: batch),
                threadgroup: MTLSize(width: 32, height: 8, depth: 1),
            )
        }
    }

    /// Copies a region of a buffer with the given shape into a contiguous result.
    static func copyRegion<Element>(of source: Buffer<Element, GPU>, shape: [Int], ranges: [Range<Int>], to result: MutableBuffer<Element, GPU>) {
        let strides = GPULayout.contiguousStrides(shape)
        let offset = zip(ranges, strides).map { $0.lowerBound * $1 }.reduce(0, +)
        let regionShape = ranges.map(\.count)
        copy(
            from: source.advanced(by: offset),
            strides: strides,
            to: result,
            strides: GPULayout.contiguousStrides(regionShape),
            shape: regionShape,
        )
    }

    /// Copies a contiguous source into a region of a buffer with the given shape.
    static func writeRegion<Element>(of result: MutableBuffer<Element, GPU>, shape: [Int], ranges: [Range<Int>], from source: Buffer<Element, GPU>) {
        let strides = GPULayout.contiguousStrides(shape)
        let offset = zip(ranges, strides).map { $0.lowerBound * $1 }.reduce(0, +)
        let regionShape = ranges.map(\.count)
        copy(
            from: source,
            strides: GPULayout.contiguousStrides(regionShape),
            to: result.advanced(by: offset),
            strides: strides,
            shape: regionShape,
        )
    }

    /// Copies elements between strided regions of two buffers, on the host for small regions and elements without GPU kernels.
    static func copy<Element>(from source: Buffer<Element, GPU>, strides sourceStrides: [Int], to result: MutableBuffer<Element, GPU>, strides resultStrides: [Int], shape: [Int]) {
        let count = shape.reduce(1, *)
        guard count > 0 else {
            return
        }
        let layout = GPULayout(shape: shape, strides: [sourceStrides, resultStrides])
        if MemoryLayout<Element>.stride == 4, layout.isSupported, !GPUPlacement.runsOnHost(elements: count, reading: [source.memory], writing: [result.memory]) {
            stridedCopy(source: source.memory, result: result.memory, layout: layout)
            return
        }
        let sourcePointer = source.host.memory.baseAddress!.assumingMemoryBound(to: Element.self)
        let resultPointer = result.host.memory.baseAddress!.assumingMemoryBound(to: Element.self)
        StridedIteration.forEachOffset(shape: layout.shape, strides: layout.strides[0], layout.strides[1]) { sourceOffset, resultOffset in
            resultPointer[resultOffset] = sourcePointer[sourceOffset]
        }
    }

    // MARK: Reductions

    enum Reduction: String {
        case sum
        case max
        case min
    }

    /// Reduces a buffer with the shape [outer, length, inner] along the middle axis into a buffer with the shape [outer, inner].
    ///
    /// - Parameters:
    ///   - context: Buffer for the indices of the maxima or minima along the reduced axis, or nil.
    ///   - scale: Factor for the result of a sum of floats.
    ///   - accumulate: Whether the reduced values are added to the current values of the result.
    static func reduce(_ reduction: Reduction, _ element: GPUElement, values: GPUBuffer, result: GPUBuffer, context: GPUBuffer? = nil, outer: Int, length: Int, inner: Int, scale: Float = 1, accumulate: Bool = false) {
        guard outer * inner > 0 else {
            return
        }
        precondition(length > 0, "A reduction needs at least one element.")
        if let context {
            precondition(reduction != .sum, "Only maxima and minima have a context.")
            if inner == 1, length >= 32 {
                reduceRows("reduce_rows_arg\(reduction.rawValue)_\(element.rawValue)", values: values, result: result, context: context, outer: outer, length: length, segments: 1, scale: scale)
            } else {
                reduceColumns("reduce_columns_arg\(reduction.rawValue)_\(element.rawValue)", values: values, result: result, context: context, outer: outer, length: length, inner: inner, segments: 1, scale: scale)
            }
            return
        }
        let rows = "reduce_rows_\(reduction.rawValue)_\(element.rawValue)"
        let columns = "reduce_columns_\(reduction.rawValue)_\(element.rawValue)"
        if inner == 1, length >= 32 {
            // Few long rows are split into segments, so that enough threadgroups run.
            guard outer < 64, length > 8192 else {
                reduceRows(rows, values: values, result: result, outer: outer, length: length, segments: 1, scale: scale, accumulate: accumulate)
                return
            }
            let segments = min((length + 4095) / 4096, 256)
            let partial = temporary(count: outer * segments)
            reduceRows(rows, values: values, result: partial, outer: outer, length: length, segments: segments, scale: 1)
            reduceRows(rows, values: partial, result: result, outer: outer, length: segments, segments: 1, scale: scale, accumulate: accumulate)
            return
        }
        let columnCount = outer * inner
        guard columnCount < 32768, length >= 128 else {
            reduceColumns(columns, values: values, result: result, outer: outer, length: length, inner: inner, segments: 1, scale: scale, accumulate: accumulate)
            return
        }
        let segmentLength = max(32, (length * columnCount + 65535) / 65536)
        let segments = (length + segmentLength - 1) / segmentLength
        let partial = temporary(count: segments * columnCount)
        reduceColumns(columns, values: values, result: partial, outer: outer, length: length, inner: inner, segments: segments, scale: 1)
        reduceColumns(columns, values: partial, result: result, outer: 1, length: segments, inner: columnCount, segments: 1, scale: scale, accumulate: accumulate)
    }

    private struct ReduceParameters {
        var outer: UInt32
        var length: UInt32
        var inner: UInt32
        var segmentLength: UInt32
        var segments: UInt32
        var scale: Float
        var accumulate: UInt32
    }

    private static func reduceRows(_ name: String, values: GPUBuffer, result: GPUBuffer, context: GPUBuffer? = nil, outer: Int, length: Int, segments: Int, scale: Float, accumulate: Bool = false) {
        let pipeline = pipeline(name, in: .reduction)
        let segmentLength = (length + segments - 1) / segments
        let outputs = outer * segments
        let parameters = ReduceParameters(outer: UInt32(outer), length: UInt32(length), inner: 1, segmentLength: UInt32(segmentLength), segments: UInt32(segments), scale: scale, accumulate: accumulate ? 1 : 0)
        // A SIMD group reduces a short segment, a threadgroup of 256 threads a long one.
        let width = segmentLength <= 1024 ? 32 : 256
        let height = width == 32 ? 8 : 1
        GPUContext.current.compute(pipeline, reading: accumulate ? [values, result] : [values], writing: [result] + (context.map { [$0] } ?? [])) { arguments in
            arguments.buffer(values)
            arguments.buffer(result)
            if let context {
                arguments.buffer(context)
            }
            arguments.value(parameters)
            arguments.dispatch(
                threadgroups: MTLSize(width: 1, height: (outputs + height - 1) / height, depth: 1),
                threadgroup: MTLSize(width: width, height: height, depth: 1),
            )
        }
    }

    private static func reduceColumns(_ name: String, values: GPUBuffer, result: GPUBuffer, context: GPUBuffer? = nil, outer: Int, length: Int, inner: Int, segments: Int, scale: Float, accumulate: Bool = false) {
        let pipeline = pipeline(name, in: .reduction)
        let segmentLength = (length + segments - 1) / segments
        let parameters = ReduceParameters(outer: UInt32(outer), length: UInt32(length), inner: UInt32(inner), segmentLength: UInt32(segmentLength), segments: UInt32(segments), scale: scale, accumulate: accumulate ? 1 : 0)
        let width = min(inner, 64)
        let height = max(1, min(outer, 256 / width))
        GPUContext.current.compute(pipeline, reading: accumulate ? [values, result] : [values], writing: [result] + (context.map { [$0] } ?? [])) { arguments in
            arguments.buffer(values)
            arguments.buffer(result)
            if let context {
                arguments.buffer(context)
            }
            arguments.value(parameters)
            arguments.dispatch(
                threads: MTLSize(width: inner, height: outer, depth: segments),
                threadgroup: MTLSize(width: width, height: height, depth: 1),
            )
        }
    }
}
#endif
