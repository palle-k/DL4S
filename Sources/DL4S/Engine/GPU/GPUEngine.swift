//
//  GPUEngine.swift
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

/// Basic operations of the GPU.
///
/// Every operation records a GPU kernel, or runs the CPU implementation on the shared memory:
/// for element types without GPU kernels, and for small operations whose operands are available on the host.
/// A CPU implementation waits for the GPU work that its operands depend on.
public struct GPUEngine: EngineType {
    public typealias Device = GPU

    /// The GPU element type of `N` when the operation runs on the GPU, or nil when it runs on the host.
    @inline(__always)
    private static func gpuElement<N>(_: N.Type, elements: Int, reading: [GPUBuffer], writing: [GPUBuffer]) -> GPUElement? {
        guard let element = GPUElement(of: N.self), !GPUPlacement.runsOnHost(elements: elements, reading: reading, writing: writing) else {
            return nil
        }
        return element
    }

    public static func fill<N: NumericType>(value: N, result: MutableBuffer<N, GPU>, count: Int) {
        guard gpuElement(N.self, elements: count, reading: [], writing: [result.memory]) != nil else {
            CPUEngine.fill(value: value, result: result.host, count: count)
            return
        }
        GPUKernels.fill(result.memory, word: withUnsafeBytes(of: value) { $0.load(as: UInt32.self) }, count: count)
    }

    // MARK: Vector operations

    private static func vector<N: NumericType>(_ name: String, lhs: Buffer<N, GPU>, rhs: Buffer<N, GPU>, result: MutableBuffer<N, GPU>, count: Int, host: (Buffer<N, CPU>, Buffer<N, CPU>, MutableBuffer<N, CPU>, Int) -> Void) {
        guard let element = gpuElement(N.self, elements: count, reading: [lhs.memory, rhs.memory], writing: [result.memory]) else {
            host(lhs.host, rhs.host, result.host, count)
            return
        }
        GPUKernels.binary(name, element, .vectorVector, lhs: lhs.memory, rhs: rhs.memory, result: result.memory, count: count)
    }

    public static func vAdd<N: NumericType>(lhs: Buffer<N, GPU>, rhs: Buffer<N, GPU>, result: MutableBuffer<N, GPU>, count: Int) {
        vector("add", lhs: lhs, rhs: rhs, result: result, count: count, host: CPUEngine.vAdd)
    }

    public static func vNeg<N: NumericType>(val: Buffer<N, GPU>, result: MutableBuffer<N, GPU>, count: Int) {
        guard let element = gpuElement(N.self, elements: count, reading: [val.memory], writing: [result.memory]) else {
            CPUEngine.vNeg(val: val.host, result: result.host, count: count)
            return
        }
        GPUKernels.unary("neg", element, values: val.memory, result: result.memory, count: count)
    }

    public static func vSub<N: NumericType>(lhs: Buffer<N, GPU>, rhs: Buffer<N, GPU>, result: MutableBuffer<N, GPU>, count: Int) {
        vector("sub", lhs: lhs, rhs: rhs, result: result, count: count, host: CPUEngine.vSub)
    }

    public static func vMul<N: NumericType>(lhs: Buffer<N, GPU>, rhs: Buffer<N, GPU>, result: MutableBuffer<N, GPU>, count: Int) {
        vector("mul", lhs: lhs, rhs: rhs, result: result, count: count, host: CPUEngine.vMul)
    }

    public static func vDiv<N: NumericType>(lhs: Buffer<N, GPU>, rhs: Buffer<N, GPU>, result: MutableBuffer<N, GPU>, count: Int) {
        vector("div", lhs: lhs, rhs: rhs, result: result, count: count, host: CPUEngine.vDiv)
    }

    // MARK: Matrix operations

    public static func fillDiagonal<N: NumericType>(values: ShapedBuffer<N, GPU>, target: MutableShapedBuffer<N, GPU>) {
        precondition(values.dim == 1, "values must be a vector")
        precondition(target.dim == 2, "target must be a matrix")
        let count = Swift.min(target.shape[0], target.shape[1])
        precondition(values.count == count, "number of values must be equal to smaller dimension of matrix")
        guard gpuElement(N.self, elements: count, reading: [values.values.memory], writing: [target.values.memory]) != nil else {
            CPUEngine.fillDiagonal(values: values.host, target: target.host)
            return
        }
        diagonal(values: values.values.memory, result: target.values.memory, count: count, stride: target.shape[1] + 1, extracts: false)
    }

    public static func fillDiagonal<N: NumericType>(value: N, target: MutableShapedBuffer<N, GPU>) {
        precondition(target.dim == 2, "target must be a matrix")
        let count = Swift.min(target.shape[0], target.shape[1])
        guard gpuElement(N.self, elements: count, reading: [], writing: [target.values.memory]) != nil else {
            CPUEngine.fillDiagonal(value: value, target: target.host)
            return
        }
        let word = withUnsafeBytes(of: value) { $0.load(as: UInt32.self) }
        let pipeline = GPUKernels.pipeline("fill_diagonal_u32", in: .elementwise)
        let result = target.values.memory
        GPUContext.current.compute(pipeline, reading: [], writing: [result]) { arguments in
            arguments.buffer(result)
            arguments.value(SIMD3<UInt32>(UInt32(count), UInt32(target.shape[1] + 1), word))
            arguments.dispatch(count: count)
        }
    }

    public static func extractDiagonal<N: NumericType>(values: ShapedBuffer<N, GPU>, target: MutableShapedBuffer<N, GPU>) {
        precondition(target.dim == 1, "values must be a vector")
        precondition(values.dim == 2, "target must be a matrix")
        let count = Swift.min(values.shape[0], values.shape[1])
        precondition(target.count == count, "number of values must be equal to smaller dimension of matrix")
        guard gpuElement(N.self, elements: count, reading: [values.values.memory], writing: [target.values.memory]) != nil else {
            CPUEngine.extractDiagonal(values: values.host, target: target.host)
            return
        }
        // The stride matches the CPU implementation, which reads the elements i * (count + 1).
        diagonal(values: values.values.memory, result: target.values.memory, count: count, stride: target.shape[0] + 1, extracts: true)
    }

    private static func diagonal(values: GPUBuffer, result: GPUBuffer, count: Int, stride: Int, extracts: Bool) {
        let pipeline = GPUKernels.pipeline("diagonal_u32", in: .elementwise)
        GPUContext.current.compute(pipeline, reading: [values], writing: [result]) { arguments in
            arguments.buffer(values)
            arguments.buffer(result)
            arguments.value(SIMD3<UInt32>(UInt32(count), UInt32(stride), extracts ? 1 : 0))
            arguments.dispatch(count: count)
        }
    }

    public static func gemm<N: NumericType>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool) {
        let rows = result.shape[0]
        let columns = result.shape[1]
        let inner = transposeFirst ? lhs.shape[0] : lhs.shape[1]
        // A product runs on the host when it has at most as many multiplications as a small matrix of 64 x 64 x 64 elements.
        let multiplications = rows * columns * inner
        guard N.self == Float.self,
              !GPUPlacement.runsOnHost(elements: multiplications / 64, reading: [lhs.values.memory, rhs.values.memory], writing: [result.values.memory])
        else {
            CPUEngine.gemm(lhs: lhs.host, rhs: rhs.host, result: result.host, alpha: alpha, beta: beta, transposeFirst: transposeFirst, transposeSecond: transposeSecond)
            return
        }
        GPUMatrixMultiplication.encode(
            lhs: lhs.values.memory,
            lhsShape: lhs.shape,
            rhs: rhs.values.memory,
            rhsShape: rhs.shape,
            result: result.values.memory,
            rows: rows,
            columns: columns,
            inner: inner,
            alpha: alpha.floatValue,
            beta: beta.floatValue,
            transposeFirst: transposeFirst,
            transposeSecond: transposeSecond,
        )
    }

    public static func gemmBatched<N: NumericType>(lhs: ShapedBuffer<N, GPU>, lhsStride: Int, rhs: ShapedBuffer<N, GPU>, rhsStride: Int, result: MutableShapedBuffer<N, GPU>, count: Int, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool) {
        let rows = result.shape[0]
        let columns = result.shape[1]
        let inner = transposeFirst ? lhs.shape[0] : lhs.shape[1]
        let multiplications = rows * columns * inner * count
        guard N.self == Float.self,
              !GPUPlacement.runsOnHost(elements: multiplications / 64, reading: [lhs.values.memory, rhs.values.memory], writing: [result.values.memory])
        else {
            CPUEngine.gemmBatched(lhs: lhs.host, lhsStride: lhsStride, rhs: rhs.host, rhsStride: rhsStride, result: result.host, count: count, alpha: alpha, beta: beta, transposeFirst: transposeFirst, transposeSecond: transposeSecond)
            return
        }
        GPUMatrixMultiplication.encodeBatch(
            lhs: lhs.values.memory, rhs: rhs.values.memory, result: result.values.memory,
            count: count, rows: rows, columns: columns, inner: inner,
            lhsColumns: lhs.shape[1], rhsColumns: rhs.shape[1], strides: (lhsStride, rhsStride, rows * columns),
            alpha: alpha.floatValue, beta: beta.floatValue, transposeFirst: transposeFirst, transposeSecond: transposeSecond,
        )
    }

    public static func band<N: NumericType>(buffer: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, belowDiagonal: Int?, aboveDiagonal: Int?) {
        precondition(buffer.shape == result.shape, "Shape of result must be equal to shape of buffer.")
        precondition(buffer.dim == 2, "Band can only be computed on tensor of dimensionality 2.")
        guard gpuElement(N.self, elements: buffer.count, reading: [buffer.values.memory], writing: [result.values.memory]) != nil else {
            CPUEngine.band(buffer: buffer.host, result: result.host, belowDiagonal: belowDiagonal, aboveDiagonal: aboveDiagonal)
            return
        }
        let (rows, columns) = (buffer.shape[0], buffer.shape[1])
        let limit = Swift.max(rows, columns)
        let pipeline = GPUKernels.pipeline("band_u32", in: .elementwise)
        let (values, target) = (buffer.values.memory, result.values.memory)
        GPUContext.current.compute(pipeline, reading: [values], writing: [target]) { arguments in
            arguments.buffer(values)
            arguments.buffer(target)
            arguments.value(SIMD4<Int32>(Int32(rows), Int32(columns), Int32(belowDiagonal ?? limit), Int32(aboveDiagonal ?? limit)))
            arguments.dispatch(threads: .init(width: columns, height: rows, depth: 1), threadgroup: .init(width: 32, height: 8, depth: 1))
        }
    }

    // MARK: Broadcasting

    private static func broadcast<N: NumericType>(
        _ name: String,
        lhs: ShapedBuffer<N, GPU>,
        rhs: ShapedBuffer<N, GPU>,
        result: MutableShapedBuffer<N, GPU>,
        host: (ShapedBuffer<N, CPU>, ShapedBuffer<N, CPU>, MutableShapedBuffer<N, CPU>) -> Void,
    ) {
        let count = result.count
        guard let element = gpuElement(N.self, elements: count, reading: [lhs.values.memory, rhs.values.memory], writing: [result.values.memory]) else {
            host(lhs.host, rhs.host, result.host)
            return
        }
        let (a, b, target) = (lhs.values.memory, rhs.values.memory, result.values.memory)
        if lhs.count == count, rhs.count == count {
            GPUKernels.binary(name, element, .vectorVector, lhs: a, rhs: b, result: target, count: count)
        } else if lhs.count == count, rhs.count == 1 {
            GPUKernels.binary(name, element, .vectorScalar, lhs: a, rhs: b, result: target, count: count)
        } else if lhs.count == 1, rhs.count == count {
            GPUKernels.binary(name, element, .scalarVector, lhs: a, rhs: b, result: target, count: count)
        } else {
            let layout = GPULayout(shape: result.shape, strides: [
                broadcastStrides(of: lhs.shape, dim: result.dim),
                broadcastStrides(of: rhs.shape, dim: result.dim),
            ])
            guard layout.isSupported else {
                host(lhs.host, rhs.host, result.host)
                return
            }
            GPUKernels.broadcast(name, element, lhs: a, rhs: b, result: target, layout: layout, count: count)
        }
    }

    /// Row-major strides of a shape that is padded with ones to the given dimensionality, with the stride 0 for axes of size 1.
    private static func broadcastStrides(of shape: [Int], dim: Int) -> [Int] {
        let padded = [Int](repeating: 1, count: dim - shape.count) + shape
        return zip(padded, GPULayout.contiguousStrides(padded)).map { $0 == 1 ? 0 : $1 }
    }

    public static func broadcastAdd<N: NumericType>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        broadcast("add", lhs: lhs, rhs: rhs, result: result, host: CPUEngine.broadcastAdd)
    }

    public static func broadcastSub<N: NumericType>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        broadcast("sub", lhs: lhs, rhs: rhs, result: result, host: CPUEngine.broadcastSub)
    }

    public static func broadcastMul<N: NumericType>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        broadcast("mul", lhs: lhs, rhs: rhs, result: result, host: CPUEngine.broadcastMul)
    }

    public static func broadcastDiv<N: NumericType>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        broadcast("div", lhs: lhs, rhs: rhs, result: result, host: CPUEngine.broadcastDiv)
    }

    // MARK: Reduction

    /// Reduces along the given axes on the GPU, or calls `host`.
    ///
    /// Axes that are neighbors in the shape are reduced as one axis. When the reduced axes do not form one block of
    /// neighboring axes, the values are first permuted so that the reduced axes are the last axes.
    private static func reduce<N: NumericType>(
        _ reduction: GPUKernels.Reduction,
        values: ShapedBuffer<N, GPU>,
        result: MutableShapedBuffer<N, GPU>,
        context: MutableShapedBuffer<Int32, GPU>?,
        axes: [Int],
        isMean: Bool = false,
        host: () -> Void,
    ) {
        let writing = [result.values.memory] + (context.map { [$0.values.memory] } ?? [])
        guard let element = gpuElement(N.self, elements: values.count, reading: [values.values.memory], writing: writing), !(isMean && element == .int) else {
            host()
            return
        }
        guard values.count > 0 else {
            return
        }
        let axes = axes.sorted()
        let length = axes.map { values.shape[$0] }.reduce(1, *)
        let scale: Float = isMean ? 1 / Float(length) : 1
        let isReduced = values.shape.indices.map { axes.contains($0) }

        // Groups of neighboring axes with more than one element that are all reduced or all kept.
        var groups: [(size: Int, isReduced: Bool)] = []
        for axis in values.shape.indices where values.shape[axis] != 1 {
            if let last = groups.last, last.isReduced == isReduced[axis] {
                groups[groups.count - 1].size *= values.shape[axis]
            } else {
                groups.append((values.shape[axis], isReduced[axis]))
            }
        }
        let reducedGroups = groups.indices.filter { groups[$0].isReduced }
        if reducedGroups.count <= 1 {
            let outer = reducedGroups.first.map { groups[..<$0].map(\.size).reduce(1, *) } ?? values.count
            let inner = reducedGroups.first.map { groups[($0 + 1)...].map(\.size).reduce(1, *) } ?? 1
            GPUKernels.reduce(reduction, element, values: values.values.memory, result: result.values.memory, context: context?.values.memory, outer: outer, length: length, inner: inner, scale: scale)
            return
        }
        let kept = values.shape.indices.filter { !isReduced[$0] }
        var arrangement = [Int](repeating: 0, count: values.dim)
        for (destination, source) in (kept + axes).enumerated() {
            arrangement[source] = destination
        }
        let (shape, strides) = StridedIteration.permutationLayout(sourceShape: values.shape, arrangement: arrangement)
        let permuted = GPUKernels.temporary(count: values.count)
        GPUKernels.stridedCopy(source: values.values.memory, result: permuted, layout: GPULayout(shape: shape, strides: [strides, GPULayout.contiguousStrides(shape)]))
        GPUKernels.reduce(reduction, element, values: permuted, result: result.values.memory, context: context?.values.memory, outer: values.count / length, length: length, inner: 1, scale: scale)
    }

    public static func reduceSum<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, axis: Int) {
        reduce(.sum, values: values, result: result, context: nil, axes: [axis]) {
            CPUEngine.reduceSum(values: values.host, result: result.host, axis: axis)
        }
    }

    public static func reduceMax<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, context: MutableShapedBuffer<Int32, GPU>?, axis: Int) {
        reduce(.max, values: values, result: result, context: context, axes: [axis]) {
            CPUEngine.reduceMax(values: values.host, result: result.host, context: context?.host, axis: axis)
        }
    }

    public static func reduceMin<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, context: MutableShapedBuffer<Int32, GPU>?, axis: Int) {
        reduce(.min, values: values, result: result, context: context, axes: [axis]) {
            CPUEngine.reduceMin(values: values.host, result: result.host, context: context?.host, axis: axis)
        }
    }

    public static func reduceMean<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, axis: Int) {
        reduce(.sum, values: values, result: result, context: nil, axes: [axis], isMean: true) {
            CPUEngine.reduceMean(values: values.host, result: result.host, axis: axis)
        }
    }

    public static func reduceSum<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, axes: [Int]) {
        reduce(.sum, values: values, result: result, context: nil, axes: axes) {
            CPUEngine.reduceSum(values: values.host, result: result.host, axes: axes)
        }
    }

    public static func reduceMax<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, context: MutableShapedBuffer<Int32, GPU>?, axes: [Int]) {
        reduce(.max, values: values, result: result, context: context, axes: axes) {
            CPUEngine.reduceMax(values: values.host, result: result.host, context: context?.host, axes: axes)
        }
    }

    public static func reduceMin<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, context: MutableShapedBuffer<Int32, GPU>?, axes: [Int]) {
        reduce(.min, values: values, result: result, context: context, axes: axes) {
            CPUEngine.reduceMin(values: values.host, result: result.host, context: context?.host, axes: axes)
        }
    }

    public static func reduceMean<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, axes: [Int]) {
        reduce(.sum, values: values, result: result, context: nil, axes: axes, isMean: true) {
            CPUEngine.reduceMean(values: values.host, result: result.host, axes: axes)
        }
    }

    public static func sum<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        reduce(.sum, values: values, result: result, context: nil, axes: Array(values.shape.indices)) {
            CPUEngine.sum(values: values.host, result: result.host)
        }
    }

    public static func mean<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        reduce(.sum, values: values, result: result, context: nil, axes: Array(values.shape.indices), isMean: true) {
            CPUEngine.mean(values: values.host, result: result.host)
        }
    }

    // The maximum and minimum of a whole buffer return their argument to the host, so they run on the host.

    @discardableResult
    public static func max<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) -> Int {
        CPUEngine.max(values: values.host, result: result.host)
    }

    @discardableResult
    public static func min<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) -> Int {
        CPUEngine.min(values: values.host, result: result.host)
    }

    public static func argmax<N: NumericType>(values: Buffer<N, GPU>, count: Int) -> (Int, N) {
        CPUEngine.argmax(values: values.host, count: count)
    }

    // MARK: Element-wise functions

    private static func unary<N: NumericType>(_ name: String, supportsIntegers: Bool = false, values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, host: (ShapedBuffer<N, CPU>, MutableShapedBuffer<N, CPU>) -> Void) {
        guard let element = gpuElement(N.self, elements: result.count, reading: [values.values.memory], writing: [result.values.memory]), element == .float || supportsIntegers else {
            host(values.host, result.host)
            return
        }
        GPUKernels.unary(name, element, values: values.values.memory, result: result.values.memory, count: result.count)
    }

    public static func exp<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("exp", values: values, result: result, host: CPUEngine.exp)
    }

    public static func log<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("log", values: values, result: result, host: CPUEngine.log)
    }

    public static func sqrt<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("sqrt", values: values, result: result, host: CPUEngine.sqrt)
    }

    public static func relu<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("relu", supportsIntegers: true, values: values, result: result, host: CPUEngine.relu)
    }

    public static func heaviside<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("heaviside", supportsIntegers: true, values: values, result: result, host: CPUEngine.heaviside)
    }

    public static func sin<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("sin", values: values, result: result, host: CPUEngine.sin)
    }

    public static func cos<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("cos", values: values, result: result, host: CPUEngine.cos)
    }

    public static func tan<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("tan", values: values, result: result, host: CPUEngine.tan)
    }

    public static func sinh<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("sinh", values: values, result: result, host: CPUEngine.sinh)
    }

    public static func cosh<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("cosh", values: values, result: result, host: CPUEngine.cosh)
    }

    public static func tanh<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        unary("tanh", values: values, result: result, host: CPUEngine.tanh)
    }

    public static func max<N: NumericType>(_ lhs: ShapedBuffer<N, GPU>, _ rhs: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        precondition(lhs.shape == rhs.shape, "Shapes of lhs and rhs must match")
        precondition(lhs.shape == result.shape, "Shapes of inputs and result must match")
        vector("max", lhs: lhs.values, rhs: rhs.values, result: result.values, count: result.count) { lhs, rhs, target, _ in
            CPUEngine.max(ShapedBuffer(values: lhs, shape: result.shape), ShapedBuffer(values: rhs, shape: result.shape), result: MutableShapedBuffer(values: target, shape: result.shape))
        }
    }

    public static func max<N: NumericType>(_ lhs: ShapedBuffer<N, GPU>, _ rhs: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, context: MutableShapedBuffer<N, GPU>) {
        select("max", lhs, rhs, result: result, context: context) {
            CPUEngine.max(lhs.host, rhs.host, result: result.host, context: context.host)
        }
    }

    public static func min<N: NumericType>(_ lhs: ShapedBuffer<N, GPU>, _ rhs: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        precondition(lhs.shape == rhs.shape, "Shapes of lhs and rhs must match")
        precondition(lhs.shape == result.shape, "Shapes of inputs and result must match")
        vector("min", lhs: lhs.values, rhs: rhs.values, result: result.values, count: result.count) { lhs, rhs, target, _ in
            CPUEngine.min(ShapedBuffer(values: lhs, shape: result.shape), ShapedBuffer(values: rhs, shape: result.shape), result: MutableShapedBuffer(values: target, shape: result.shape))
        }
    }

    public static func min<N: NumericType>(_ lhs: ShapedBuffer<N, GPU>, _ rhs: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, context: MutableShapedBuffer<N, GPU>) {
        select("min", lhs, rhs, result: result, context: context) {
            CPUEngine.min(lhs.host, rhs.host, result: result.host, context: context.host)
        }
    }

    private static func select<N: NumericType>(_ name: String, _ lhs: ShapedBuffer<N, GPU>, _ rhs: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, context: MutableShapedBuffer<N, GPU>, host: () -> Void) {
        precondition(lhs.shape == rhs.shape, "Shapes of lhs and rhs must match")
        precondition(lhs.shape == result.shape, "Shapes of inputs and result must match")
        precondition(context.shape == result.shape, "Shapes of context and result must match")
        guard let element = gpuElement(N.self, elements: result.count, reading: [lhs.values.memory, rhs.values.memory], writing: [result.values.memory, context.values.memory]) else {
            host()
            return
        }
        GPUKernels.select(name, element, lhs: lhs.values.memory, rhs: rhs.values.memory, result: result.values.memory, context: context.values.memory, count: result.count)
    }

    // MARK: Shuffling

    public static func scatter<N: NumericType>(reduced: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>, result: MutableShapedBuffer<N, GPU>, axis: Int, ignoreIndex: Int32) {
        guard gpuElement(N.self, elements: result.count, reading: [reduced.values.memory, context.values.memory], writing: [result.values.memory]) != nil else {
            CPUEngine.scatter(reduced: reduced.host, context: context.host, result: result.host, axis: axis, ignoreIndex: ignoreIndex)
            return
        }
        let axisSize = result.shape[axis]
        let inner = result.shape[(axis + 1)...].reduce(1, *)
        GPUKernels.fill(result.values.memory, word: 0, count: result.count)
        indexed("scatter_u32", values: reduced.values.memory, indices: context.values.memory, result: result.values.memory, count: context.count, axisSize: axisSize, inner: inner, ignoreIndex: ignoreIndex)
    }

    public static func gather<N: NumericType>(expanded: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>, result: MutableShapedBuffer<N, GPU>, axis: Int, ignoreIndex: Int32) {
        guard gpuElement(N.self, elements: result.count, reading: [expanded.values.memory, context.values.memory], writing: [result.values.memory]) != nil else {
            CPUEngine.gather(expanded: expanded.host, context: context.host, result: result.host, axis: axis, ignoreIndex: ignoreIndex)
            return
        }
        let axisSize = expanded.shape[axis]
        let inner = expanded.shape[(axis + 1)...].reduce(1, *)
        indexed("gather_u32", values: expanded.values.memory, indices: context.values.memory, result: result.values.memory, count: context.count, axisSize: axisSize, inner: inner, ignoreIndex: ignoreIndex)
    }

    private static func indexed(_ name: String, values: GPUBuffer, indices: GPUBuffer, result: GPUBuffer, count: Int, axisSize: Int, inner: Int, ignoreIndex: Int32) {
        let pipeline = GPUKernels.pipeline(name, in: .copy)
        GPUContext.current.compute(pipeline, reading: [values, indices], writing: [result]) { arguments in
            arguments.buffer(values)
            arguments.buffer(indices)
            arguments.buffer(result)
            arguments.value(SIMD4<Int32>(Int32(count), Int32(axisSize), Int32(inner), ignoreIndex))
            arguments.dispatch(count: count)
        }
    }

    public static func permuteAxes<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, arangement: [Int]) {
        guard values.count > 0 else {
            return
        }
        guard gpuElement(N.self, elements: result.count, reading: [values.values.memory], writing: [result.values.memory]) != nil else {
            CPUEngine.permuteAxes(values: values.host, result: result.host, arangement: arangement)
            return
        }
        let (shape, sourceStrides) = StridedIteration.permutationLayout(sourceShape: values.shape, arrangement: arangement)
        let dim = shape.count
        // When the last two axes swap places and the batch is contiguous, a tiled transpose reads and writes contiguous rows.
        if dim >= 2, dim <= 3, sourceStrides[dim - 2] == 1, sourceStrides[dim - 1] == shape[dim - 2], dim == 2 || sourceStrides[0] == shape[1] * shape[2] {
            GPUKernels.transpose(source: values.values.memory, result: result.values.memory, batch: dim == 3 ? shape[0] : 1, rows: shape[dim - 1], columns: shape[dim - 2])
            return
        }
        let layout = GPULayout(shape: shape, strides: [sourceStrides, GPULayout.contiguousStrides(shape)])
        guard layout.isSupported else {
            CPUEngine.permuteAxes(values: values.host, result: result.host, arangement: arangement)
            return
        }
        GPUKernels.stridedCopy(source: values.values.memory, result: result.values.memory, layout: layout)
    }

    public static func permuteAxesAdd<N: NumericType>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, arangement: [Int]) {
        guard values.count > 0 else {
            return
        }
        let (shape, sourceStrides) = StridedIteration.permutationLayout(sourceShape: values.shape, arrangement: arangement)
        let contiguous = GPULayout.contiguousStrides(shape)
        let layout = GPULayout(shape: shape, strides: [sourceStrides, contiguous, contiguous])
        guard let element = gpuElement(N.self, elements: result.count, reading: [values.values.memory, add.values.memory], writing: [result.values.memory]), layout.isSupported else {
            CPUEngine.permuteAxesAdd(values: values.host, add: add.host, result: result.host, arangement: arangement)
            return
        }
        GPUKernels.stridedCopy(source: values.values.memory, result: result.values.memory, summand: (add.values.memory, element), layout: layout)
    }

    public static func arange<N: NumericType>(lowerBound: N, upperBound: N, result: MutableShapedBuffer<N, GPU>) {
        guard N.self == Float.self, !GPUPlacement.runsOnHost(elements: result.count, reading: [], writing: [result.values.memory]) else {
            CPUEngine.arange(lowerBound: lowerBound, upperBound: upperBound, result: result.host)
            return
        }
        // The increment matches the CPU implementation.
        let start = lowerBound.floatValue
        let increment = upperBound.floatValue / Float(result.count)
        let pipeline = GPUKernels.pipeline("arange_float", in: .elementwise)
        let target = result.values.memory
        GPUContext.current.compute(pipeline, reading: [], writing: [target]) { arguments in
            arguments.buffer(target)
            arguments.value(SIMD2<Float>(start, increment))
            arguments.value(UInt32(result.count))
            arguments.dispatch(count: result.count)
        }
    }

    /// Ranges of the elements that an index selects, for every axis of the shape.
    private static func ranges(of index: [Int?], shape: [Int]) -> [Range<Int>] {
        shape.indices.map { axis in
            (axis < index.count ? index[axis] : nil).map { $0 ..< $0 + 1 } ?? 0 ..< shape[axis]
        }
    }

    public static func subscriptRead<N>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, index: [Int?]) {
        GPUKernels.copyRegion(of: values.values, shape: values.shape, ranges: ranges(of: index, shape: values.shape), to: result.values)
    }

    public static func subscriptWrite<N>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, index: [Int?]) {
        GPUKernels.writeRegion(of: result.values, shape: result.shape, ranges: ranges(of: index, shape: result.shape), from: values.values)
    }

    public static func subscriptReadAdd<N: NumericType>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, index: [Int?]) {
        CPUEngine.subscriptReadAdd(values: values.host, add: add.host, result: result.host, index: index)
    }

    public static func subscriptWriteAdd<N: NumericType>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, index: [Int?]) {
        CPUEngine.subscriptWriteAdd(values: values.host, add: add.host, result: result.host, index: index)
    }

    public static func stack<N>(buffers: [ShapedBuffer<N, GPU>], result: MutableShapedBuffer<N, GPU>, axis: Int) {
        var position = 0
        for buffer in buffers {
            let ranges = result.shape.indices.map { $0 == axis ? position ..< position + buffer.shape[axis] : 0 ..< result.shape[$0] }
            GPUKernels.writeRegion(of: result.values, shape: result.shape, ranges: ranges, from: buffer.values)
            position += buffer.shape[axis]
        }
    }

    public static func unstackAdd<N: NumericType>(stacked: ShapedBuffer<N, GPU>, add: [ShapedBuffer<N, GPU>], result: [MutableShapedBuffer<N, GPU>], axis: Int) {
        let stackedStrides = GPULayout.contiguousStrides(stacked.shape)
        var position = 0
        for (target, summand) in zip(result, add) {
            let contiguous = GPULayout.contiguousStrides(target.shape)
            let layout = GPULayout(shape: target.shape, strides: [stackedStrides, contiguous, contiguous])
            let source = stacked.values.advanced(by: position * stackedStrides[axis])
            if let element = gpuElement(N.self, elements: target.count, reading: [source.memory, summand.values.memory], writing: [target.values.memory]), layout.isSupported {
                GPUKernels.stridedCopy(source: source.memory, result: target.values.memory, summand: (summand.values.memory, element), layout: layout)
            } else {
                let (sourcePointer, summandPointer) = (source.host.memory.bindMemory(to: N.self), summand.values.host.memory.bindMemory(to: N.self))
                let targetPointer = target.values.host.memory.bindMemory(to: N.self)
                var index = 0
                StridedIteration.forEachOffset(shape: layout.shape, strides: layout.strides[0], layout.strides[1]) { sourceOffset, targetOffset in
                    targetPointer[targetOffset] = sourcePointer[sourceOffset] + summandPointer[index]
                    index += 1
                }
            }
            position += target.shape[axis]
        }
    }

    public static func unstack<N: NumericType>(stacked: ShapedBuffer<N, GPU>, result: [MutableShapedBuffer<N, GPU>], axis: Int) {
        var position = 0
        for target in result {
            let ranges = stacked.shape.indices.map { $0 == axis ? position ..< position + target.shape[axis] : 0 ..< stacked.shape[$0] }
            GPUKernels.copyRegion(of: stacked.values, shape: stacked.shape, ranges: ranges, to: target.values)
            position += target.shape[axis]
        }
    }

    public static func reverse<N>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        precondition(values.shape == result.shape)
        guard values.count > 0 else {
            return
        }
        let (count, rowLength) = (values.shape[0], values.count / values.shape[0])
        // The source starts at its last row and moves backwards.
        GPUKernels.copy(
            from: values.values.advanced(by: (count - 1) * rowLength),
            strides: [-rowLength, 1],
            to: result.values,
            strides: [rowLength, 1],
            shape: [count, rowLength],
        )
    }

    public static func reverseAdd<N: NumericType>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        precondition(values.shape == result.shape && values.shape == add.shape)
        guard values.count > 0 else {
            return
        }
        let (count, rowLength) = (values.shape[0], values.count / values.shape[0])
        let layout = GPULayout(shape: [count, rowLength], strides: [[-rowLength, 1], [rowLength, 1], [rowLength, 1]])
        let source = values.values.advanced(by: (count - 1) * rowLength)
        guard let element = gpuElement(N.self, elements: result.count, reading: [values.values.memory, add.values.memory], writing: [result.values.memory]) else {
            CPUEngine.reverseAdd(values: values.host, add: add.host, result: result.host)
            return
        }
        GPUKernels.stridedCopy(source: source.memory, result: result.values.memory, summand: (add.values.memory, element), layout: layout)
    }

    // MARK: Convolution helpers

    private struct WindowGeometry {
        var batchSize, channels, height, width: Int32
        var kernelHeight, kernelWidth, padding, stride: Int32
        var outputHeight, outputWidth: Int32

        init(imageShape shape: [Int], kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
            batchSize = Int32(shape[0])
            channels = Int32(shape[1])
            height = Int32(shape[2])
            width = Int32(shape[3])
            self.kernelHeight = Int32(kernelHeight)
            self.kernelWidth = Int32(kernelWidth)
            self.padding = Int32(padding)
            self.stride = Int32(stride)
            outputHeight = Int32((shape[2] + 2 * padding - kernelHeight) / stride + 1)
            outputWidth = Int32((shape[3] + 2 * padding - kernelWidth) / stride + 1)
        }
    }

    public static func img2col<N: NumericType>(values: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) {
        precondition(values.dim == 4, "im2col input must be 4D tensor (batchSize x channels x height x width)")
        guard gpuElement(N.self, elements: result.count, reading: [values.values.memory], writing: [result.values.memory]) != nil else {
            CPUEngine.img2col(values: values.host, result: result.host, kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride)
            return
        }
        let geometry = WindowGeometry(imageShape: values.shape, kernelHeight: kernelHeight, kernelWidth: kernelWidth, padding: padding, stride: stride)
        let rows = values.shape[1] * kernelHeight * kernelWidth
        let columns = Int(geometry.batchSize * geometry.outputHeight * geometry.outputWidth)
        let pipeline = GPUKernels.pipeline("img2col_u32", in: .copy)
        let (image, matrix) = (values.values.memory, result.values.memory)
        GPUContext.current.compute(pipeline, reading: [image], writing: [matrix]) { arguments in
            arguments.buffer(image)
            arguments.buffer(matrix)
            arguments.value(geometry)
            arguments.dispatch(threads: .init(width: columns, height: rows, depth: 1), threadgroup: .init(width: 64, height: 4, depth: 1))
        }
    }

    public static func col2img<N: NumericType>(matrix: ShapedBuffer<N, GPU>, image: MutableShapedBuffer<N, GPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) {
        precondition(image.dim == 4, "im2col input must be 4D tensor (batchSize x channels x height x width)")
        guard N.self == Float.self, !GPUPlacement.runsOnHost(elements: image.count, reading: [matrix.values.memory], writing: [image.values.memory]) else {
            CPUEngine.col2img(matrix: matrix.host, image: image.host, kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride)
            return
        }
        let geometry = WindowGeometry(imageShape: image.shape, kernelHeight: kernelHeight, kernelWidth: kernelWidth, padding: padding, stride: stride)
        let pipeline = GPUKernels.pipeline("col2img_float", in: .copy)
        let (source, target) = (matrix.values.memory, image.values.memory)
        GPUContext.current.compute(pipeline, reading: [source], writing: [target]) { arguments in
            arguments.buffer(source)
            arguments.buffer(target)
            arguments.value(geometry)
            arguments.dispatch(count: image.count)
        }
    }
}
#endif
