//
//  BufferMath.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.09.26.
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

import Foundation

/// A buffer that a backward requirement of ``FusedOperationsType`` writes the gradient of one source into.
public struct GradientBuffer<Element: NumericType, Device: DeviceType> {
    /// Elements of the gradient, with the shape of the source
    public let values: MutableShapedBuffer<Element, Device>

    /// Whether the elements hold an accumulated gradient, to which the gradient is added.
    /// Otherwise, the elements are not initialized, and the gradient overwrites them.
    public let adds: Bool

    /// Creates a buffer for the gradient of a source.
    /// - Parameters:
    ///   - values: Elements of the gradient, with the shape of the source
    ///   - adds: Whether the gradient is added to the elements
    public init(values: MutableShapedBuffer<Element, Device>, adds: Bool) {
        self.values = values
        self.adds = adds
    }

    /// Factor of the current elements in a write, `values = gradient + beta * values`: 1 when the gradient is added, and 0 otherwise.
    public var beta: Element {
        adds ? 1 : 0
    }

    var shape: [Int] {
        values.shape
    }
}

/// A buffer whose elements the operations of ``BufferMath`` can read.
protocol ReadableBuffer<Element, Device> {
    associatedtype Element
    associatedtype Device: DeviceType

    var readable: ShapedBuffer<Element, Device> { get }
}

extension ShapedBuffer: ReadableBuffer {
    var readable: ShapedBuffer<Element, Device> {
        self
    }
}

extension MutableShapedBuffer: ReadableBuffer {
    var readable: ShapedBuffer<Element, Device> {
        ShapedBuffer(self)
    }
}

/// Arithmetic on the buffers of a device, for the default implementations of fused operations.
///
/// Every operation writes into a given result, which can be one of the operands when both have the same shape, so an
/// implementation can compute in the buffers of its results and in few intermediate buffers.
/// Intermediate buffers from ``temporary(_:)`` and ``constant(_:)`` stay allocated until ``release()``.
struct BufferMath<N: NumericType, Device: DeviceType> {
    typealias Engine = Device.Engine
    typealias Writable = MutableShapedBuffer<N, Device>

    private final class Temporaries {
        var buffers: [MutableShapedBuffer<N, Device>] = []
        var positions: [MutableShapedBuffer<Int32, Device>] = []
    }

    private let temporaries = Temporaries()

    /// Releases every intermediate buffer.
    func release() {
        temporaries.buffers.forEach(Device.Memory.free)
        temporaries.positions.forEach(Device.Memory.free)
        temporaries.buffers = []
        temporaries.positions = []
    }

    /// Returns an intermediate buffer with elements that are not initialized.
    func temporary(_ shape: [Int]) -> Writable {
        let buffer = Device.Memory.allocateBuffer(withShape: shape, type: N.self)
        temporaries.buffers.append(buffer)
        return buffer
    }

    /// Returns an intermediate buffer for positions, with elements that are not initialized.
    func positions(_ shape: [Int]) -> MutableShapedBuffer<Int32, Device> {
        let buffer = Device.Memory.allocateBuffer(withShape: shape, type: Int32.self)
        temporaries.positions.append(buffer)
        return buffer
    }

    /// Returns an intermediate buffer of the given shape, whose elements have the given value.
    func constant(_ value: N, shape: [Int] = []) -> ShapedBuffer<N, Device> {
        let buffer = temporary(shape)
        Engine.fill(value: value, result: buffer.values, count: buffer.count)
        return ShapedBuffer(buffer)
    }

    // MARK: Element-wise operations with broadcasting

    func add(_ lhs: some ReadableBuffer<N, Device>, _ rhs: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.broadcastAdd(lhs: lhs.readable, rhs: rhs.readable, result: result)
    }

    func subtract(_ lhs: some ReadableBuffer<N, Device>, _ rhs: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.broadcastSub(lhs: lhs.readable, rhs: rhs.readable, result: result)
    }

    func multiply(_ lhs: some ReadableBuffer<N, Device>, _ rhs: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.broadcastMul(lhs: lhs.readable, rhs: rhs.readable, result: result)
    }

    func divide(_ lhs: some ReadableBuffer<N, Device>, _ rhs: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.broadcastDiv(lhs: lhs.readable, rhs: rhs.readable, result: result)
    }

    func add(_ lhs: some ReadableBuffer<N, Device>, _ rhs: N, into result: Writable) {
        add(lhs, constant(rhs), into: result)
    }

    func multiply(_ lhs: some ReadableBuffer<N, Device>, _ rhs: N, into result: Writable) {
        multiply(lhs, constant(rhs), into: result)
    }

    /// Computes `lhs - values`.
    func subtract(_ lhs: N, _ values: some ReadableBuffer<N, Device>, into result: Writable) {
        subtract(constant(lhs), values, into: result)
    }

    func maximum(_ lhs: some ReadableBuffer<N, Device>, _ rhs: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.max(lhs.readable, rhs.readable, result: result)
    }

    // MARK: Element-wise functions

    func negate(_ values: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.vNeg(val: values.readable.values, result: result.values, count: result.count)
    }

    func exp(_ values: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.exp(values: values.readable, result: result)
    }

    func log(_ values: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.log(values: values.readable, result: result)
    }

    func tanh(_ values: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.tanh(values: values.readable, result: result)
    }

    func sqrt(_ values: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.sqrt(values: values.readable, result: result)
    }

    func relu(_ values: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.relu(values: values.readable, result: result)
    }

    /// Writes 1 for every positive element and 0 for the others.
    func heaviside(_ values: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.heaviside(values: values.readable, result: result)
    }

    func sine(_ values: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.sin(values: values.readable, result: result)
    }

    func cosine(_ values: some ReadableBuffer<N, Device>, into result: Writable) {
        Engine.cos(values: values.readable, result: result)
    }

    /// Computes `sigmoid(scale * values)` as `tanh(scale * values / 2) / 2 + 1 / 2`, which does not overflow for large magnitudes.
    func sigmoid(_ values: some ReadableBuffer<N, Device>, scale: N = 1, into result: Writable) {
        let half = N(0.5)
        multiply(values, scale * half, into: result)
        tanh(result, into: result)
        multiply(result, half, into: result)
        add(result, half, into: result)
    }

    func copy(_ values: some ReadableBuffer<N, Device>, into result: Writable) {
        Device.Memory.assign(from: values.readable.values, to: result.values, count: result.count)
    }

    // MARK: Reductions and matrices

    /// Sums along the axes into a result with the shape of the values without the axes, or with the same number of elements.
    func sum(_ values: some ReadableBuffer<N, Device>, along axes: [Int], into result: Writable) {
        let values = values.readable
        Engine.reduceSum(values: values, result: result.reshaped(to: ShapeUtil.reducedShape(of: values.shape, along: axes)), axes: axes)
    }

    /// Computes the mean along the axes into a result with the shape of the values without the axes, or with the same number of elements.
    func mean(_ values: some ReadableBuffer<N, Device>, along axes: [Int], into result: Writable) {
        let values = values.readable
        Engine.reduceMean(values: values, result: result.reshaped(to: ShapeUtil.reducedShape(of: values.shape, along: axes)), axes: axes)
    }

    /// Writes the largest value along the axis, and optionally its position, into results with the shape of the values without the axis.
    func maximum(_ values: some ReadableBuffer<N, Device>, along axis: Int, into result: Writable, positions: MutableShapedBuffer<Int32, Device>? = nil) {
        let values = values.readable
        let shape = ShapeUtil.reducedShape(of: values.shape, along: [axis])
        Engine.reduceMax(values: values, result: result.reshaped(to: shape), context: positions?.reshaped(to: shape), axis: axis)
    }

    /// Computes `result = alpha * op(lhs) × op(rhs) + beta * result` for matrices.
    func multiplyMatrices(
        _ lhs: some ReadableBuffer<N, Device>,
        _ rhs: some ReadableBuffer<N, Device>,
        lhsTransposed transposeLhs: Bool = false,
        rhsTransposed transposeRhs: Bool = false,
        into result: Writable,
        alpha: N = 1,
        beta: N = 0,
    ) {
        Engine.gemm(lhs: lhs.readable, rhs: rhs.readable, result: result, alpha: alpha, beta: beta, transposeFirst: transposeLhs, transposeSecond: transposeRhs)
    }

    /// Computes `result = alpha * op(lhs) × op(rhs) + beta * result` for the matrices along the last two axes, with broadcasting
    /// along the other axes. The result has the broadcast shape of the other axes, followed by the rows and the columns.
    func multiplyBatchedMatrices(
        _ lhs: some ReadableBuffer<N, Device>,
        _ rhs: some ReadableBuffer<N, Device>,
        lhsTransposed transposeLhs: Bool = false,
        rhsTransposed transposeRhs: Bool = false,
        into result: Writable,
        alpha: N = 1,
        beta: N = 0,
    ) {
        let (lhs, rhs) = (lhs.readable, rhs.readable)
        let dim = Swift.max(lhs.dim, rhs.dim)
        let lhsShape = Array(repeating: 1, count: dim - lhs.dim) + lhs.shape
        let rhsShape = Array(repeating: 1, count: dim - rhs.dim) + rhs.shape
        let (lhsMatrix, rhsMatrix) = (Array(lhsShape.suffix(2)), Array(rhsShape.suffix(2)))
        let (lhsBatch, rhsBatch) = (Array(lhsShape.dropLast(2)), Array(rhsShape.dropLast(2)))
        let batchShape = shapeForBroadcastedOperands(lhsBatch, rhsBatch)
        let rows = transposeLhs ? lhsMatrix[1] : lhsMatrix[0]
        let columns = transposeRhs ? rhsMatrix[0] : rhsMatrix[1]
        precondition(result.count == batchShape.reduce(1, *) * rows * columns, "The result must have the shape of the product.")

        // When the right operand is one matrix and the left one is not transposed, the rows of all left matrices form one matrix.
        if rhsBatch.allSatisfy({ $0 == 1 }), !transposeLhs {
            let batchCount = batchShape.reduce(1, *)
            multiplyMatrices(
                lhs.reshaped(to: [batchCount * rows, lhsMatrix[1]]),
                rhs.reshaped(to: rhsMatrix),
                rhsTransposed: transposeRhs,
                into: result.reshaped(to: [batchCount * rows, columns]),
                alpha: alpha,
                beta: beta,
            )
            return
        }

        // A broadcast axis has the stride 0, so every product of the batch reads the same matrix.
        func batchStrides(_ batch: [Int], matrixSize: Int) -> [Int] {
            var strides = [Int](repeating: 0, count: batch.count)
            var stride = matrixSize
            for axis in batch.indices.reversed() {
                strides[axis] = batch[axis] == 1 ? 0 : stride
                stride *= batch[axis]
            }
            return strides
        }
        let lhsStrides = batchStrides(lhsBatch, matrixSize: lhsMatrix[0] * lhsMatrix[1])
        let rhsStrides = batchStrides(rhsBatch, matrixSize: rhsMatrix[0] * rhsMatrix[1])
        var resultOffset = 0
        StridedIteration.forEachOffset(shape: batchShape, strides: lhsStrides, rhsStrides) { lhsOffset, rhsOffset in
            multiplyMatrices(
                lhs.slice(offset: lhsOffset, shape: lhsMatrix),
                rhs.slice(offset: rhsOffset, shape: rhsMatrix),
                lhsTransposed: transposeLhs,
                rhsTransposed: transposeRhs,
                into: result.slice(offset: resultOffset, shape: [rows, columns]),
                alpha: alpha,
                beta: beta,
            )
            resultOffset += rows * columns
        }
    }

    /// Computes the softmax along an axis. The values and the result can be the same memory.
    func softmax(_ values: some ReadableBuffer<N, Device>, along axis: Int, into result: Writable) {
        let values = values.readable
        let reduced = temporary(ShapeUtil.keptShape(of: values.shape, along: [axis]))
        maximum(values, along: axis, into: reduced)
        subtract(values, reduced, into: result)
        exp(result, into: result)
        sum(result, along: [axis], into: reduced)
        divide(result, reduced, into: result)
    }

    /// Computes the gradient of the softmax along an axis, `output * (outputGradient - sum(outputGradient * output))`.
    /// The gradient of the output and the result can be the same memory.
    func softmaxGradient(output: some ReadableBuffer<N, Device>, outputGradient: some ReadableBuffer<N, Device>, along axis: Int, into result: Writable) {
        let (output, outputGradient) = (output.readable, outputGradient.readable)
        let products = temporary(output.shape)
        let sums = temporary(ShapeUtil.keptShape(of: output.shape, along: [axis]))
        multiply(outputGradient, output, into: products)
        sum(products, along: [axis], into: sums)
        subtract(outputGradient, sums, into: result)
        multiply(result, output, into: result)
    }

    func permute(_ values: some ReadableBuffer<N, Device>, to arrangement: [Int], into result: Writable) {
        Engine.permuteAxes(values: values.readable, result: result, arangement: arrangement)
    }

    // MARK: Gradients

    /// Writes a gradient: `body` computes it into the gradient buffer when the gradient overwrites it, and otherwise into an
    /// intermediate buffer, which is then added to the gradient buffer. Nothing happens for a gradient that is not requested.
    func write(_ gradient: GradientBuffer<N, Device>?, _ body: (Writable) -> Void) {
        guard let gradient else {
            return
        }
        guard gradient.adds else {
            body(gradient.values)
            return
        }
        let values = temporary(gradient.shape)
        body(values)
        add(gradient.values, values, into: gradient.values)
    }

    /// Writes the sum of the values along the axes that broadcast the shape of the gradient to the shape of the values.
    func writeSum(of values: some ReadableBuffer<N, Device>, into gradient: GradientBuffer<N, Device>?) {
        guard let gradient else {
            return
        }
        let values = values.readable
        let axes = ShapeUtil.broadcastAxes(from: gradient.shape, to: values.shape)
        write(gradient) { result in
            if axes.isEmpty {
                copy(values, into: result)
            } else {
                sum(values, along: axes, into: result)
            }
        }
    }

    /// Writes the batched matrix product `alpha * op(lhs) × op(rhs)` into a gradient. When the product has more batch axes or
    /// larger ones than the gradient, it is summed along the axes that broadcasting expanded.
    func writeBatchedProduct(
        _ lhs: some ReadableBuffer<N, Device>,
        _ rhs: some ReadableBuffer<N, Device>,
        lhsTransposed transposeLhs: Bool = false,
        rhsTransposed transposeRhs: Bool = false,
        alpha: N = 1,
        into gradient: GradientBuffer<N, Device>?,
    ) {
        guard let gradient else {
            return
        }
        let (lhs, rhs) = (lhs.readable, rhs.readable)
        let productShape = ShapeUtil.batchedProductShape(lhs.shape, rhs.shape, lhsTransposed: transposeLhs, rhsTransposed: transposeRhs)
        if productShape == gradient.shape {
            multiplyBatchedMatrices(lhs, rhs, lhsTransposed: transposeLhs, rhsTransposed: transposeRhs, into: gradient.values, alpha: alpha, beta: gradient.beta)
            return
        }
        let product = temporary(productShape)
        multiplyBatchedMatrices(lhs, rhs, lhsTransposed: transposeLhs, rhsTransposed: transposeRhs, into: product, alpha: alpha)
        writeSum(of: product, into: gradient)
    }
}
