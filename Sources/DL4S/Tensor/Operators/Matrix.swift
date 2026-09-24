//
//  Matrix.swift
//  DL4S
//
//  Created by Palle Klewitz on 04.10.19.
//  Copyright (c) 2019 - Palle Klewitz
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

// MARK: Matrix Multiplication

public extension Tensor {
    /// Computes the matrix-matrix product, the vector-matrix product, the matrix-vector product or the vector-vector product of the tensor with the given other tensor
    /// - Parameter other: Tensor to multiply with self.
    /// - Parameter transposeSelf: Whether to transpose the left hand side matrix before multiplying. Ignored when self.dim == 1.
    /// - Parameter transposeOther: Whether to transpose the right hand side matrix before multiplying. Ignored when other.dim == 1.
    func matrixMultiplied(with other: Self, transposeSelf: Bool = false, transposeOther: Bool = false) -> Self {
        let lhs = self
        let rhs = other

        precondition(1 ... 2 ~= lhs.dim && 1 ... 2 ~= rhs.dim, "Matrix multiplication operands must both be one or two dimensional.")
        // lhs.dim == 2 and rhs.dim == 2 implies matching shapes
        precondition(!(lhs.dim == 2 && rhs.dim == 2) || lhs.shape[transposeSelf ? 0 : 1] == rhs.shape[transposeOther ? 1 : 0], "Matrix multiplication operands must have matching shapes.")

        let resultViewShape: [Int]

        let lhsView: Self
        let rhsView: Self

        switch (lhs.dim, rhs.dim) {
        case (1, 1):
            resultViewShape = []
            lhsView = lhs.view(as: [1, -1])
            rhsView = rhs.view(as: [-1, 1])
        case (1, 2):
            lhsView = lhs.view(as: [1, -1])
            rhsView = rhs
            resultViewShape = [rhs.shape[transposeOther ? 0 : 1]]
        case (2, 1):
            lhsView = lhs
            rhsView = rhs.view(as: [-1, 1])
            resultViewShape = [lhs.shape[transposeSelf ? 1 : 0]]
        case (_, _):
            lhsView = lhs
            rhsView = rhs
            resultViewShape = [lhs.shape[transposeSelf ? 1 : 0], rhs.shape[transposeOther ? 0 : 1]]
        }

        return lhsView._matMul(rhsView, transposeSelf: transposeSelf && lhs.dim == 2, transposeOther: transposeOther && rhs.dim == 2).view(as: resultViewShape)
    }

    /// Broadcast matrix multiplies self with the given other operand.
    ///
    /// Broadcasting is applied along all axes except the last two.
    /// Operands are expected to have a dimensionality of 2 or higher.
    ///
    /// - Parameters:
    ///   - other: Other operand
    ///   - transposeSelf: Whether to transpose self before multiplication
    ///   - transposeOther: Whether to transpose the other operand before the multiplication
    func broadcastMatrixMultiplied(with other: Self, transposeSelf: Bool = false, transposeOther: Bool = false) -> Self {
        precondition(dim >= 2 && other.dim >= 2, "Operands must both be at least 2-dimensional.")
        precondition(Array(shape.suffix(2))[transposeSelf ? 0 : 1] == Array(other.shape.suffix(2))[transposeOther ? 1 : 0], "Matmul operands must have matching shapes")

        let result = Self.batchedMatrixProduct(detached(), other.detached(), transposeLhs: transposeSelf, transposeRhs: transposeOther)
        return result.attachingContext(tag: "broadcastMatMul", sources: [self, other]) { resultGradient in
            // The gradients use the batched product themselves, so they are differentiable for higher derivatives.
            var lhsGradient: Self?
            if self.requiresGradient {
                let gradient = if transposeSelf {
                    other.broadcastMatrixMultiplied(with: resultGradient, transposeSelf: transposeOther, transposeOther: true)
                } else {
                    resultGradient.broadcastMatrixMultiplied(with: other, transposeOther: !transposeOther)
                }
                lhsGradient = gradient.reducingBroadcast(to: self.shape)
            }
            var rhsGradient: Self?
            if other.requiresGradient {
                if other.dim == 2, !transposeSelf {
                    // The right operand is shared by every matrix of the left operand, so its gradient is one product over all rows.
                    let rows = self.view(as: [-1, self.shape[self.dim - 1]])
                    let gradient = resultGradient.view(as: [-1, resultGradient.shape[resultGradient.dim - 1]])
                    rhsGradient = transposeOther
                        ? gradient.matrixMultiplied(with: rows, transposeSelf: true)
                        : rows.matrixMultiplied(with: gradient, transposeSelf: true)
                } else {
                    let gradient = if transposeOther {
                        resultGradient.broadcastMatrixMultiplied(with: self, transposeSelf: true, transposeOther: transposeSelf)
                    } else {
                        self.broadcastMatrixMultiplied(with: resultGradient, transposeSelf: !transposeSelf)
                    }
                    rhsGradient = gradient.reducingBroadcast(to: other.shape)
                }
            }
            return [lhsGradient, rhsGradient]
        }
    }

    /// Multiplies the matrices of two tensors without context, with broadcasting along all axes except the last two.
    ///
    /// Every product is one GEMM into its slice of the result. When the right operand is a single matrix and the left one
    /// is not transposed, the rows of all left matrices form one matrix, and the product is a single GEMM.
    private static func batchedMatrixProduct(_ lhs: Self, _ rhs: Self, transposeLhs: Bool, transposeRhs: Bool) -> Self {
        let dim = Swift.max(lhs.dim, rhs.dim)
        let lhsShape = Array(repeating: 1, count: dim - lhs.dim) + lhs.shape
        let rhsShape = Array(repeating: 1, count: dim - rhs.dim) + rhs.shape
        let (lhsMatrix, rhsMatrix) = (Array(lhsShape.suffix(2)), Array(rhsShape.suffix(2)))
        let (lhsBatch, rhsBatch) = (Array(lhsShape.dropLast(2)), Array(rhsShape.dropLast(2)))
        let batchShape = shapeForBroadcastedOperands(lhsBatch, rhsBatch)
        let rows = transposeLhs ? lhsMatrix[1] : lhsMatrix[0]
        let columns = transposeRhs ? rhsMatrix[0] : rhsMatrix[1]
        let result = Device.Memory.allocateBuffer(withShape: batchShape + [rows, columns], type: Element.self)

        if rhsBatch.allSatisfy({ $0 == 1 }), !transposeLhs {
            let batchCount = batchShape.reduce(1, *)
            Device.Engine.gemm(
                lhs: lhs.values.reshaped(to: [batchCount * rows, lhsMatrix[1]]),
                rhs: rhs.values.reshaped(to: rhsMatrix),
                result: result.slice(offset: 0, shape: [batchCount * rows, columns]),
                alpha: 1,
                beta: 0,
                transposeFirst: false,
                transposeSecond: transposeRhs,
            )
            return Tensor(using: result, context: nil)
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
            Device.Engine.gemm(
                lhs: lhs.values.slice(offset: lhsOffset, shape: lhsMatrix),
                rhs: rhs.values.slice(offset: rhsOffset, shape: rhsMatrix),
                result: result.slice(offset: resultOffset, shape: [rows, columns]),
                alpha: 1,
                beta: 0,
                transposeFirst: transposeLhs,
                transposeSecond: transposeRhs,
            )
            resultOffset += rows * columns
        }
        return Tensor(using: result, context: nil)
    }

    private func _matMul(_ other: Self, transposeSelf: Bool = false, transposeOther: Bool = false) -> Self {
        precondition(dim == 2)
        precondition(other.dim == 2)
        precondition(shape[transposeSelf ? 0 : 1] == other.shape[transposeOther ? 1 : 0])

        let resultShape = [shape[transposeSelf ? 1 : 0], other.shape[transposeOther ? 0 : 1]]

        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.gemm(
            lhs: values,
            rhs: other.values,
            result: resultBuffer,
            alpha: 1,
            beta: 0,
            transposeFirst: transposeSelf,
            transposeSecond: transposeOther,
        )

        return Tensor(
            using: resultBuffer,
            context: (requiresGradient || other.requiresGradient) ? TensorContext(
                tag: "mmul",
                sources: [self, other],
                backpropagateAccumulate: Self.matMulBackwards(lhs: self, rhs: other, transposeLhs: transposeSelf, transposeRhs: transposeOther),
            ) : nil,
        )
    }

    /// Returns the matrix product of `self` and `other` added to `add`. When `add` is nil, returns the product.
    private func _matMulAdd(_ other: Self, add: consuming Self?, transposeSelf: Bool = false, transposeOther: Bool = false) -> Self {
        switch consume add {
        case .none:
            return _matMul(other, transposeSelf: transposeSelf, transposeOther: transposeOther)
        case var .some(target):
            precondition(self.dim == 2)
            precondition(other.dim == 2)
            precondition(self.shape[transposeSelf ? 0 : 1] == other.shape[transposeOther ? 1 : 0])
            precondition(target.shape == [self.shape[transposeSelf ? 1 : 0], other.shape[transposeOther ? 0 : 1]])

            // When we're capturing a graph, we need target pre-addition. Without a graph, we can work in-place.
            let original = target.requiresGradient ? target : nil

            Device.Engine.gemm(
                lhs: self.values,
                rhs: other.values,
                result: target.mutableValues,
                alpha: 1,
                beta: 1,
                transposeFirst: transposeSelf,
                transposeSecond: transposeOther,
            )

            guard self.requiresGradient || other.requiresGradient || original != nil else {
                return target
            }

            var sources = [self, other]
            var backpropagate = Self.matMulBackwards(lhs: self, rhs: other, transposeLhs: transposeSelf, transposeRhs: transposeOther)
            if let original {
                sources.append(original)
                backpropagate.append { resultGradient, acc in
                    acc.map { $0 + resultGradient } ?? resultGradient
                }
            }
            target.context = TensorContext(tag: "gemm", sources: sources, backpropagateAccumulate: backpropagate)
            target.requiresGradient = true
            return target
        }
    }

    /// Backpropagation closures for the product of `lhs` and `rhs`, one per operand.
    private static func matMulBackwards(lhs: Self, rhs: Self, transposeLhs: Bool, transposeRhs: Bool) -> [@Sendable (Self, consuming Self?) -> Self] {
        [
            { resultGradient, acc in
                if transposeLhs {
                    rhs._matMulAdd(resultGradient, add: acc, transposeSelf: transposeRhs, transposeOther: true)
                } else {
                    resultGradient._matMulAdd(rhs, add: acc, transposeSelf: false, transposeOther: !transposeRhs)
                }
            },
            { resultGradient, acc in
                if transposeRhs {
                    resultGradient._matMulAdd(lhs, add: acc, transposeSelf: true, transposeOther: transposeLhs)
                } else {
                    lhs._matMulAdd(resultGradient, add: acc, transposeSelf: !transposeLhs, transposeOther: false)
                }
            },
        ]
    }
}

public extension Tensor {
    /// Multiplies the tensor with the given weights and adds the given bias.
    ///
    /// - Parameters:
    ///   - weights: Weights, shape [inputSize, outputSize]
    ///   - bias: Bias, shape [outputSize], or nil for no bias
    /// - Returns: Tensor of shape [batchSize, outputSize] for a tensor of shape [batchSize, inputSize], or [outputSize] for a vector of shape [inputSize]
    func linearlyTransformed(weights: Self, bias: Self? = nil) -> Self {
        precondition(1 ... 2 ~= dim && weights.dim == 2, "The tensor must be a vector or a matrix, and the weights must be a matrix.")
        precondition(shape[dim - 1] == weights.shape[0], "The tensor must have one element per row of the weights.")
        precondition(bias.map { $0.shape == [weights.shape[1]] } ?? true, "The bias must have one element per column of the weights.")
        let input = dim == 1 ? view(as: [1, -1]) : self
        let result = Device.FusedOperations.linear(input: input, weights: weights, bias: bias)

        let output = result.attachingContext(tag: "linear", sources: [input, weights] + (bias.map { [$0] } ?? [])) { resultGradient in
            let gradients = if resultGradient.requiresGradient {
                Composed.linearGradients(
                    input: input,
                    weights: weights,
                    outputGradient: resultGradient,
                    computesInput: input.requiresGradient,
                    computesWeights: weights.requiresGradient,
                    computesBias: bias?.requiresGradient ?? false,
                )
            } else {
                Device.FusedOperations.linearBackward(input: input, weights: weights, bias: bias, outputGradient: resultGradient)
            }
            return [gradients.input, gradients.weights] + (bias == nil ? [] : [gradients.bias])
        }
        return dim == 1 ? output.view(as: [weights.shape[1]]) : output
    }
}

/// Computes the matrix-matrix product, the vector-matrix product, the matrix-vector product or the vector-vector product of the given two tensors
/// - Parameters:
///   - lhs: left hand side operand
///   - rhs: right hand side operand
public func matMul<Element, Device>(_ lhs: Tensor<Element, Device>, _ rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
    lhs.matrixMultiplied(with: rhs)
}
