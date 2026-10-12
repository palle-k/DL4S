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
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
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

        return lhsView.broadcastMatrixMultiplied(with: rhsView, transposeSelf: transposeSelf && lhs.dim == 2, transposeOther: transposeOther && rhs.dim == 2).view(as: resultViewShape)
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
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func broadcastMatrixMultiplied(with other: Self, transposeSelf: Bool = false, transposeOther: Bool = false) -> Self {
        precondition(dim >= 2 && other.dim >= 2, "Operands must both be at least 2-dimensional.")
        precondition(Array(shape.suffix(2))[transposeSelf ? 0 : 1] == Array(other.shape.suffix(2))[transposeOther ? 1 : 0], "Matmul operands must have matching shapes")

        var result = Self(uninitializedShape: ShapeUtil.batchedProductShape(shape, other.shape, lhsTransposed: transposeSelf, rhsTransposed: transposeOther))
        BufferMath<Element, Device>().multiplyBatchedMatrices(values, other.values, lhsTransposed: transposeSelf, rhsTransposed: transposeOther, into: result.mutableValues)
        return result.attachingContext(tag: "broadcastMatMul", sources: self, other) { resultGradient, selfGradient, otherGradient in
            // The gradients use the batched product themselves, so they are differentiable for higher derivatives.
            if selfGradient.isRequested {
                let gradient = if transposeSelf {
                    other.broadcastMatrixMultiplied(with: resultGradient, transposeSelf: transposeOther, transposeOther: true)
                } else {
                    resultGradient.broadcastMatrixMultiplied(with: other, transposeOther: !transposeOther)
                }
                selfGradient.add(gradient.reducingBroadcast(to: self.shape))
            }
            if otherGradient.isRequested {
                let gradient = if transposeOther {
                    resultGradient.broadcastMatrixMultiplied(with: self, transposeSelf: true, transposeOther: transposeSelf)
                } else {
                    self.broadcastMatrixMultiplied(with: resultGradient, transposeSelf: !transposeSelf)
                }
                otherGradient.add(gradient.reducingBroadcast(to: other.shape))
            }
        } fused: { resultGradient, selfGradient, otherGradient in
            let math = BufferMath<Element, Device>()
            let (lhs, rhs) = (self.values, other.values)
            if transposeSelf {
                math.writeBatchedProduct(rhs, resultGradient, lhsTransposed: transposeOther, rhsTransposed: true, into: selfGradient)
            } else {
                math.writeBatchedProduct(resultGradient, rhs, rhsTransposed: !transposeOther, into: selfGradient)
            }
            guard let otherGradient else {
                return
            }
            if rhs.dim == 2, !transposeSelf {
                // The right operand is shared by every matrix of the left operand, so its gradient is one product over all rows.
                let rows = lhs.reshaped(to: [lhs.count / lhs.shape[lhs.dim - 1], lhs.shape[lhs.dim - 1]])
                let gradientRows = resultGradient.reshaped(to: [resultGradient.count / resultGradient.shape[resultGradient.dim - 1], resultGradient.shape[resultGradient.dim - 1]])
                if transposeOther {
                    math.writeProduct(gradientRows, rows, lhsTransposed: true, into: otherGradient)
                } else {
                    math.writeProduct(rows, gradientRows, lhsTransposed: true, into: otherGradient)
                }
            } else if transposeOther {
                math.writeBatchedProduct(resultGradient, lhs, lhsTransposed: true, rhsTransposed: transposeSelf, into: otherGradient)
            } else {
                math.writeBatchedProduct(lhs, resultGradient, lhsTransposed: !transposeSelf, into: otherGradient)
            }
        }
    }
}

public extension Tensor {
    /// Multiplies the tensor with the given weights and adds the given bias.
    ///
    /// - Parameters:
    ///   - weights: Weights, shape [inputSize, outputSize]
    ///   - bias: Bias, shape [outputSize], or nil for no bias
    /// - Returns: Tensor of shape [batchSize, outputSize] for a tensor of shape [batchSize, inputSize], or [outputSize] for a vector of shape [inputSize]
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func linearlyTransformed(weights: Self, bias: Self? = nil) -> Self {
        precondition(1 ... 2 ~= dim && weights.dim == 2, "The tensor must be a vector or a matrix, and the weights must be a matrix.")
        precondition(shape[dim - 1] == weights.shape[0], "The tensor must have one element per row of the weights.")
        precondition(bias.map { $0.shape == [weights.shape[1]] } ?? true, "The bias must have one element per column of the weights.")
        let input = dim == 1 ? view(as: [1, -1]) : self
        var result = Self(uninitializedShape: [input.shape[0], weights.shape[1]])
        guard result.count > 0 else {
            return result
        }
        Device.FusedOperations.linear(input: input.values, weights: weights.values, bias: bias?.values, result: result.mutableValues)

        let output = result.attachingContext(tag: "linear", sources: input, weights, bias) { resultGradient, inputGradient, weightGradient, biasGradient in
            Composed.linearBackward(input: input, weights: weights, outputGradient: resultGradient, inputGradient: &inputGradient, weightGradient: &weightGradient, biasGradient: &biasGradient)
        } fused: { resultGradient, inputGradient, weightGradient, biasGradient in
            Device.FusedOperations.linearBackward(input: input.values, weights: weights.values, bias: bias?.values, outputGradient: resultGradient, inputGradient: inputGradient, weightGradient: weightGradient, biasGradient: biasGradient)
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
