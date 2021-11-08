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

//MARK: Matrix Multiplication
public extension Tensor {
    
    /// Computes the matrix-matrix product, the vector-matrix product, the matrix-vector product or the vector-vector product of the tensor with the given other tensor
    /// - Parameter other: Tensor to multiply with self.
    //  - Parameter transposeSelf: Whether to transpose the left hand side matrix before multiplying. Ignored when self.dim == 1.
    //  - Parameter transposeOther: Whether to transpose the right hand side matrix before multiplying. Ignored when other.dim == 1.
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
        precondition(self.dim >= 2 && other.dim >= 2, "Operands must both be at least 2-dimensional.")
        precondition(Array(self.shape.suffix(2))[transposeSelf ? 0 : 1] == Array(other.shape.suffix(2))[transposeOther ? 1 : 0], "Matmul operands must have matching shapes")
        
        // TODO: Fix Gradients
        
        let lhs: Self
        let rhs: Self
        
        if self.dim > other.dim {
            lhs = self
            rhs = other.view(as: Array(repeating: 1, count: self.dim - other.dim) + other.shape)
        } else if self.dim < other.dim {
            lhs = self.view(as: Array(repeating: 1, count: other.dim - self.dim) + self.shape)
            rhs = other
        } else {
            lhs = self
            rhs = other
        }
        
        let broadcastResultShape = shapeForBroadcastedOperands(lhs.shape.dropLast(2), rhs.shape.dropLast(2))
        let matMulResultShape = [Array(lhs.shape.suffix(2))[transposeSelf ? 1 : 0], Array(rhs.shape.suffix(2))[transposeOther ? 0 : 1]]
        
        var results: [Self] = []
        var lhsIdx = Array(repeating: 0, count: broadcastResultShape.count)
        var rhsIdx = Array(repeating: 0, count: broadcastResultShape.count)
        
        for idx in iterate(broadcastResultShape) {
            for i in idx.indices {
                lhsIdx[i] = Swift.min(idx[i], lhs.shape[i] - 1)
                rhsIdx[i] = Swift.min(idx[i], rhs.shape[i] - 1)
            }
            let lhsOp = lhs[lhsIdx]
            let rhsOp = rhs[rhsIdx]
            results.append(lhsOp._matMul(rhsOp, transposeSelf: transposeSelf, transposeOther: transposeOther))
        }
        
        return Tensor(stacking: results).view(as: broadcastResultShape + matMulResultShape)
    }
    
    private func _matMul(_ other: Self, transposeSelf: Bool = false, transposeOther: Bool = false) -> Self {
        precondition(self.dim == 2)
        precondition(other.dim == 2)
        precondition(self.shape[transposeSelf ? 0 : 1] == other.shape[transposeOther ? 1 : 0])
        
        let resultShape = [self.shape[transposeSelf ? 1 : 0], other.shape[transposeOther ? 0 : 1]]
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.gemm(
            lhs: self.values,
            rhs: other.values,
            result: resultBuffer,
            alpha: 1,
            beta: 0,
            transposeFirst: transposeSelf,
            transposeSecond: transposeOther
        )
        
        return Tensor(
            using: resultBuffer,
            context: (self.requiresGradient || other.requiresGradient) ? TensorContext(
                tag: "mmul",
                sources: [self, other],
                backpropagateAccumulate: [
                    { resultGradient, acc in
                        let res: Self
                        if let acc = acc {
                            let acc = transposeSelf ? acc.transposed() : acc
                            res = resultGradient._matMulAdd(other, add: acc, transposeSelf: false, transposeOther: !transposeOther, inplaceAdd: !acc.requiresGradient)
                        } else {
                             res = resultGradient._matMul(other, transposeSelf: false, transposeOther: !transposeOther)
                        }
                        
                        if transposeSelf {
                            return res.transposed()
                        } else {
                            return res
                        }
                    }, { resultGradient, acc in
                        let res: Self
                        
                        if let acc = acc {
                            let acc = transposeOther ? acc.transposed() : acc
                            res = self._matMulAdd(resultGradient, add: acc, transposeSelf: !transposeSelf, transposeOther: false, inplaceAdd: !acc.requiresGradient)
                        } else {
                            res = self._matMul(resultGradient, transposeSelf: !transposeSelf, transposeOther: false)
                        }
                        
                        if transposeOther {
                            return res.transposed()
                        } else {
                            return res
                        }
                    }
                ]
            ) : nil
        )
    }
    
    private func _matMulAdd(_ other: Self, add: Self, transposeSelf: Bool = false, transposeOther: Bool = false, inplaceAdd: Bool = false) -> Self {
        precondition(self.dim == 2)
        precondition(other.dim == 2)
        precondition(self.shape[transposeSelf ? 0 : 1] == other.shape[transposeOther ? 1 : 0])
        
        let resultShape = [self.shape[transposeSelf ? 1 : 0], other.shape[transposeOther ? 0 : 1]]
        precondition(resultShape == add.shape)
        
        var target = add
        if !inplaceAdd {
            target.ensureOwnership()
        }
        
        Device.Engine.gemm(
            lhs: self.values,
            rhs: other.values,
            result: target.values,
            alpha: 1,
            beta: 1,
            transposeFirst: transposeSelf,
            transposeSecond: transposeOther
        )
        
        if self.requiresGradient || other.requiresGradient || add.requiresGradient {
            target.requiresGradient = true
            target.context = TensorContext(
                tag: "gemm",
                sources: [self, other, add],
                backpropagateAccumulate: [
                    { resultGradient, sourceGradient in
                        if let src = sourceGradient {
                            let res = resultGradient._matMulAdd(other, add: transposeSelf ? src.transposed() : src, transposeSelf: false, transposeOther: !transposeOther)
                            if transposeSelf {
                                return res.transposed()
                            } else {
                                return res
                            }
                        } else {
                            let res = resultGradient._matMul(other, transposeSelf: false, transposeOther: !transposeOther)
                            if transposeSelf {
                                return res.transposed()
                            } else {
                                return res
                            }
                        }
                    }, { resultGradient, sourceGradient in
                        if let src = sourceGradient {
                            let res = self._matMulAdd(resultGradient, add: transposeOther ? src.transposed() : src, transposeSelf: !transposeSelf, transposeOther: false)
                            if transposeOther {
                                return res.transposed()
                            } else {
                                return res
                            }
                        } else {
                            let res = self._matMul(resultGradient, transposeSelf: !transposeSelf, transposeOther: false)
                            if transposeOther {
                                return res.transposed()
                            } else {
                                return res
                            }
                        }
                    }, { resultGradient, sourceGradient in
                        sourceGradient.map {$0 + resultGradient} ?? resultGradient
                    }
                ]
            )
        }
        
        return target
    }
}

/// Computes the matrix-matrix product, the vector-matrix product, the matrix-vector product or the vector-vector product of the given two tensors
/// - Parameters:
///   - lhs: left hand side operand
///   - rhs: right hand side operand
public func matMul<Element, Device>(_ lhs: Tensor<Element, Device>, _ rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
    lhs.matrixMultiplied(with: rhs)
}
