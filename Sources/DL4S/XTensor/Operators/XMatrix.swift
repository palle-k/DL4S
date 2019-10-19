//
//  XMatrix.swift
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


public extension XTensor {
    func matrixMultiplied(with other: XTensor<Element, Device>) -> XTensor<Element, Device> {
        let lhs = self
        let rhs = other
        
        precondition(1 ... 2 ~= lhs.dim && 1 ... 2 ~= rhs.dim, "Matrix multiplication operands must both be one or two dimensional.")
        // lhs.dim == 2 and rhs.dim == 2 implies matching shapes
        precondition(!(lhs.dim == 2 && rhs.dim == 2) || lhs.shape[1] == rhs.shape[0], "Matrix multiplication operands must have matching shapes.")
        
        let resultViewShape: [Int]
        
        let lhsView: XTensor<Element, Device>
        let rhsView: XTensor<Element, Device>
        
        switch (lhs.dim, rhs.dim) {
        case (1, 1):
            resultViewShape = []
            lhsView = lhs.view(as: [1, -1])
            rhsView = rhs.view(as: [-1, 1])
        case (1, 2):
            lhsView = lhs.view(as: [1, -1])
            rhsView = rhs
            resultViewShape = [rhs.shape[1]]
        case (2, 1):
            lhsView = lhs
            rhsView = rhs.view(as: [-1, 1])
            resultViewShape = [lhs.shape[0]]
        case (_, _):
            lhsView = lhs
            rhsView = rhs
            resultViewShape = [lhs.shape[0], rhs.shape[1]]
        }
        
        return lhsView._matMul(rhsView).view(as: resultViewShape)
    }
    
    func broadcastMatrixMultiplied(with other: XTensor<Element, Device>, transposeSelf: Bool = false, transposeOther: Bool = false) -> XTensor<Element, Device> {
        precondition(self.dim >= 2 && other.dim >= 2, "Operands must both be at least 2-dimensional.")
        precondition(self.shape.suffix(2)[transposeSelf ? 0 : 1] == other.shape.suffix(2)[transposeOther ? 1 : 0], "Matmul operands must have matching shapes")
        
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
        let matMulResultShape = [lhs.shape.suffix(2)[transposeSelf ? 1 : 0], rhs.shape.suffix(2)[transposeOther ? 0 : 1]]
        
        let resultBuffer = Device.Memory.allocateBuffer(
            withShape: broadcastResultShape + matMulResultShape,
            type: Element.self
        )
        Device.Engine.broadcastGemm(
            lhs: lhs.values,
            rhs: rhs.values,
            result: resultBuffer,
            alpha: 1,
            beta: 0,
            transposeFirst: transposeSelf,
            transposeSecond: transposeOther
        )
        
        return XTensor(
            using: resultBuffer,
            context: (lhs.requiresGradient || rhs.requiresGradient) ? XTensorContext(
                tag: "bmmul",
                sources: [lhs, rhs],
                backpropagate: [
                    { (resultGradient: XTensor<Element, Device>) in
                        let res = resultGradient.broadcastMatrixMultiplied(with: other, transposeSelf: false, transposeOther: !transposeOther)
                        if transposeSelf {
                            return res.permuted(to: Array(0 ..< (lhs.dim - 2)) + [lhs.dim - 1, lhs.dim - 2])
                        } else {
                            return res
                        }
                    }, { resultGradient in
                        let res = self.broadcastMatrixMultiplied(with: resultGradient, transposeSelf: !transposeSelf, transposeOther: false)
                        if transposeOther {
                            return res.permuted(to: Array(0 ..< (rhs.dim - 2)) + [rhs.dim - 1, rhs.dim - 2])
                        } else {
                            return res
                        }
                    }
                ]
            ) : nil
        )
    }
    
    private func _matMul(_ other: XTensor<Element, Device>, transposeSelf: Bool = false, transposeOther: Bool = false) -> XTensor<Element, Device> {
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
        
        return XTensor(
            using: resultBuffer,
            context: (self.requiresGradient || other.requiresGradient) ? XTensorContext(
                tag: "mmul",
                sources: [self, other],
                backpropagate: [
                    { resultGradient in
                        let res = resultGradient._matMul(other, transposeSelf: false, transposeOther: !transposeOther)
                        if transposeSelf {
                            return res.transposed()
                        } else {
                            return res
                        }
                    }, { resultGradient in
                        let res = self._matMul(resultGradient, transposeSelf: !transposeSelf, transposeOther: false)
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
}

public func matMul<Element, Device>(_ lhs: XTensor<Element, Device>, _ rhs: XTensor<Element, Device>) -> XTensor<Element, Device> {
    lhs.matrixMultiplied(with: rhs)
}
