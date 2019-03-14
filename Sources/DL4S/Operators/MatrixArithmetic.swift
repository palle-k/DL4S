//
//  MatArithmetic.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
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


private struct MatmulOperation<Element: NumericType, Device: DeviceType>: BinaryTensorOperation {
    var lhs: Tensor<Element, Device>
    var rhs: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        
        if let lhsGradient = lhs.gradient {
            let temp2: Buffer<Element, Device> = Device.Memory.allocateBuffer(withCapacity: rhs.count, type: Element.self)
            let temp3: Buffer<Element, Device> = Device.Memory.allocateBuffer(withCapacity: lhs.count, type: Element.self)
            
            Device.Engine.transpose(val: rhs.values, result: temp2, srcRows: rhs.shape[0], srcCols: rhs.shape[1])
            Device.Engine.matMul(lhs: vectorGradient, rhs: temp2, result: temp3, lhsRows: vector.shape[0], lhsCols: vector.shape[1], rhsCols: rhs.shape[0])
            Device.Engine.vAdd(lhs: temp3, rhs: lhsGradient, result: lhsGradient, count: lhs.count)
            
            Device.Memory.free(temp2)
            Device.Memory.free(temp3)
        }
        if let rhsGradient = rhs.gradient {
            let temp1: Buffer<Element, Device> = Device.Memory.allocateBuffer(withCapacity: lhs.count, type: Element.self)
            let temp4: Buffer<Element, Device> = Device.Memory.allocateBuffer(withCapacity: rhs.count, type: Element.self)
            
            Device.Engine.transpose(val: lhs.values, result: temp1, srcRows: lhs.shape[0], srcCols: lhs.shape[1])
            Device.Engine.matMul(lhs: temp1, rhs: vectorGradient, result: temp4, lhsRows: lhs.shape[1], lhsCols: lhs.shape[0], rhsCols: vector.shape[1])
            Device.Engine.vAdd(lhs: temp4, rhs: rhsGradient, result: rhsGradient, count: rhs.count)
            
            Device.Memory.free(temp1)
            Device.Memory.free(temp4)
        }
    }
    
    var symbol: String {
        return "matmul"
    }
}


public func mmul<Element: NumericType, Device: DeviceType>(_ lhs: Tensor<Element, Device>, _ rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
    precondition(1 ... 2 ~= lhs.dim && 1 ... 2 ~= rhs.dim, "Matrix multiplication operands must both be one or two dimensional.")
    // lhs.dim == 2 and rhs.dim == 2 implies matching shapes
    precondition(!(lhs.dim == 2 && rhs.dim == 2) || lhs.shape[1] == rhs.shape[0], "Matrix multiplication operands must have matching shapes.")
    
    let resultShape: [Int]
    let resultViewShape: [Int]
    
    let lhsView: Tensor<Element, Device>
    let rhsView: Tensor<Element, Device>
    
    switch (lhs.dim, rhs.dim) {
    case (1, 1):
        resultShape = [1, 1]
        resultViewShape = []
        lhsView = lhs.view(as: 1, -1)
        rhsView = rhs.view(as: -1, 1)
    case (1, 2):
        lhsView = lhs.view(as: 1, -1)
        rhsView = rhs
        resultShape = [1, rhs.shape[1]]
        resultViewShape = [rhs.shape[1]]
    case (2, 1):
        lhsView = lhs
        rhsView = rhs.view(as: -1, 1)
        resultShape = [lhs.shape[0], 1]
        resultViewShape = [lhs.shape[0]]
    case (_, _):
        lhsView = lhs
        rhsView = rhs
        resultShape = [lhs.shape[0], rhs.shape[1]]
        resultViewShape = [lhs.shape[0], rhs.shape[1]]
    }
    
    let result = Tensor<Element, Device>(
        shape: resultShape,
        parent: nil,
        context: lhs.requiresGradient || rhs.requiresGradient ? MatmulOperation(lhs: lhsView, rhs: rhsView).asAny() : nil
    )
    
    Device.Engine.matMul(lhs: lhsView.values, rhs: rhsView.values, result: result.values, lhsRows: lhsView.shape[0], lhsCols: lhsView.shape[1], rhsCols: rhsView.shape[1])
    
    return result.view(as: resultViewShape)
}

struct TransposeOperation<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.gradient, let sourceGradient = source.gradient else {
            return
        }
        let temp: Buffer<Element, Device> = Device.Memory.allocateBuffer(withCapacity: vector.count, type: Element.self)
        Device.Engine.transpose(val: vectorGradient, result: temp, srcRows: vector.shape[0], srcCols: vector.shape[1])
        Device.Engine.vAdd(lhs: sourceGradient, rhs: temp, result: sourceGradient, count: vector.count)
        Device.Memory.free(temp)
    }
    
    var symbol: String {
        return "transpose"
    }
}

public extension Tensor {
    var T: Tensor<Element, Device> {
        precondition(dim <= 2, "Dimensionality for vector transpose must be smaller than or equal to 2")
        
        if dim <= 1 {
            return self
        } else {
            let result = Tensor<Element, Device>(
                shape: [self.shape[1], self.shape[0]],
                parent: nil,
                context: requiresGradient ? TransposeOperation(source: self).asAny() : nil
            )
            
            Device.Engine.transpose(val: values, result: result.values, srcRows: shape[0], srcCols: shape[1])
            
            return result
        }
    }
}
