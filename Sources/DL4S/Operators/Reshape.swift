//
//  Reshaping.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.04.19.
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


fileprivate struct ReshapeOperation<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        if !Tensor.sameIdentity(source, vector) {
            Device.Engine.vAdd(lhs: vectorGradient, rhs: sourceGradient, result: sourceGradient, count: source.count)
        }
    }
    
    var symbol: String {
        return "reshape"
    }
}

public extension Tensor {
    func view(as shape: Int...) -> Tensor<Element, Device> {
        return view(as: shape)
    }
    
    func view(as shape: [Int]) -> Tensor<Element, Device> {
        precondition(shape.count(where: {$0 == -1}) <= 1, "The size of at most one dimension can be unknown (-1).")
        precondition(shape.allSatisfy {$0 >= -1}, "All dimensions must be greater than or equal to -1.")
        precondition(shape.contains(-1) || shape.reduce(1, *) == self.count, "Number of elements in result must be equal to number of elements in source")
        
        var shape = shape
        if let idx = shape.firstIndex(of: -1) {
            let remaining = count / shape.lazy.filter {$0 >= 0}.reduce(1, *)
            shape[idx] = remaining
        }
        
        return Tensor(
            values: values,
            gradient: gradient,
            shape: shape,
            parent: self,
            context: requiresGradient ? ReshapeOperation(source: self).asAny() : nil
        )
    }
    
    func viewAsScalar() -> Tensor<Element, Device> {
        precondition(count == 1, "Only vectors with exactly one element can be viewed as a scalar.")
        
        return Tensor(values: values, gradient: gradient, shape: [], parent: self, context: ReshapeOperation(source: self).asAny())
    }
}
