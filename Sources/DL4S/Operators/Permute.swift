//
//  Permutation.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
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


fileprivate struct PermutationOperation<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var symbol: String {
        return "permute"
    }
    
    var source: Tensor<Element, Device>
    var axisArangement: [Int]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let srcGrad = source.shapedGradient, let dstGrad = vector.shapedGradient else {
            return
        }
        
        var invArangement = [Int](repeating: 0, count: vector.dim)
        
        for (i, j) in axisArangement.enumerated() {
            invArangement[i] = j
        }
        
        //Device.Engine.permuteAxesAdd(input: dstGrad, arangement: invArangement, shape: vector.shape, add: srcGrad, destination: srcGrad)
        Device.Engine.permuteAxesAdd(values: dstGrad, add: srcGrad, result: srcGrad, arangement: invArangement)
    }
}


public extension Tensor {
    func permuted(to axisArangement: [Int]) -> Tensor<Element, Device> {
        precondition(axisArangement.count == dim, "Axis arangement must have dimensionality of source tensor")
        precondition(Set(axisArangement).count == dim, "Axis arangement must not contain duplicate axes")
        
        var dstShape = [Int](repeating: 0, count: dim)
        
        for i in dstShape.indices {
            dstShape[axisArangement[i]] = shape[i]
        }
        
        let result = Tensor<Element, Device>(
            shape: dstShape,
            parent: nil,
            context: requiresGradient ? PermutationOperation(source: self, axisArangement: axisArangement).asAny() : nil
        )
        
        //Device.Engine.permuteAxes(input: self.values, arangement: axisArangement, shape: self.shape, destination: result.values)
        Device.Engine.permuteAxes(values: self.shapedValues, result: result.shapedValues, arangement: axisArangement)
        
        return result
    }
    
    func permuted(to axisArangement: Int...) -> Tensor<Element, Device> {
        return permuted(to: axisArangement)
    }
}
