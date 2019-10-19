//
//  XShape.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
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
    func view(as shape: [Int]) -> XTensor<Element, Device> {
        precondition(shape.count(where: {$0 == -1}) <= 1, "The size of at most one dimension can be unknown (-1).")
        precondition(shape.allSatisfy {$0 >= -1}, "All dimensions must be greater than or equal to -1.")
        precondition(shape.contains(-1) || shape.reduce(1, *) == self.count, "Number of elements in result must be equal to number of elements in source")
        
        var shape = shape
        if let idx = shape.firstIndex(of: -1) {
            let used = shape.lazy.filter {$0 >= 0}.reduce(1, *)
            assert(count % used == 0, "Cannot transform tensor of shape \(self.shape) into tensor shaped \(shape).")
            shape[idx] = count / used
        }
        
        if shape == self.shape {
            return self
        }
        
        return XTensor(
            handle: self.handle,
            shape: shape,
            context: requiresGradient ? XTensorContext(
                tag: "view\(shape)",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.view(as: self.shape)
                }]
            ) : nil
        )
    }
    
    func view(as shape: Int...) -> Self {
        view(as: shape)
    }
    
    func unsqueezed(at axis: Int) -> Self {
        var shape = self.shape
        shape.insert(1, at: axis)
        return view(as: shape)
    }
    
    func squeezed(at axis: Int) -> Self {
        var shape = self.shape
        if shape[axis] == 1 {
            shape.remove(at: axis)
        }
        return view(as: shape)
    }
    
    func squeezed() -> Self {
        view(as: shape.filter {$0 != 1})
    }
    
    func flattened() -> Self {
        view(as: [-1])
    }
}

public extension XTensor {
    func permuted(to axisArangement: [Int]) -> Self {
        precondition(axisArangement.count == dim, "Axis arangement must have dimensionality of source tensor")
        precondition(Set(axisArangement).count == dim, "Axis arangement must not contain duplicate axes")
        
        var dstShape = [Int](repeating: 0, count: dim)
        
        for i in dstShape.indices {
            dstShape[axisArangement[i]] = shape[i]
        }
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: dstShape, type: Element.self)
        Device.Engine.permuteAxes(values: self.values, result: resultBuffer, arangement: axisArangement)
        
        let dim = self.dim
        
        return XTensor(
            using: resultBuffer,
            context: XTensorContext(
                tag: "permute\(axisArangement)",
                sources: [self],
                backpropagateAccumulate: [{ resultGradient, acc in
                    var invArangement = [Int](repeating: 0, count: dim)
                    for (i, j) in axisArangement.enumerated() {
                        invArangement[i] = j
                    }
                    if var acc = acc {
                        acc.addingPermuted(resultGradient, permutation: invArangement)
                        return acc
                    } else {
                        return resultGradient.permuted(to: invArangement)
                    }
                }]
            )
        )
    }
    
    mutating func addingPermuted(_ other: Self, permutation: [Int]) {
        precondition(permutation.count == dim, "Axis arangement must have dimensionality of source tensor")
        precondition(Set(permutation).count == dim, "Axis arangement must not contain duplicate axes")
        
        if self.requiresGradient || other.requiresGradient {
            let original = self
            
            ensureOwnership()
            Device.Engine.permuteAxesAdd(values: other.values, add: self.values, result: self.values, arangement: permutation)
            let dim = self.dim
            self.context = XTensorContext(
                tag: "permutedAdd",
                sources: [original, other],
                backpropagateAccumulate: [
                    { resultGradient, acc in
                        if let acc = acc {
                            return acc + resultGradient
                        } else {
                            return resultGradient
                        }
                    }, { resultGradient, acc in
                        var invArangement = [Int](repeating: 0, count: dim)
                        for (i, j) in permutation.enumerated() {
                            invArangement[i] = j
                        }
                        if var acc = acc {
                            acc.addingPermuted(resultGradient, permutation: invArangement)
                            return acc
                        } else {
                            return resultGradient.permuted(to: invArangement)
                        }
                    }
                ]
            )
        } else {
            ensureOwnership()
            Device.Engine.permuteAxesAdd(values: other.values, add: self.values, result: self.values, arangement: permutation)
        }
    }
    
    func permuted(to axisArangement: Int...) -> Self {
        permuted(to: axisArangement)
    }
    
    func transposed() -> Self {
        permuted(to: [1, 0])
    }
    
    var T: Self {
        transposed()
    }
}
