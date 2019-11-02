//
//  Shape.swift
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

//MARK: Tensor Shape Modifiers

public extension Tensor {
    
    /// Reshapes the tensor to the given shape.
    ///
    /// The shape must be compatible with the source shape, i.e. the number of elements must be the same.
    ///
    /// The shape may contain a -1. The size of the result tensor along that axis is then computed as needed.
    ///
    /// - Parameter shape: Shape to view the tensor in.
    /// - Returns: Tensor with given shape, where occurrences of -1 have been replaced.
    func view(as shape: [Int]) -> Tensor<Element, Device> {
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
        
        return Tensor(
            handle: self.handle,
            shape: shape,
            context: requiresGradient ? TensorContext(
                tag: "view\(shape)",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.view(as: self.shape)
                }]
            ) : nil
        )
    }
    
    /// Reshapes the tensor to the given shape.
    ///
    /// The shape must be compatible with the source shape, i.e. the number of elements must be the same.
    ///
    /// The shape may contain a -1. The size of the result tensor along that axis is then computed as needed.
    ///
    /// - Parameter shape: Shape to view the tensor in.
    /// - Returns: Tensor with given shape, where occurrences of -1 have been replaced.
    func view(as shape: Int...) -> Self {
        view(as: shape)
    }
    
    /// Adds an axis to the shape of the tensor.
    /// The axis will have a size of 1.
    /// - Parameter axis: Axis to expand at.
    func unsqueezed(at axis: Int) -> Self {
        var shape = self.shape
        shape.insert(1, at: axis)
        return view(as: shape)
    }
    
    /// Removes an axis from the tensor if the axis has a size of 1.
    /// Otherwise, the original tensor is returned.
    /// - Parameter axis: Axis to remove if possible.
    func squeezed(at axis: Int) -> Self {
        var shape = self.shape
        if shape[axis] == 1 {
            shape.remove(at: axis)
        }
        return view(as: shape)
    }
    
    /// Removes all axes from the tensor that have a size of 1.
    func squeezed() -> Self {
        view(as: shape.filter {$0 != 1})
    }
    
    /// Flattens the tensor into a tensor of shape [count]
    func flattened() -> Self {
        view(as: [-1])
    }
}

public extension Tensor {
    
    /// Swaps the axes of the tensor.
    ///
    /// The axis arangement must have a count of `tensor.dim` and contain all elements in `0 ..< tensor.dim`.
    ///
    /// With axis arangement of [1, 0], this operation is equivalent to `tensor.transposed()`
    ///
    /// - Parameter axisArangement: Arangement of axes in the resulting tensor.
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
        
        return Tensor(
            using: resultBuffer,
            context: TensorContext(
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
    
    /// Permutes the other tensor along the given axes and adds it to the current tensor in place.
    ///
    /// The permutation must have a count of `tensor.dim` and contain all elements in `0 ..< tensor.dim`.
    ///
    /// - Parameters:
    ///   - other: Tensor to add permuted to the current tensor
    ///   - permutation: Desired arangement of axes of the summand.
    mutating func addingPermuted(_ other: Self, permutation: [Int]) {
        precondition(permutation.count == dim, "Axis arangement must have dimensionality of source tensor")
        precondition(Set(permutation).count == dim, "Axis arangement must not contain duplicate axes")
        
        if self.requiresGradient || other.requiresGradient {
            let original = self
            
            ensureOwnership()
            Device.Engine.permuteAxesAdd(values: other.values, add: self.values, result: self.values, arangement: permutation)
            let dim = self.dim
            self.context = TensorContext(
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
    
    
    /// Swaps the axes of the tensor.
    /// 
    /// The axis arangement must have a count of `tensor.dim` and contain all elements in `0 ..< tensor.dim`.
    ///
    /// With axis arangement of [1, 0], this operation is equivalent to `tensor.transposed()`
    ///
    /// - Parameter axisArangement: Arangement of axes in the resulting tensor.
    func permuted(to axisArangement: Int...) -> Self {
        permuted(to: axisArangement)
    }
    
    /// Transposes the given tensor.
    /// The tensor must have a dimensionality of 2.
    func transposed() -> Self {
        permuted(to: [1, 0])
    }
    
    /// Transposes the given tensor.
    /// The tensor must have a dimensionality of 2.
    var T: Self {
        transposed()
    }
}
