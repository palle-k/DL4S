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

// MARK: Tensor Shape Modifiers

public extension Tensor {
    /// Reshapes the tensor to the given shape.
    ///
    /// The shape must be compatible with the source shape, i.e. the number of elements must be the same.
    ///
    /// The shape may contain a -1. The size of the result tensor along that axis is then computed as needed.
    ///
    /// - Parameter shape: Shape to view the tensor in.
    /// - Returns: Tensor with given shape, where occurrences of -1 have been replaced.
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func view(as shape: [Int]) -> Tensor<Element, Device> {
        precondition(shape.count(where: { $0 == -1 }) <= 1, "The size of at most one dimension can be unknown (-1).")
        precondition(shape.allSatisfy { $0 >= -1 }, "All dimensions must be greater than or equal to -1.")
        precondition(shape.contains(-1) || shape.reduce(1, *) == count, "Number of elements in result must be equal to number of elements in source")

        var shape = shape
        if let idx = shape.firstIndex(of: -1) {
            let used = shape.lazy.filter { $0 >= 0 }.reduce(1, *)
            assert(count % used == 0, "Cannot transform tensor of shape \(self.shape) into tensor shaped \(shape).")
            shape[idx] = count / used
        }

        if shape == self.shape {
            return self
        }

        let sourceShape = self.shape
        return Tensor(memory: memory, shape: shape, context: nil).attachingContext(tag: "view\(shape)", source: self) { resultGradient, gradient in
            gradient.add(resultGradient.view(as: sourceShape))
        }
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
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func unsqueezed(at axis: Int) -> Self {
        var shape = shape
        shape.insert(1, at: axis)
        return view(as: shape)
    }

    /// Removes an axis from the tensor if the axis has a size of 1.
    /// Otherwise, the original tensor is returned.
    /// - Parameter axis: Axis to remove if possible.
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func squeezed(at axis: Int) -> Self {
        var shape = shape
        if shape[axis] == 1 {
            shape.remove(at: axis)
        }
        return view(as: shape)
    }

    /// Removes all axes from the tensor that have a size of 1.
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func squeezed() -> Self {
        view(as: shape.filter { $0 != 1 })
    }

    /// Flattens the tensor into a tensor of shape [count]
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func flattened() -> Self {
        view(as: [-1])
    }
}

public extension Tensor {
    /// Swaps the axes of the tensor.
    ///
    /// The axis arrangement must have a count of `tensor.dim` and contain all elements in `0 ..< tensor.dim`.
    ///
    /// With axis arrangement of [1, 0], this operation is equivalent to `tensor.transposed()`
    ///
    /// - Parameter arrangement: Arrangement of axes in the resulting tensor.
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func permuted(to arrangement: [Int]) -> Self {
        precondition(arrangement.count == dim, "Axis arrangement must have dimensionality of source tensor")
        precondition(Set(arrangement).count == dim, "Axis arrangement must not contain duplicate axes")

        var resultShape = [Int](repeating: 0, count: dim)
        for i in resultShape.indices {
            resultShape[arrangement[i]] = shape[i]
        }

        var result = Self(uninitializedShape: resultShape)
        Device.Engine.permuteAxes(values: values, result: result.mutableValues, arrangement: arrangement)

        return result.attachingContext(tag: "permute\(arrangement)", source: self) { resultGradient, gradient in
            gradient.add(resultGradient.permuted(to: Self.inversePermutation(arrangement)))
        } fused: { resultGradient, gradient in
            let inverse = Self.inversePermutation(arrangement)
            if gradient.adds {
                Device.Engine.permuteAxesAdd(values: resultGradient, add: ShapedBuffer(gradient.values), result: gradient.values, arrangement: inverse)
            } else {
                Device.Engine.permuteAxes(values: resultGradient, result: gradient.values, arrangement: inverse)
            }
        }
    }

    /// Swaps the axes of the tensor.
    ///
    /// The axis arrangement must have a count of `tensor.dim` and contain all elements in `0 ..< tensor.dim`.
    ///
    /// With axis arrangement of [1, 0], this operation is equivalent to `tensor.transposed()`
    ///
    /// - Parameter arrangement: Arrangement of axes in the resulting tensor.
    func permuted(to arrangement: Int...) -> Self {
        permuted(to: arrangement)
    }

    /// Transposes the given tensor.
    /// The tensor must have a dimensionality of 2.
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func transposed() -> Self {
        permuted(to: [1, 0])
    }

    /// Transposes the given tensor.
    /// The tensor must have a dimensionality of 2.
    var T: Self {
        transposed()
    }
}

extension Tensor {
    /// The arrangement that undoes a permutation: when axis `i` moves to `permutation[i]`, the inverse moves axis
    /// `permutation[i]` back to `i`.
    static func inversePermutation(_ permutation: [Int]) -> [Int] {
        var inverse = [Int](repeating: 0, count: permutation.count)
        for (axis, destination) in permutation.enumerated() {
            inverse[destination] = axis
        }
        return inverse
    }
}
