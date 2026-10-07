//
//  UtilOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.10.19.
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

// MARK: Utility operations

public extension Tensor where Element == Int32 {
    /// One-hot encodes a tensor of indices
    /// - Parameters:
    ///   - dim: Size of encoding axis. Every index must be in `0 ..< dim`.
    ///   - type: Data type of the result.
    /// - Returns: Tensor with the shape `shape + [dim]`, which is 1 at the given indices and 0 everywhere else.
    func oneHotEncoded<Target>(dim: Int, type: Target.Type = Target.self) -> Tensor<Target, Device> {
        // The indices are read once and the result is written once, because every element access on a device is a
        // separate read or command.
        var encoded = [Target](repeating: 0, count: count * dim)
        for (position, index) in elements.enumerated() {
            precondition(0 ..< Int32(dim) ~= index, "Index \(index) is not in 0 ..< \(dim).")
            encoded[position * dim + Int(index)] = 1
        }
        return Tensor<Target, Device>(encoded, shape: shape + [dim])
    }
}

public extension Tensor {
    /// Creates the values from the lower bound up to the upper bound, which is not included, with the given increment.
    ///
    /// The tensor has `ceil((upperBound - lowerBound) / stride)` elements, and the element at index `i` is
    /// `lowerBound + i * stride`. When the increment does not move from the lower bound towards the upper bound,
    /// the tensor is empty.
    /// - Parameters:
    ///   - lowerBound: First value
    ///   - upperBound: Value after the last value
    ///   - stride: Increment between elements. It must not be 0.
    init(linearRampWithLowerBound lowerBound: Element = 0, upperBound: Element, by stride: Element = 1) {
        precondition(stride != 0, "The stride must not be 0.")
        let count = Swift.max(0, Int(((upperBound - lowerBound).doubleValue / stride.doubleValue).rounded(.up)))
        let buffer = Device.Memory.allocateBuffer(withShape: [count], type: Element.self)
        if count > 0 {
            // The engine divides the distance between its bounds into count equal steps.
            Device.Engine.arange(lowerBound: lowerBound, upperBound: lowerBound + stride * Element(count), result: buffer)
        }
        self.init(using: buffer, context: nil)
    }

    /// Repeats the tensor `times` times and stacks the result along the 0th axis.
    /// - Parameter times: Number of repetitions
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func repeated(_ times: Int) -> Self {
        Tensor(stacking: Array(repeating: self, count: times))
    }

    /// Pads the tensor with the given leading and trailing padding for each axis.
    /// - Parameters:
    ///   - value: Padding value
    ///   - padding: Number of padded elements before and after the tensor.
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func padded(with value: Element = 0, padding: [(Int, Int)]) -> Self {
        precondition(padding.count == dim)

        var result = Self(repeating: value, shape: zip(shape, padding).map { $0 + $1.0 + $1.1 })
        let index = zip(shape, padding).map { $1.0 ..< ($0 + $1.0) }
        result[index] = self

        return result
    }

    /// Pads the tensor with the given leading and trailing padding for each axis.
    /// - Parameters:
    ///   - value: Padding value
    ///   - padding: Number of padded elements before and after the tensor.
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func padded(with value: Element = 0, padding: [Int]) -> Self {
        precondition(padding.count == dim)

        var result = Self(repeating: value, shape: zip(shape, padding).map { $0 + $1 * 2 })
        let index = zip(shape, padding).map { $1 ..< ($0 + $1) }
        result[index] = self

        return result
    }

    /// Reverses the tensor along the 0th axis.
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func reversed() -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.reverse(values: values, result: resultBuffer)

        return Tensor(
            using: resultBuffer,
            context: requiresGradient ? TensorContext(
                tag: "Reverse",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.reversed()
                }],
            ) : nil,
        )
    }

    /// Computes a diagonal matrix with the given number of elements below and above the diagonal.
    /// Remaining elements are filled with zeros.
    /// - Parameters:
    ///   - belowDiagonal: Number of elements below diagonal or nil, if all elements should be copied.
    ///   - aboveDiagonal: Number of elements above the diagonal or nil, if all elements should be copied.
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func bandMatrix(belowDiagonal: Int?, aboveDiagonal: Int?) -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.fill(value: 0, result: resultBuffer.values, count: resultBuffer.count)
        Device.Engine.band(buffer: values, result: resultBuffer, belowDiagonal: belowDiagonal, aboveDiagonal: aboveDiagonal)

        return Tensor(
            using: resultBuffer,
            context: requiresGradient ? TensorContext(
                tag: "band",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.bandMatrix(belowDiagonal: belowDiagonal, aboveDiagonal: aboveDiagonal)
                }],
            ) : nil,
        )
    }

    /// Computes the vector of diagonal elements of a matrix
    ///
    /// Source tensor must have dimensionality of 2.
    /// For backpropagation, the matrix must have square shape.
    ///
    /// - Returns: Vector containing matrix diagonal elements
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func diagonalElements() -> Self {
        precondition(dim == 2, "source tensor must be matrix")

        let resultCount = Swift.min(shape[0], shape[1])
        let resultBuffer = Device.Memory.allocateBuffer(withShape: [resultCount], type: Element.self)
        Device.Engine.extractDiagonal(values: values, target: resultBuffer)

        return Tensor(
            using: resultBuffer,
            context: requiresGradient ? TensorContext(
                tag: "diag",
                sources: [self],
                backpropagate: [{ vector in
                    vector.diagonalMatrix()
                }],
            ) : nil,
        )
    }

    /// Computes a matrix that contains the elements of a vector in its diagonal. The remaining elements will be filled with zeros.
    ///
    /// The source tensor must have a dimensionality of two.
    /// The resulting matrix will have a number of rows and columns equal to number of elements in the vector
    ///
    /// - Returns: Square diagonal matrix
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func diagonalMatrix() -> Self {
        precondition(dim == 1, "diagonal element tensor must be vector")

        let matrixValues = Device.Memory.allocateBuffer(withShape: [count, count], type: Element.self)
        Device.Engine.fill(value: 0, result: matrixValues.values, count: matrixValues.count)
        Device.Engine.fillDiagonal(values: values, target: matrixValues)

        return Tensor(
            using: matrixValues,
            context: requiresGradient ? TensorContext(
                tag: "diag-mat",
                sources: [self],
                backpropagate: [{ matrix in
                    matrix.diagonalElements()
                }],
            ) : nil,
        )
    }

    /// Creates a matrix filled with the given value on its diagonal and zeros everywhere else
    /// - Parameters:
    ///   - value: Value to fill diagonal with
    ///   - size: Number of rows and columns of the resulting matrix
    ///   - requiresGradient: Whether to include the tensor in the compute graph for gradient computation
    init(fillingDiagonalWith value: Element, size: Int, requiresGradient: Bool = false) {
        let matrixValues = Device.Memory.allocateBuffer(withShape: [size, size], type: Element.self)
        Device.Engine.fill(value: 0, result: matrixValues.values, count: matrixValues.count)
        Device.Engine.fillDiagonal(value: value, target: matrixValues)

        self.init(
            using: matrixValues,
            context: nil,
        )
        self.requiresGradient = requiresGradient
    }
}

// MARK: Dropout

public extension Tensor {
    /// Sets random elements of the tensor to zero. The other elements keep their values.
    ///
    /// - Parameter rate: Probability, with which an element is set to zero
    /// - Returns: Tensor with the shape of the tensor
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func droppedOut(rate: Float) -> Self {
        var result = Self(uninitializedShape: shape)
        var mask = Self(uninitializedShape: shape)
        Device.FusedOperations.dropout(input: values, rate: rate, result: result.mutableValues, mask: mask.mutableValues)
        return result.attachingContext(tag: "dropout", source: self) { [mask] resultGradient, gradient in
            Composed.dropoutBackward(mask: mask, outputGradient: resultGradient, inputGradient: &gradient)
        } fused: { [mask] resultGradient, gradient in
            Device.FusedOperations.dropoutBackward(mask: mask.values, outputGradient: resultGradient, inputGradient: gradient)
        }
    }
}
