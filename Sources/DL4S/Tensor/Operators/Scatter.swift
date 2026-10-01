//
//  Scatter.swift
//  DL4S
//
//  Created by Palle Klewitz on 29.10.19.
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

// MARK: Indexing

public extension Tensor {
    /// Gathers elements at indices determined by the context along the specified axis.
    ///
    /// ```
    /// Example: Gathering from Tensor [[1,2,3], [4,5,6], [7,8,9]]
    /// Context: [0, 1, 2], axis: 0
    /// => [1,5,9]
    ///
    /// Context: [2, 2, 1], axis: 1
    /// => [3, 6, 8]
    /// ```
    ///
    /// - Parameters:
    ///   - context: Indices along gathering axis.
    ///   - axis: Axis to gather from
    func gather(using context: Tensor<Int32, Device>, alongAxis axis: Int, ignoreIndex: Int32 = -1) -> Self {
        var resultShape = shape
        resultShape.remove(at: axis)
        let originalAxisSize = shape[axis]

        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)

        Device.Engine.gather(expanded: values, context: context.values, result: resultBuffer, axis: axis, ignoreIndex: ignoreIndex)
        return Tensor(
            using: resultBuffer,
            context: requiresGradient ? TensorContext(
                tag: "gather",
                sources: [self],
                backpropagate: [{ resultGradient -> Self in
                    resultGradient.scatter(using: context, alongAxis: axis, withSize: originalAxisSize, ignoreIndex: ignoreIndex)
                }],
            ) : nil,
        )
    }

    /// Scatters elements to indices determined by the context along the specified axis.
    ///
    /// ```
    /// Example: Scattering Tensor [3, 1, 4]
    /// Context: [0, 1, 2], axis: 0, axisSize: 3
    /// => [[3, 0, 0], [0, 1, 0], [0, 0, 4]]
    ///
    /// Context: [2, 2, 1], axis: 1, axisSize: 3
    /// => [[0, 0, 3], [0, 0, 1], [0, 4, 0]]
    /// ```
    ///
    /// - Parameters:
    ///   - context: Indices along scattering axis
    ///   - axis: Axis to scatter along
    ///   - axisSize: Number of elements along the axis in the result tensor. Must be greater than `max(context)`
    func scatter(using context: Tensor<Int32, Device>, alongAxis axis: Int, withSize axisSize: Int, ignoreIndex: Int32 = -1) -> Self {
        var resultShape = shape
        resultShape.insert(axisSize, at: axis)
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.scatter(reduced: values, context: context.values, result: resultBuffer, axis: axis, ignoreIndex: ignoreIndex)
        return Tensor(
            using: resultBuffer,
            context: requiresGradient ? TensorContext(
                tag: "scatter",
                sources: [self],
                backpropagate: [{ resultGradient -> Self in
                    resultGradient.gather(using: context, alongAxis: axis, ignoreIndex: ignoreIndex)
                }],
            ) : nil,
        )
    }

    /// Gathers the rows at the given indices.
    ///
    /// A row is a slice along the first axis. The result has the shape of the indices followed by the row shape, and
    /// the row at a position of the indices is the row of the tensor at that index, or zeros for `ignoreIndex`.
    ///
    /// ```
    /// Example: Gathering rows from Tensor [[1, 2], [3, 4], [5, 6]]
    /// Indices: [2, 0, 2]
    /// => [[5, 6], [1, 2], [5, 6]]
    /// ```
    ///
    /// - Parameters:
    ///   - indices: Indices of rows. Every index must be in `0 ..< shape[0]` or equal to `ignoreIndex`.
    ///   - ignoreIndex: Index for which the row of the result is zero
    /// - Returns: Tensor with the shape `indices.shape + shape[1...]`
    func gatheringRows(at indices: Tensor<Int32, Device>, ignoreIndex: Int32 = -1) -> Self {
        precondition(dim >= 1, "Rows can only be gathered from a tensor with at least one axis.")
        let shape = shape
        let rowsShape = [indices.count] + shape.dropFirst()
        var result = Self(uninitializedShape: indices.shape + shape.dropFirst())
        Device.Engine.gatherRows(values: values, indices: indices.values, result: result.mutableValues.reshaped(to: rowsShape), ignoreIndex: ignoreIndex)
        return result.attachingContext(tag: "gatherRows", source: self) { resultGradient, gradient in
            guard !resultGradient.requiresGradient, gradient.value?.requiresGradient != true else {
                gradient.add(resultGradient.scatteringRows(at: indices, rowCount: shape[0], ignoreIndex: ignoreIndex))
                return
            }
            // The accumulated gradient was taken out of the accumulator, so the rows are added to it in place.
            var accumulated = gradient.value.take() ?? Self(repeating: 0, shape: shape)
            Device.Engine.scatterAddRows(values: resultGradient.values.reshaped(to: rowsShape), indices: indices.values, result: accumulated.mutableValues, ignoreIndex: ignoreIndex)
            gradient.value = accumulated
        }
    }

    /// Adds the rows of the tensor to the rows at the given indices of a tensor that is filled with zeros.
    ///
    /// The shape of the tensor starts with the shape of the indices, and the rest is the row shape. Rows with the same
    /// index are added, and rows with the index `ignoreIndex` are not added.
    ///
    /// ```
    /// Example: Scattering rows of Tensor [[1, 2], [3, 4], [5, 6]]
    /// Indices: [2, 0, 2], row count: 3
    /// => [[3, 4], [0, 0], [6, 8]]
    /// ```
    ///
    /// - Parameters:
    ///   - indices: Indices of rows of the result. Every index must be in `0 ..< rowCount` or equal to `ignoreIndex`.
    ///   - rowCount: Number of rows of the result
    ///   - ignoreIndex: Index of rows that are not added
    /// - Returns: Tensor with the shape `[rowCount] + shape[indices.dim...]`
    func scatteringRows(at indices: Tensor<Int32, Device>, rowCount: Int, ignoreIndex: Int32 = -1) -> Self {
        precondition(shape.starts(with: indices.shape), "The shape of the tensor must start with the shape of the indices.")
        var result = Self(repeating: 0, shape: [rowCount] + shape[indices.dim...])
        Device.Engine.scatterAddRows(values: values.reshaped(to: [indices.count] + shape[indices.dim...]), indices: indices.values, result: result.mutableValues, ignoreIndex: ignoreIndex)
        return result.attachingContext(tag: "scatterRows", source: self) { resultGradient, gradient in
            gradient.add(resultGradient.gatheringRows(at: indices, ignoreIndex: ignoreIndex))
        }
    }
}
