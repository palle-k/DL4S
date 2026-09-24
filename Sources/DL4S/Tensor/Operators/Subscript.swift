//
//  Subscript.swift
//  DL4S
//
//  Created by Palle Klewitz on 15.10.19.
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

// MARK: Subscripting

public extension Tensor {
    /// Gets or sets a subtensor at the given index.
    ///
    /// When an element of the index is nil, all elements along the corresponding axis are read or written.
    ///
    /// Example:
    /// ```
    /// let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
    /// print(a[nil, 1]) // [2, 5]
    /// print(a[1]) // [4, 5, 6]
    /// print(a[1, nil] == a[1]) // true
    /// ```
    subscript(index: [Int?]) -> Self {
        get {
            let index = Self.resolvingNegativeIndices(index, shape: shape)
            let (val, isCopy, shape) = Device.Memory.get(slice: index, of: values.values, with: shape)
            let handle = TensorHandle(values: val, parent: isCopy ? nil : handle)

            return Tensor(
                handle: handle,
                shape: shape,
                context: requiresGradient ? TensorContext(
                    tag: "read",
                    sources: [self],
                    backpropagateAccumulate: [{ resultGradient, acc in
                        // Without a gradient graph, the gradient of a contiguous slice is added to the accumulator in place.
                        if !resultGradient.requiresGradient, !(acc?.requiresGradient ?? false), let offset = Self.contiguousOffset(of: index, shape: self.shape) {
                            return Self.addingInPlace(resultGradient, at: offset, to: acc, shape: self.shape)
                        }
                        var result = acc ?? Self(repeating: 0, shape: self.shape)
                        // The slice view must be released before the write, or the write copies the whole accumulator.
                        let slice = result[index] + resultGradient
                        result[index] = slice
                        return result
                    }],
                ) : nil,
            )
        }

        set(slice) {
            precondition(!requiresGradient, "Cannot write into tensor that requires gradient.")

            let index = Self.resolvingNegativeIndices(index, shape: shape)
            if slice.dim == 0, dim - index.filter({ $0 != nil }).count > 0 {
                fatalError("Assigning from a single value not supported yet.")
            }

            // TODO: Proper handling of replacement when gradient is computed.

            Device.Memory.set(slice: index, of: mutableValues.values, with: shape, from: slice.values.values, with: slice.shape)

            if slice.requiresGradient {
                requiresGradient = true
                context = TensorContext(
                    tag: "write",
                    sources: [slice],
                    backpropagate: [{ resultGradient in
                        resultGradient[index]
                    }],
                )
            }
        }
    }

    /// Gets or sets a subtensor at the given index.
    ///
    /// When an element of the index is nil, all elements along the corresponding axis are read or written.
    ///
    /// Example:
    /// ```
    /// let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
    /// print(a[nil, 1]) // [2, 5]
    /// print(a[1]) // [4, 5, 6]
    /// print(a[1, nil] == a[1]) // true
    /// ```
    subscript(index: Int?...) -> Self {
        get { self[index] }
        set(slice) { self[index] = slice }
    }

    /// Gets or sets a subtensor at the given window.
    ///
    /// When an element of the index is nil, all elements along the corresponding axis are read or written.
    ///
    /// Example:
    /// ```
    /// let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
    /// print(a[0 ..< 2]) // [[1, 2], [4, 5]]
    /// print(a[nil, 0 ..< 1]) // [[1, 2, 3]]
    /// ```
    subscript(index: [Range<Int>?]) -> Self {
        get {
            let (val, isCopy, shape) = Device.Memory.get(slice: index, of: values.values, with: shape)

            let handle: TensorHandle<Element, Device> = if isCopy {
                TensorHandle(values: val)
            } else {
                TensorHandle(values: val, parent: self.handle)
            }

            return Tensor(
                handle: handle,
                shape: shape,
                context: requiresGradient ? TensorContext(
                    tag: "SubscriptRangeRead",
                    sources: [self],
                    backpropagateAccumulate: [{ resultGradient, acc in
                        // Without a gradient graph, the gradient of a contiguous slice is added to the accumulator in place.
                        if !resultGradient.requiresGradient, !(acc?.requiresGradient ?? false), let offset = Self.contiguousOffset(of: index, shape: self.shape) {
                            return Self.addingInPlace(resultGradient, at: offset, to: acc, shape: self.shape)
                        }
                        var result = acc ?? Self(repeating: 0, shape: self.shape)
                        // The slice view must be released before the write, or the write copies the whole accumulator.
                        let slice = result[index] + resultGradient
                        result[index] = slice
                        return result
                    }],
                ) : nil,
            )
        }

        set(slice) {
            if slice.dim == 0, dim - index.filter({ $0 != nil }).count > 0 {
                fatalError("Assigning from a single value not supported yet.")
            }

            // TODO: Proper handling of replacement when gradient is computed.

            Device.Memory.set(slice: index, of: mutableValues.values, with: shape, from: slice.values.values, with: slice.shape)

            if slice.requiresGradient {
                requiresGradient = true
                context = TensorContext(
                    tag: "SubscriptRangeWrite",
                    sources: [slice],
                    backpropagate: [{ resultGradient in
                        resultGradient[index]
                    }],
                )
            }
        }
    }

    /// Gets or sets a subtensor at the given window.
    ///
    /// When an element of the index is nil, all elements along the corresponding axis are read or written.
    ///
    /// Example:
    /// ```
    /// let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
    /// print(a[0 ..< 2]) // [[1, 2], [4, 5]]
    /// print(a[nil, 0 ..< 1]) // [[1, 2, 3]]
    /// ```
    subscript(index: Range<Int>?...) -> Self {
        get { self[index] }
        set(slice) { self[index] = slice }
    }
}

extension Tensor {
    /// Offset of the slice at an index of leading integers, which is contiguous in memory, or nil for other indices.
    static func contiguousOffset(of index: [Int?], shape: [Int]) -> Int? {
        var count = index.count
        while count > 0, index[count - 1] == nil {
            count -= 1
        }
        let strides = MemoryOps.strides(from: shape)
        var offset = 0
        for axis in 0 ..< count {
            guard let position = index[axis] else {
                return nil
            }
            offset += position * strides[axis]
        }
        return offset
    }

    /// Offset of the slice at an index with one range on the first axis, which is contiguous in memory, or nil for other indices.
    static func contiguousOffset(of index: [Range<Int>?], shape: [Int]) -> Int? {
        guard let first = index.first, let range = first, index.dropFirst().allSatisfy({ $0 == nil }) else {
            return nil
        }
        return range.lowerBound * shape.dropFirst().reduce(1, *)
    }

    /// Adds the gradient of a contiguous slice to the accumulated gradient of the whole tensor, in place.
    ///
    /// Without an accumulator, the accumulated gradient starts at 0. Neither tensor may record a gradient graph.
    static func addingInPlace(_ gradient: Self, at offset: Int, to accumulator: consuming Self?, shape: [Int]) -> Self {
        var result = accumulator ?? Self(repeating: 0, shape: shape)
        // The accumulator is uniquely referenced, so the write does not copy it.
        let slice = Device.Memory.advance(buffer: result.mutableValues.values, by: offset)
        Device.Engine.vAdd(lhs: Buffer(slice), rhs: gradient.values.values, result: slice, count: gradient.count)
        return result
    }

    /// Replaces negative indices, which count from the end of their axis, with the corresponding positive indices.
    @inline(__always)
    static func resolvingNegativeIndices(_ index: [Int?], shape: [Int]) -> [Int?] {
        guard index.contains(where: { ($0 ?? 0) < 0 }) else {
            return index
        }
        return zip(index, shape).map { position, size in
            position.map { $0 < 0 ? size + $0 : $0 }
        }
    }
}
