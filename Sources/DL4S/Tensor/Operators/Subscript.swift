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
            Self.checkBounds(of: index, shape: shape)
            let (val, isCopy, shape) = Device.Memory.get(slice: index, of: values.values, with: shape)
            let handle = TensorHandle(values: val, parent: isCopy ? nil : handle)
            let sourceShape = self.shape

            // Subscripts are small operations that run in loops, so they use the context with one closure per source directly.
            return Tensor(
                handle: handle,
                shape: shape,
                context: requiresGradient ? TensorContext(
                    tag: "read",
                    sources: [self],
                    backpropagateAccumulate: [{ resultGradient, accumulated in
                        Self.addingSlice(resultGradient, to: accumulated, shape: sourceShape, contiguousOffset: Self.contiguousOffset(of: index, shape: sourceShape), read: { $0[index] }, write: { $0[index] = $1 })
                    }],
                ) : nil,
            )
        }

        set(slice) {
            precondition(!requiresGradient, "Cannot write into tensor that requires gradient.")

            let index = Self.resolvingNegativeIndices(index, shape: shape)
            Self.checkBounds(of: index, shape: shape)
            precondition(slice.dim > 0 || index.count(where: { $0 != nil }) == dim, "A scalar can only be assigned to a single element.")

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
            Self.checkBounds(of: index, shape: shape)
            let (val, isCopy, shape) = Device.Memory.get(slice: index, of: values.values, with: shape)

            let handle: TensorHandle<Element, Device> = if isCopy {
                TensorHandle(values: val)
            } else {
                TensorHandle(values: val, parent: self.handle)
            }
            let sourceShape = self.shape

            // Subscripts are small operations that run in loops, so they use the context with one closure per source directly.
            return Tensor(
                handle: handle,
                shape: shape,
                context: requiresGradient ? TensorContext(
                    tag: "SubscriptRangeRead",
                    sources: [self],
                    backpropagateAccumulate: [{ resultGradient, accumulated in
                        Self.addingSlice(resultGradient, to: accumulated, shape: sourceShape, contiguousOffset: Self.contiguousOffset(of: index, shape: sourceShape), read: { $0[index] }, write: { $0[index] = $1 })
                    }],
                ) : nil,
            )
        }

        set(slice) {
            // The context of the result only has the slice as its source, so the gradient of the values before the write
            // would be lost.
            precondition(!requiresGradient, "Cannot write into tensor that requires gradient.")
            Self.checkBounds(of: index, shape: shape)
            precondition(slice.dim > 0, "A scalar cannot be assigned to a range.")

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

    /// Checks that an index with resolved negative positions lies in a tensor of the given shape.
    @inline(__always)
    static func checkBounds(of index: [Int?], shape: [Int]) {
        precondition(index.count <= shape.count, "The index has more axes than the tensor.")
        for (position, size) in zip(index, shape) {
            if let position {
                precondition(position >= 0 && position < size, "Index \(position) is out of range for an axis of size \(size).")
            }
        }
    }

    /// Checks that a window lies in a tensor of the given shape.
    @inline(__always)
    static func checkBounds(of index: [Range<Int>?], shape: [Int]) {
        precondition(index.count <= shape.count, "The index has more axes than the tensor.")
        for (range, size) in zip(index, shape) {
            if let range {
                precondition(range.lowerBound >= 0 && range.upperBound <= size, "Range \(range) is out of range for an axis of size \(size).")
            }
        }
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

extension Tensor {
    /// Adds the gradient of a slice of the source to the accumulated gradient of the source, which has the given shape.
    ///
    /// `read` reads the slice from, and `write` writes it into, a tensor with the shape of the source. Without a gradient graph,
    /// the gradient is added in place: with one addition for a slice that is contiguous in memory, which starts at
    /// `contiguousOffset`, and with a subscript write otherwise.
    static func addingSlice(
        _ sliceGradient: Self,
        to accumulated: consuming Self?,
        shape: [Int],
        contiguousOffset: Int?,
        read: (Self) -> Self,
        write: (inout Self, Self) -> Void,
    ) -> Self {
        guard !sliceGradient.requiresGradient, !(accumulated?.requiresGradient ?? false) else {
            var scattered = Self(repeating: 0, shape: shape)
            write(&scattered, sliceGradient)
            return accumulated.map { $0 + scattered } ?? scattered
        }
        var result = accumulated ?? Self(repeating: 0, shape: shape)
        if let contiguousOffset {
            // The accumulated gradient is uniquely referenced, so the write does not copy it.
            let target = Device.Memory.advance(buffer: result.mutableValues.values, by: contiguousOffset)
            Device.Engine.vAdd(lhs: Buffer(target), rhs: sliceGradient.values.values, result: target, count: sliceGradient.count)
        } else {
            // The slice view must be released before the write, or the write copies the whole accumulated gradient.
            let sum = read(result) + sliceGradient
            write(&result, sum)
        }
        return result
    }
}
