//
//  MemoryOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
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

@_specialize(where Element == Float)
@_specialize(where Element == Int32)
@_specialize(where Element == Double)
func iterativeRead<Element>(
    source: UnsafeBufferPointer<Element>,
    destination: UnsafeMutableBufferPointer<Element>,
    srcIndex: [Int?],
    srcStrides: [Int],
    srcShape: [Int],
) {
    let srcIndex = Array(srcIndex.dropLast(while: { $0 == nil }))

    if srcIndex.isEmpty {
        let count = srcShape[0] &* srcStrides[0]
        destination.assign(from: source, count: count)
        return
    }

    // Every index of the axes without a position selects a contiguous run of copyCount elements, which are written in order.
    let copyCount = srcStrides[srcIndex.count - 1]
    let iterShape = zip(srcIndex, srcShape).map { position, size in position == nil ? size : 1 }
    let strides = Array(srcStrides.prefix(srcIndex.count))
    let baseOffset = zip(srcIndex, strides).reduce(0) { offset, axis in offset &+ (axis.0 ?? 0) &* axis.1 }
    var destinationOffset = 0
    StridedIteration.forEachOffset(shape: iterShape, strides: strides, strides) { offset, _ in
        destination.advanced(by: destinationOffset).assign(from: source.advanced(by: baseOffset &+ offset), count: copyCount)
        destinationOffset &+= copyCount
    }
}

@_specialize(where Element == Float)
@_specialize(where Element == Int32)
@_specialize(where Element == Double)
func iterativeWrite<Element>(
    source: UnsafeBufferPointer<Element>,
    destination: UnsafeMutableBufferPointer<Element>,
    dstIndex: [Int?],
    dstStrides: [Int],
    dstShape: [Int],
) {
    let dstIndex = Array(dstIndex.dropLast(while: { $0 == nil }))

    if dstIndex.isEmpty {
        let count = dstShape[0] &* dstStrides[0]
        destination.assign(from: source, count: count)
        return
    }

    // Every index of the axes without a position selects a contiguous run of copyCount elements, which are read in order.
    let copyCount = dstStrides[dstIndex.count - 1]
    let iterShape = zip(dstIndex, dstShape).map { position, size in position == nil ? size : 1 }
    let strides = Array(dstStrides.prefix(dstIndex.count))
    let baseOffset = zip(dstIndex, strides).reduce(0) { offset, axis in offset &+ (axis.0 ?? 0) &* axis.1 }
    var sourceOffset = 0
    StridedIteration.forEachOffset(shape: iterShape, strides: strides, strides) { offset, _ in
        destination.advanced(by: baseOffset &+ offset).assign(from: source.advanced(by: sourceOffset), count: copyCount)
        sourceOffset &+= copyCount
    }
}

enum MemoryOps {
    @inline(__always)
    static func strides(from shape: [Int]) -> [Int] {
        let dim = shape.count

        if dim == 0 {
            return []
        }

        var str = [Int](repeating: 1, count: dim)
        for i in (0 ..< dim - 1).reversed() {
            str[i] = str[i &+ 1] &* shape[i &+ 1]
        }
        return str
    }
}
