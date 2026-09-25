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
@inline(__always)
func iterativeRead<Element>(
    source: UnsafeBufferPointer<Element>,
    destination: UnsafeMutableBufferPointer<Element>,
    srcIndex: [Int?],
    srcStrides: [Int],
    srcShape: [Int],
) {
    let srcIndex = srcIndex.dropLast(while: { $0 == nil })

    if srcIndex.count == 0 {
        let count = srcShape[0] &* srcStrides[0]
        destination.assign(from: source, count: count)
        return
    }

    let copyCount = srcStrides[srcIndex.count - 1]

    let iterShape = zip(srcIndex, srcShape).map { idx, dim in
        idx == nil ? dim : 1
    }

    let indices = iterate(iterShape)

    for i in 0 ..< indices.count {
        let index = indices[i]
        var baseIndex = 0
        let dstIndex = i &* copyCount
        for j in 0 ..< index.count {
            baseIndex &+= (srcIndex[j] ?? index[j]) &* srcStrides[j]
        }
        destination
            .advanced(by: dstIndex)
            .assign(from: source.advanced(by: baseIndex), count: copyCount)
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
    let dstIndex = dstIndex.reversed().drop(while: { $0 == nil }).reversed()

    if dstIndex.count == 0 {
        let count = dstShape[0] &* dstStrides[0]
        destination.assign(from: source, count: count)
        return
    }

    let copyCount = dstStrides[dstIndex.count - 1]

    let iterShape = zip(dstIndex, dstShape).map { idx, dim in
        idx == nil ? dim : 1
    }

    for (i, index) in iterate(iterShape).enumerated() {
        let index = zip(dstIndex, index).map { $0 ?? $1 }
        let baseIndex = zip(index, dstStrides).map(&*).reduce(0, &+)
        let srcIndex = i &* copyCount
        destination.advanced(by: baseIndex)
            .assign(from: source.advanced(by: srcIndex), count: copyCount)
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

    static func linearIndex(from index: [Int], shape: [Int]) -> Int {
        let strides = MemoryOps.strides(from: shape)
        return zip(index, strides).map(&*).reduce(0, &+)
    }

    static func index(from linearIndex: Int, shape: [Int]) -> [Int] {
        let strides = MemoryOps.strides(from: shape)
        return zip(shape, strides).map { dim, str in (linearIndex / str) % dim }
    }
}
