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
    srcShape: [Int]
) {
    let srcIndex = srcIndex.dropLast(while: {$0 == nil})
    
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

@_specialize(where Element == Float, C1 == [Range<Int>?], C2 == [Int])
@_specialize(where Element == Int32, C1 == [Range<Int>?], C2 == [Int])
@_specialize(where Element == Double, C1 == [Range<Int>?], C2 == [Int])
@_specialize(where Element == Float, C1 == ArraySlice<Range<Int>?>, C2 == ArraySlice<Int>)
@_specialize(where Element == Int32, C1 == ArraySlice<Range<Int>?>, C2 == ArraySlice<Int>)
@_specialize(where Element == Double, C1 == ArraySlice<Range<Int>?>, C2 == ArraySlice<Int>)
func recursiveRead<Element, C1: RandomAccessCollection, C2: RandomAccessCollection>(
    source: UnsafeBufferPointer<Element>,
    destination: UnsafeMutableBufferPointer<Element>,
    srcIndex: C1,
    srcStrides: C2,
    srcShape: C2
) where C1.Element == Range<Int>?, C2.Element == Int, C1.Index == Int, C2.Index == Int {
    guard !srcIndex.isEmpty else {
        destination.pointee = source.pointee
        return
    }
    let sIdx = srcIndex[srcIndex.startIndex]
    let sStride = srcStrides[srcStrides.startIndex]
    let sDim = srcShape[srcShape.startIndex]
    
    if let sIdx = sIdx {
        if srcIndex.dropFirst().allSatisfy({$0 == nil}) {
            let offset = sIdx.lowerBound &* sStride
            let srcStart = source.advanced(by: offset)
            let count = sIdx.count &* sStride
            destination.assign(from: srcStart, count: count)
        } else {
            let dstShape = zip(srcIndex, srcStrides)
                .filter {$0.0 == nil}
                .map {$1}
            let dstStrides = MemoryOps.strides(from: dstShape)
            let dStride = dstStrides[0]
            
            for i in sIdx {
                let offset = i &* sStride
                let dstOffset = i &* dStride
                
                recursiveRead(
                    source: source.advanced(by: offset),
                    destination: destination.advanced(by: dstOffset),
                    srcIndex: srcIndex.dropFirst(),
                    srcStrides: srcStrides.dropFirst(),
                    srcShape: srcShape.dropFirst()
                )
            }
        }
    } else if srcIndex.allSatisfy({$0 == nil}) {
        let count = sDim &* sStride
        destination.assign(from: source, count: count)
    } else {
        let dstShape = zip(srcShape, srcIndex).map {
            $1?.count ?? $0
        }
        let dstStrides = MemoryOps.strides(from: dstShape)
        let dStride = dstStrides[0]
        
        for i in 0 ..< sDim {
            let srcOffset = i &* sStride
            let srcStart = source.advanced(by: srcOffset)
            let dstOffset = i &* dStride
            let dstStart = destination.advanced(by: dstOffset)
            
            recursiveRead(
                source: srcStart,
                destination: dstStart,
                srcIndex: srcIndex.dropFirst(),
                srcStrides: srcStrides.dropFirst(),
                srcShape: srcShape.dropFirst()
            )
        }
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
    dstShape: [Int]
    ) {
    let dstIndex = dstIndex.reversed().drop(while: {$0 == nil}).reversed()
    
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
        let index = zip(dstIndex, index).map {$0 ?? $1}
        let baseIndex = zip(index, dstStrides).map(&*).reduce(0, &+)
        let srcIndex = i &* copyCount
        destination.advanced(by: baseIndex)
            .assign(from: source.advanced(by: srcIndex), count: copyCount)
    }
}

@_specialize(where Element == Float, C1 == [Range<Int>?], C2 == [Int])
@_specialize(where Element == Int32, C1 == [Range<Int>?], C2 == [Int])
@_specialize(where Element == Double, C1 == [Range<Int>?], C2 == [Int])
@_specialize(where Element == Float, C1 == ArraySlice<Range<Int>?>, C2 == ArraySlice<Int>)
@_specialize(where Element == Int32, C1 == ArraySlice<Range<Int>?>, C2 == ArraySlice<Int>)
@_specialize(where Element == Double, C1 == ArraySlice<Range<Int>?>, C2 == ArraySlice<Int>)
func recursiveWrite<Element, C1: RandomAccessCollection, C2: RandomAccessCollection>(
    source: UnsafeBufferPointer<Element>,
    destination: UnsafeMutableBufferPointer<Element>,
    dstIndex: C1,
    dstStrides: C2,
    dstShape: C2
) where C1.Element == Range<Int>?, C2.Element == Int {
    guard
        let dIdx = dstIndex.first,
        let dStride = dstStrides.first,
        let dDim = dstShape.first
    else {
        destination.pointee = source.pointee
        return
    }
    
    if let dIdx = dIdx {
        if dstIndex.dropFirst().allSatisfy({$0 == nil}) {
            let offset = dIdx.lowerBound &* dStride
            let dstStart = destination.advanced(by: offset)
            let count = dIdx.count &* dStride
            dstStart.assign(from: source, count: count)
        } else {
            let srcShape = zip(dstIndex, dstStrides)
                .map {$1}
            let srcStrides = MemoryOps.strides(from: srcShape)
            let sStride = srcStrides[0]
            
            for i in dIdx {
                let offset = i &* dStride
                let dstStart = destination.advanced(by: offset)
                let srcOffset = i &* sStride
                let srcStart = source.advanced(by: srcOffset)
                
                recursiveWrite(
                    source: srcStart,
                    destination: dstStart,
                    dstIndex: dstIndex.dropFirst(),
                    dstStrides: dstStrides.dropFirst(),
                    dstShape: dstShape.dropFirst()
                )
            }
        }
    } else if dstIndex.allSatisfy({$0 == nil}) {
        let count = dDim &* dStride
        destination.assign(from: source, count: count)
    } else {
        let srcShape = zip(dstIndex, dstShape).map {
            $0?.count ?? $1
        }
        
        let srcStrides = MemoryOps.strides(from: srcShape)
        let sStride = srcStrides[0]
        
        for i in 0 ..< dDim {
            let srcOffset = i &* sStride
            let srcStart = source.advanced(by: srcOffset)
            let dstOffset = i &* dStride
            let dstStart = destination.advanced(by: dstOffset)
            
            recursiveWrite(
                source: srcStart,
                destination: dstStart,
                dstIndex: dstIndex.dropFirst(),
                dstStrides: dstStrides.dropFirst(),
                dstShape: dstShape.dropFirst()
            )
        }
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
        return zip(shape, strides).map { dim, str in (linearIndex / str) % dim}
    }
}
