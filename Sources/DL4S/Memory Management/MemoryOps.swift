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


func recursiveRead<Element>(
    source: UnsafeBufferPointer<Element>,
    destination: UnsafeMutableBufferPointer<Element>,
    srcIndex: [Int?],
    srcStrides: [Int],
    srcShape: [Int]
) {
    guard
        let sIdx = srcIndex.first,
        let sStride = srcStrides.first,
        let sDim = srcShape.first
    else {
        destination.pointee = source.pointee
        return
    }
    
    if let sIdx = sIdx {
        let offset = sIdx * sStride
        
        let srcStart = source.advanced(by: offset)
        
        recursiveRead(
            source: srcStart,
            destination: destination,
            srcIndex: Array(srcIndex.dropFirst()),
            srcStrides: Array(srcStrides.dropFirst()),
            srcShape: Array(srcShape.dropFirst())
        )
    } else if srcIndex.allSatisfy({$0 == nil}) {
        let count = sDim * sStride
        destination.assign(from: source, count: count)
    } else {
        let dstShape = zip(srcIndex, srcStrides)
            .filter {$0.0 == nil}
            .map {$1}
        let dstStrides = MemoryOps.strides(from: dstShape)
        let dStride = dstStrides[0]
        
        for i in 0 ..< sDim {
            let srcOffset = i * sStride
            let srcStart = source.advanced(by: srcOffset)
            let dstOffset = i * dStride
            let dstStart = destination.advanced(by: dstOffset)
            
            recursiveRead(
                source: srcStart,
                destination: dstStart,
                srcIndex: Array(srcIndex.dropFirst()),
                srcStrides: Array(srcStrides.dropFirst()),
                srcShape: Array(srcShape.dropFirst())
            )
        }
    }
}

func recursiveRead<Element>(
    source: UnsafeBufferPointer<Element>,
    destination: UnsafeMutableBufferPointer<Element>,
    srcIndex: [Range<Int>?],
    srcStrides: [Int],
    srcShape: [Int]
) {
    guard
        let sIdx = srcIndex.first,
        let sStride = srcStrides.first,
        let sDim = srcShape.first
    else {
        destination.pointee = source.pointee
        return
    }
    
    if let sIdx = sIdx {
        if srcIndex.dropFirst().allSatisfy({$0 == nil}) {
            let offset = sIdx.lowerBound * sStride
            let srcStart = source.advanced(by: offset)
            let count = sIdx.count * sStride
            destination.assign(from: srcStart, count: count)
        } else {
            let dstShape = zip(srcIndex, srcStrides)
                .filter {$0.0 == nil}
                .map {$1}
            let dstStrides = MemoryOps.strides(from: dstShape)
            let dStride = dstStrides[0]
            
            for i in sIdx {
                let offset = i * sStride
                let dstOffset = i * dStride
                
                recursiveRead(
                    source: source.advanced(by: offset),
                    destination: destination.advanced(by: dstOffset),
                    srcIndex: Array(srcIndex.dropFirst()),
                    srcStrides: Array(srcStrides.dropFirst()),
                    srcShape: Array(srcShape.dropFirst())
                )
            }
        }
    } else if srcIndex.allSatisfy({$0 == nil}) {
        let count = sDim * sStride
        destination.assign(from: source, count: count)
    } else {
        let dstShape = zip(srcIndex, srcStrides)
            .filter {$0.0 == nil}
            .map {$1}
        let dstStrides = MemoryOps.strides(from: dstShape)
        let dStride = dstStrides[0]
        
        for i in 0 ..< sDim {
            let srcOffset = i * sStride
            let srcStart = source.advanced(by: srcOffset)
            let dstOffset = i * dStride
            let dstStart = destination.advanced(by: dstOffset)
            
            recursiveRead(
                source: srcStart,
                destination: dstStart,
                srcIndex: Array(srcIndex.dropFirst()),
                srcStrides: Array(srcStrides.dropFirst()),
                srcShape: Array(srcShape.dropFirst())
            )
        }
    }
}

func recursiveWrite<Element>(
    source: UnsafeBufferPointer<Element>,
    destination: UnsafeMutableBufferPointer<Element>,
    dstIndex: [Int?],
    dstStrides: [Int],
    dstShape: [Int]
) {
    guard
        let dIdx = dstIndex.first,
        let dStride = dstStrides.first,
        let dDim = dstShape.first
    else {
        destination.pointee = source.pointee
        return
    }
    
    if let dIdx = dIdx {
        let offset = dIdx * dStride
        let dstStart = destination.advanced(by: offset)
        
        recursiveWrite(
            source: source,
            destination: dstStart,
            dstIndex: Array(dstIndex.dropFirst()),
            dstStrides: Array(dstStrides.dropFirst()),
            dstShape: Array(dstShape.dropFirst())
        )
    } else if dstIndex.allSatisfy({$0 == nil}) {
        let count = dDim * dStride
        destination.assign(from: source, count: count)
    } else {
        let srcShape = zip(dstIndex, dstStrides)
            .filter {$0.0 == nil}
            .map {$1}
        let srcStrides = MemoryOps.strides(from: srcShape)
        let sStride = srcStrides[0]
        
        for i in 0 ..< dDim {
            let srcOffset = i * sStride
            let srcStart = source.advanced(by: srcOffset)
            let dstOffset = i * dStride
            let dstStart = destination.advanced(by: dstOffset)
            
            recursiveWrite(
                source: srcStart,
                destination: dstStart,
                dstIndex: Array(dstIndex.dropFirst()),
                dstStrides: Array(dstStrides.dropFirst()),
                dstShape: Array(dstShape.dropFirst())
            )
        }
    }
}

func recursiveWrite<Element>(
    source: UnsafeBufferPointer<Element>,
    destination: UnsafeMutableBufferPointer<Element>,
    dstIndex: [Range<Int>?],
    dstStrides: [Int],
    dstShape: [Int]
) {
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
            let offset = dIdx.lowerBound * dStride
            let dstStart = destination.advanced(by: offset)
            let count = dIdx.count * dStride
            dstStart.assign(from: source, count: count)
        } else {
            let srcShape = zip(dstIndex, dstStrides)
                .filter {$0.0 == nil}
                .map {$1}
            let srcStrides = MemoryOps.strides(from: srcShape)
            let sStride = srcStrides[0]
            
            for i in dIdx {
                let offset = i * dStride
                let dstStart = destination.advanced(by: offset)
                let srcOffset = i * sStride
                let srcStart = source.advanced(by: srcOffset)
                
                recursiveWrite(
                    source: srcStart,
                    destination: dstStart,
                    dstIndex: Array(dstIndex.dropFirst()),
                    dstStrides: Array(dstStrides.dropFirst()),
                    dstShape: Array(dstShape.dropFirst())
                )
            }
        }
    } else if dstIndex.allSatisfy({$0 == nil}) {
        let count = dDim * dStride
        destination.assign(from: source, count: count)
    } else {
        let srcShape = zip(dstIndex, dstStrides)
            .filter {$0.0 == nil}
            .map {$1}
        let srcStrides = MemoryOps.strides(from: srcShape)
        let sStride = srcStrides[0]
        
        for i in 0 ..< dDim {
            let srcOffset = i * sStride
            let srcStart = source.advanced(by: srcOffset)
            let dstOffset = i * dStride
            let dstStart = destination.advanced(by: dstOffset)
            
            recursiveWrite(
                source: srcStart,
                destination: dstStart,
                dstIndex: Array(dstIndex.dropFirst()),
                dstStrides: Array(dstStrides.dropFirst()),
                dstShape: Array(dstShape.dropFirst())
            )
        }
    }
}

enum MemoryOps {
    static func strides(from shape: [Int]) -> [Int] {
        let dim = shape.count
        
        if dim == 0 {
            return []
        }
        
        var str = [Int](repeating: 1, count: dim)
        for i in (0 ..< dim - 1).reversed() {
            str[i] = str[i + 1] * shape[i + 1]
        }
        return str
    }
    
    static func linearIndex(from index: [Int], shape: [Int]) -> Int {
        let strides = MemoryOps.strides(from: shape)
        return zip(index, strides).map(*).reduce(0, +)
    }
    
    static func index(from linearIndex: Int, shape: [Int]) -> [Int] {
        let strides = MemoryOps.strides(from: shape)
        return zip(shape, strides).map { dim, str in (linearIndex / str) % dim}
    }
    
    
    /// Retrieves the values of the source buffer and copies them into a newly allocated destination buffer if needed.
    ///
    /// - Parameters:
    ///   - slice: Index or range to access the vector at
    ///   - buffer: Vector
    ///   - shape: Shape of vector
    /// - Returns: Vector and boolean indicating, whether a new memory region has been allocated, and the shape of the result.
    static func get<Element>(slice: [Int?], of buffer: UnsafeMutableBufferPointer<Element>, with shape: [Int]) -> (UnsafeMutableBufferPointer<Element>, Bool, [Int]) {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        let nonNilIndices = slice.compactMap {$0}
        let strides = MemoryOps.strides(from: shape)
        
        if nonNilIndices.count == slice.count {
            // Simple offset into storage
            let offset = zip(nonNilIndices, strides).map(*).reduce(0, +)
            return (buffer.advanced(by: offset), false, Array(shape.dropFirst(nonNilIndices.count)))
        } else {
            let padded = slice + [Int?](repeating: nil, count: shape.count - slice.count)
            
            let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int? in
                let (index, dimSize) = el
                return index == nil ? dimSize : nil
            }
            let flattenedResultShape = resultShape.compactMap {$0}
            
            let resultCount = flattenedResultShape.reduce(1, *)
            let resultBuffer: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: resultCount)

            recursiveRead(source: buffer.immutable, destination: resultBuffer, srcIndex: padded, srcStrides: strides, srcShape: shape)
            
            return (resultBuffer, true, flattenedResultShape)
        }
        
    }
    
    /// Retrieves the values of the source buffer and copies them into a newly allocated destination buffer if needed.
    ///
    /// - Parameters:
    ///   - slice: Index or range to access the vector at
    ///   - buffer: Vector
    ///   - shape: Shape of vector
    /// - Returns: Vector and boolean indicating, whether a new memory region has been allocated, and the shape of the result.
    static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: UnsafeMutableBufferPointer<Element>, with shape: [Int]) -> (UnsafeMutableBufferPointer<Element>, Bool, [Int]) {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        let strides = MemoryOps.strides(from: shape)
        
        let padded = slice + [Range<Int>?](repeating: nil, count: shape.count - slice.count)
        
        let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int? in
            let (index, dimSize) = el
            return index.map {$0.count} ?? dimSize
        }
        let flattenedResultShape = resultShape.compactMap {$0}
        
        let resultCount = flattenedResultShape.reduce(1, *)
        let resultBuffer: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: resultCount)
        
        recursiveRead(source: buffer.immutable, destination: resultBuffer, srcIndex: padded, srcStrides: strides, srcShape: shape)
        
        return (resultBuffer, true, flattenedResultShape)
    }
    
    static func set<Element>(slice: [Int?], of buffer: UnsafeMutableBufferPointer<Element>, with dstShape: [Int], from source: UnsafeBufferPointer<Element>, with sourceShape: [Int]) {
        precondition(sourceShape.count == dstShape.count - slice.filter {$0 != nil}.count, "Shape of source must be equal to source of destination minus number of knowns in slice")
        
        let padded = slice + [Int?](repeating: nil, count: dstShape.count - slice.count)
        
        let dstStrides = MemoryOps.strides(from: dstShape)
        recursiveWrite(source: source, destination: buffer, dstIndex: padded, dstStrides: dstStrides, dstShape: dstShape)
    }
    
    static func set<Element>(slice: [Range<Int>?], of buffer: UnsafeMutableBufferPointer<Element>, with dstShape: [Int], from source: UnsafeBufferPointer<Element>, with sourceShape: [Int]) {
        precondition(sourceShape.count == dstShape.count - slice.filter {$0 != nil}.count, "Shape of source must be equal to source of destination minus number of knowns in slice")
        
        let padded = slice + [Range<Int>?](repeating: nil, count: dstShape.count - slice.count)
        let dstStrides = MemoryOps.strides(from: dstShape)
        
        recursiveWrite(source: source, destination: buffer, dstIndex: padded, dstStrides: dstStrides, dstShape: dstShape)
    }
}
