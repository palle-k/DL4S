//
//  MemoryOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
//

import Foundation


private func recursiveCopy<Element>(
    source: UnsafePointer<Element>,
    destination: UnsafeMutablePointer<Element>,
    sourceShape: [Int],
    sourceStrides: [Int],
    destinationStrides: [Int],
    indices: [Int?]
) {
    guard
        let sStride = sourceStrides.first,
        let dStride = destinationStrides.first,
        let idx = indices.first,
        let dimSize = sourceShape.first
        else {
            destination.pointee = source.pointee
            return
    }
    
    if let idx = idx {
        let srcStart = source.advanced(by: idx * sStride)
        let dstStart = destination.advanced(by: idx * dStride)
        
        recursiveCopy(
            source: srcStart,
            destination: dstStart,
            sourceShape: Array(sourceShape.dropFirst()),
            sourceStrides: Array(sourceStrides.dropFirst()),
            destinationStrides: Array(destinationStrides.dropFirst()),
            indices: Array(indices.dropFirst())
        )
    } else if indices.allSatisfy({$0 == nil}) {
        // Rest of the indices are nil -> perform optimized copy
        
        let count = dimSize * sStride
        destination.assign(from: source, count: count)
        
    } else {
        for i in 0 ..< dimSize {
            let srcStart = source.advanced(by: i * sStride)
            let dstStart = destination.advanced(by: i * dStride)
            
            recursiveCopy(
                source: srcStart,
                destination: dstStart,
                sourceShape: Array(sourceShape.dropFirst()),
                sourceStrides: Array(sourceStrides.dropFirst()),
                destinationStrides: Array(destinationStrides.dropFirst()),
                indices: Array(indices.dropFirst())
            )
        }
    }
}

private func recursiveCopy<Element>(
    source: UnsafePointer<Element>,
    destination: UnsafeMutablePointer<Element>,
    sourceShape: [Int],
    sourceStrides: [Int],
    destinationStrides: [Int],
    indices: [Range<Int>?]
) {
    guard
        let sStride = sourceStrides.first,
        let dStride = destinationStrides.first,
        let idx = indices.first,
        let dimSize = sourceShape.first
    else {
            destination.pointee = source.pointee
            return
    }
    
    if zip(indices, sourceShape).allSatisfy({$0 == nil || ($0?.lowerBound == 0 && $0?.upperBound == $1)}) {
        // Rest of the indices are nil -> perform optimized copy
        
        let count = dimSize * sStride
        destination.assign(from: source, count: count)
        
    } else {
        let start = idx?.lowerBound ?? 0
        let end = idx?.upperBound ?? dimSize
        
        if zip(indices.dropFirst(), sourceShape.dropFirst()).allSatisfy({$0 == nil || ($0?.lowerBound == 0 && $0?.upperBound == $1)}) {
            let srcStart = source.advanced(by: start * sStride)
            let dstStart = destination.advanced(by: start * dStride)
            
            let count = sStride * (end - start)
            
            dstStart.assign(from: srcStart, count: count)
            return
        }
        
        for i in start ..< end {
            let srcStart = source.advanced(by: i * sStride)
            let dstStart = destination.advanced(by: i * dStride)
            
            recursiveCopy(
                source: srcStart,
                destination: dstStart,
                sourceShape: Array(sourceShape.dropFirst()),
                sourceStrides: Array(sourceStrides.dropFirst()),
                destinationStrides: Array(destinationStrides.dropFirst()),
                indices: Array(indices.dropFirst())
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
    
    
    /// Retrieves the values of the source buffer and copies them into a newly allocated destination buffer if needed.
    ///
    /// - Parameters:
    ///   - slice: Index or range to access the vector at
    ///   - buffer: Vector
    ///   - shape: Shape of vector
    /// - Returns: Vector and boolean indicating, whether a new memory region has been allocated, and the shape of the result.
    static func get<Element>(slice: [Int?], of buffer: UnsafeMutablePointer<Element>, with shape: [Int]) -> (UnsafeMutablePointer<Element>, Bool, [Int]) {
        
        
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
            let resultBuffer = UnsafeMutablePointer<Element>.allocate(capacity: resultCount)
            
            let dstStrides = MemoryOps.strides(from: resultShape.map {$0 ?? 1})
            
            recursiveCopy(
                source: buffer,
                destination: resultBuffer,
                sourceShape: shape,
                sourceStrides: strides,
                destinationStrides: dstStrides,
                indices: padded
            )
            
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
    static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: UnsafeMutablePointer<Element>, with shape: [Int]) -> (UnsafeMutablePointer<Element>, Bool, [Int]) {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        let strides = MemoryOps.strides(from: shape)
        
        let padded = slice + [Range<Int>?](repeating: nil, count: shape.count - slice.count)
        
        let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int? in
            let (index, dimSize) = el
            return index.map {$0.count} ?? dimSize
        }
        let flattenedResultShape = resultShape.compactMap {$0}
        
        let resultCount = flattenedResultShape.reduce(1, *)
        let resultBuffer = UnsafeMutablePointer<Element>.allocate(capacity: resultCount)
        
        let dstStrides = MemoryOps.strides(from: resultShape.map {$0 ?? 1})
        
        recursiveCopy(
            source: buffer,
            destination: resultBuffer,
            sourceShape: shape,
            sourceStrides: strides,
            destinationStrides: dstStrides,
            indices: padded
        )
        
        return (resultBuffer, true, flattenedResultShape)
    }
    
    static func set<Element>(slice: [Int?], of buffer: UnsafeMutablePointer<Element>, with dstShape: [Int], from source: UnsafePointer<Element>, with sourceShape: [Int]) {
        precondition(sourceShape.count == dstShape.count - slice.filter {$0 != nil}.count, "Shape of source must be equal to source of destination minus number of knowns in slice")
        
        let padded = slice + [Int?](repeating: nil, count: dstShape.count - slice.count)
        
        let nonflatSrcShape = zip(padded, sourceShape).enumerated().map { idx, el -> Int? in
            let (index, dimSize) = el
            return index == nil ? dimSize : nil
        }
        
        let sourceStrides = MemoryOps.strides(from: nonflatSrcShape.map {$0 ?? 1})
        let dstStrides = MemoryOps.strides(from: dstShape)
        
        recursiveCopy(
            source: source,
            destination: buffer,
            sourceShape: sourceShape,
            sourceStrides: sourceStrides,
            destinationStrides: dstStrides,
            indices: padded
        )
    }
    
    static func set<Element>(slice: [Range<Int>?], of buffer: UnsafeMutablePointer<Element>, with dstShape: [Int], from source: UnsafePointer<Element>, with sourceShape: [Int]) {
        precondition(sourceShape.count == dstShape.count - slice.filter {$0 != nil}.count, "Shape of source must be equal to source of destination minus number of knowns in slice")
        
        let padded = slice + [Range<Int>?](repeating: nil, count: dstShape.count - slice.count)
        
        let nonflatSrcShape = zip(padded, sourceShape).enumerated().map { idx, el -> Int? in
            let (index, dimSize) = el
            return index.map {$0.count} ?? dimSize
        }
        
        let sourceStrides = MemoryOps.strides(from: nonflatSrcShape.map {$0 ?? 1})
        let dstStrides = MemoryOps.strides(from: dstShape)
        
        recursiveCopy(
            source: source,
            destination: buffer,
            sourceShape: sourceShape,
            sourceStrides: sourceStrides,
            destinationStrides: dstStrides,
            indices: padded
        )
    }
    
    
}
