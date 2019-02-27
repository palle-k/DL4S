//
//  Allocator.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
//

import Foundation

extension Collection {
    func minIndex(by comparator: (Element, Element) throws -> Bool) rethrows -> Index? {
        var minIndex: Index? = nil
        var minValue: Element? = nil
        for index in indices {
            if let mv = minValue, try !comparator(mv, self[index]) {
                minIndex = index
                minValue = self[index]
            }
        }
        return minIndex
    }
}

enum Allocator {
    private static var maxCache: Int = 1_000_000_000 // 1GB
    private static var freeBuffers: [(UnsafeMutableRawPointer, Int)] = []
    private static var usedBuffers: [UnsafeMutableRawPointer: Int] = [:]
    
    private static var unusedMemory: Int {
        return freeBuffers.map {$1}.reduce(0, +)
    }
    
    static func allocate<T>(count: Int) -> UnsafeMutablePointer<T> {
        let stride = MemoryLayout<T>.stride
        let alignment = MemoryLayout<T>.alignment
        
        let byteCount = count * stride
        
        if let idx = freeBuffers.filter({$0.1 > byteCount}).minIndex(by: {$0.1 < $1.1}) {
            let (reusedBuffer, capacity) = freeBuffers.remove(at: idx)
            usedBuffers[reusedBuffer] = capacity
            return reusedBuffer.assumingMemoryBound(to: T.self)
        } else {
            let newBuffer = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: alignment)
            usedBuffers[newBuffer] = byteCount
            return newBuffer.assumingMemoryBound(to: T.self)
        }
    }
    
    static func free<T>(_ buffer: UnsafeMutablePointer<T>) {
        let rawBuffer = UnsafeMutableRawPointer(buffer)
        if let capacity = usedBuffers[rawBuffer] {
            if unusedMemory + capacity > maxCache {
                rawBuffer.deallocate()
            } else {
                freeBuffers.append((rawBuffer, capacity))
            }
        } else {
            rawBuffer.deallocate()
        }
    }
}
