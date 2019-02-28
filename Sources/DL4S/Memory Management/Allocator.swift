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

extension UnsafeMutableRawBufferPointer: Hashable {
    public static func == (lhs: UnsafeMutableRawBufferPointer, rhs: UnsafeMutableRawBufferPointer) -> Bool {
        return lhs.baseAddress == rhs.baseAddress && lhs.count == rhs.count
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(bytes: UnsafeRawBufferPointer(self))
    }
}

extension UnsafeMutableBufferPointer {
    var immutable: UnsafeBufferPointer<Element> {
        return UnsafeBufferPointer(self)
    }
    
    var pointee: Element {
        get {
            precondition(count > 0, "Out of bounds access")
            return self[0]
        }
        nonmutating set {
            precondition(count > 0, "Out of bounds access")
            self[0] = newValue
        }
    }
    
    func advanced(by offset: Int) -> UnsafeMutableBufferPointer<Element> {
        precondition(offset < count, "Out of bounds access")
        return UnsafeMutableBufferPointer(start: baseAddress!.advanced(by: offset), count: count - offset)
    }
    
    func assign(from ptr: UnsafeBufferPointer<Element>, count: Int) {
        precondition(ptr.count >= count, "Out of bounds access")
        precondition(self.count >= count, "Out of bounds write")
        memcpy(self.baseAddress!, ptr.baseAddress!, count)
    }
    
    func pointer(capacity: Int) -> UnsafeMutablePointer<Element> {
        precondition(capacity <= count, "Out of bounds access")
        return baseAddress!
    }
}

extension UnsafeBufferPointer {
    var pointee: Element {
        get {
            precondition(count > 0, "Out of bounds memory access")
            return self[0]
        }
    }
    
    func advanced(by offset: Int) -> UnsafeBufferPointer<Element> {
        precondition(offset < count, "Out of bounds access")
        return UnsafeBufferPointer(start: baseAddress!.advanced(by: offset), count: count - offset)
    }
    
    func pointer(capacity: Int) -> UnsafePointer<Element> {
        precondition(capacity <= count, "Out of bounds access")
        return baseAddress!
    }
}

extension UnsafeMutableBufferPointer: Equatable {
    public static func == (lhs: UnsafeMutableBufferPointer<Element>, rhs: UnsafeMutableBufferPointer<Element>) -> Bool {
        return lhs.baseAddress == rhs.baseAddress && lhs.count == rhs.count
    }
}

enum Allocator {
    private static var maxCache: Int = 1_000_000_000 // 1GB
    private static var freeBuffers: [(UnsafeMutableRawBufferPointer, Int)] = []
    private static var usedBuffers: [UnsafeMutableRawBufferPointer: Int] = [:]
    
    private static var unusedMemory: Int {
        return freeBuffers.map {$1}.reduce(0, +)
    }
    
    static func allocate<T>(count: Int) -> UnsafeMutableBufferPointer<T> {
        let stride = MemoryLayout<T>.stride
        let alignment = MemoryLayout<T>.alignment
        
        let byteCount = count * stride
        
//        if let idx = freeBuffers.filter({$0.1 > byteCount}).minIndex(by: {$0.1 < $1.1}) {
//            let (reusedBuffer, capacity) = freeBuffers.remove(at: idx)
//            usedBuffers[reusedBuffer] = capacity
//            return reusedBuffer.bindMemory(to: T.self)
//        } else {
//            let newBuffer = UnsafeMutableRawBufferPointer.allocate(byteCount: byteCount, alignment: alignment)
//            usedBuffers[newBuffer] = byteCount
//            return newBuffer.bindMemory(to: T.self)
//        }
        let ptr = UnsafeMutableRawBufferPointer.allocate(byteCount: byteCount, alignment: alignment)
        return ptr.bindMemory(to: T.self)
    }
    
    static func free<T>(_ buffer: UnsafeMutableBufferPointer<T>) {
        let rawBuffer = UnsafeMutableRawBufferPointer(buffer)
        rawBuffer.deallocate()
//        if let capacity = usedBuffers[rawBuffer] {
//            if unusedMemory + capacity > maxCache {
//                rawBuffer.deallocate()
//            } else {
//                freeBuffers.append((rawBuffer, capacity))
//            }
//        } else {
//            rawBuffer.deallocate()
//        }
    }
}
