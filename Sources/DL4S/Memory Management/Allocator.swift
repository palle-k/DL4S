//
//  Allocator.swift
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

extension UnsafeMutableRawBufferPointer: Hashable {
    public static func == (lhs: UnsafeMutableRawBufferPointer, rhs: UnsafeMutableRawBufferPointer) -> Bool {
        return lhs.baseAddress == rhs.baseAddress && lhs.count == rhs.count
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(String(describing: self.baseAddress!))
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
        //memcpy(self.baseAddress!, ptr.baseAddress!, count * MemoryLayout<Element>.stride)
        self.baseAddress!.assign(from: ptr.baseAddress!, count: count)
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

enum CPUAllocator {
    private static let maxCache: Int = 1_000_000_000 // 1GB
    private static var usedCache: Int = 0
    
    private static var freeBuffers: [Int: [UnsafeMutableRawBufferPointer]] = [:]
    
    private static var unusedCache: Int {
        return maxCache - usedCache
    }
    
    static func allocate<T>(count: Int) -> UnsafeMutableBufferPointer<T> {
        let stride = MemoryLayout<T>.stride
        let alignment = max(MemoryLayout<T>.alignment, 64) // Some vDSP routines perform better with 16 byte alignment
        
        let byteCount = count * stride
        usedCache += byteCount
        
        if let ptr = freeBuffers[byteCount]?.first {
            freeBuffers[byteCount]?.removeFirst()
            if freeBuffers[byteCount]?.isEmpty == true {
                freeBuffers.removeValue(forKey: byteCount)
            }
            return ptr.bindMemory(to: T.self)
        } else {
            let ptr = UnsafeMutableRawBufferPointer.allocate(byteCount: byteCount, alignment: alignment)
            return ptr.bindMemory(to: T.self)
        }
    }
    
    static func free<T>(_ buffer: UnsafeMutableBufferPointer<T>) {
        let rawBuffer = UnsafeMutableRawBufferPointer(buffer)
        let capacity = rawBuffer.count
        
        rawBuffer.deallocate()
        usedCache -= capacity
        
//        if usedCache >= maxCache {
//            rawBuffer.deallocate()
//            usedCache -= capacity
//        } else {
//            freeBuffers[capacity, default: []].append(rawBuffer)
//        }
    }
}
