//
//  VRAM.swift
//  DL4S
//
//  Created by Palle Klewitz on 24.10.19.
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

#if canImport(Metal)
import Foundation
import Metal
import MetalPerformanceShaders


public struct VRAMAllocator: MemoryOperatorsType {
    public typealias RawBuffer = VRAMBuffer
    public typealias Device = GPU
    
    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> Buffer<Element, GPU> {
        let stride = MemoryLayout<Element>.stride
        guard let buffer = Device.device.makeBuffer(length: stride * capacity, options: .storageModeShared) else {
            fatalError("Could not allocate memory")
        }
        return Buffer<Element, GPU>(memory: VRAMBuffer(buffer: buffer, offset: 0, length: stride * capacity))
    }
    
    public static func allocateBuffer<Element>(withShape shape: [Int], type: Element.Type) -> ShapedBuffer<Element, GPU> {
        let count = shape.reduce(1, *)
        return ShapedBuffer(values: allocateBuffer(withCapacity: count, type: Element.self), shape: shape)
    }
    
    public static func free<Element>(_ buffer: Buffer<Element, GPU>) {
        // Noop, MTLBuffer is reference counted
        
    }
    
    public static func free<Element>(_ buffer: ShapedBuffer<Element, GPU>) {
        // Noop, MTLBuffer is reference counted
    }
    
    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, GPU>, count: Int) {
        GPU.synchronize()
        destination.memory.buffer.contents().initializeMemory(as: Element.self, from: source.pointer(capacity: count), count: count)
        GPU.synchronize()
    }
    
    public static func assign<Element>(from source: Buffer<Element, GPU>, to destination: Buffer<Element, GPU>, count: Int) {
        let buffer = GPU.currentCommandBuffer
        let encoder = buffer.makeBlitCommandEncoder()!
        encoder.copy(from: source.memory.buffer, sourceOffset: source.memory.offset, to: destination.memory.buffer, destinationOffset: destination.memory.offset, size: MemoryLayout<Element>.stride * count)
        encoder.endEncoding()
    }
    
    public static func assign<Element>(from source: Buffer<Element, GPU>, to destination: UnsafeMutableBufferPointer<Element>, count: Int) {
        GPU.synchronize()
        destination.assign(
            from: UnsafeBufferPointer(
                start: source.memory.buffer.contents()
                    .advanced(by: source.memory.offset)
                    .assumingMemoryBound(to: Element.self),
                count: count
            ),
            count: count
        )
    }
    
    public static func getValue<Element>(from source: Buffer<Element, GPU>) -> Element {
        GPU.synchronize()
        return source.memory.buffer.contents()
            .advanced(by: source.memory.offset)
            .assumingMemoryBound(to: Element.self).pointee
    }
    
    public static func getSize<Element>(of buffer: Buffer<Element, GPU>) -> Int {
        return buffer.memory.length / MemoryLayout<Element>.stride
    }
    
    public static func get<Element>(slice: [Int?], of buffer: Buffer<Element, GPU>, with shape: [Int]) -> (Buffer<Element, GPU>, Bool, [Int]) {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        // Prevent unneccessary copies when index ends with nil
        let slice = slice.reversed().drop(while: {$0 == nil}).reversed()
        
        let nonNilIndices = slice.compactMap {$0}
        let strides = CPUMemoryOperators.strides(from: shape)
        
        if nonNilIndices.count == slice.count {
            // Simple offset into storage
            let offset = zip(nonNilIndices, strides).map(*).reduce(0, +)
            let rawOffset = MemoryLayout<Element>.stride * offset
            let resultShape = Array(shape.dropFirst(nonNilIndices.count))
            let resultCount = resultShape.reduce(1, *)
            let rawCount = MemoryLayout<Element>.stride * resultCount
            
            let advanced = buffer.memory.advanced(by: rawOffset).prefix(rawCount)
            
            return (Buffer<Element, GPU>(memory: advanced), false, Array(shape.dropFirst(nonNilIndices.count)))
        } else {
            let padded = slice + [Int?](repeating: nil, count: shape.count - slice.count)
            
            let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int? in
                let (index, dimSize) = el
                return index == nil ? dimSize : nil
            }
            let flattenedResultShape = resultShape.compactMap {$0}
            
            let resultCount = flattenedResultShape.reduce(1, *)
            let resultBuffer = allocateBuffer(withCapacity: resultCount, type: Element.self)
            
            fatalError("TODO")
            //iterativeRead(source: buffer.memory.bindMemory(to: Element.self).immutable, destination: resultBuffer.memory.bindMemory(to: Element.self), srcIndex: padded, srcStrides: strides, srcShape: shape)
            
            return (resultBuffer, true, flattenedResultShape)
        }
        
    }
    
    public static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, GPU>, with shape: [Int]) -> (Buffer<Element, GPU>, Bool, [Int]) {
        fatalError("TODO")
    }
    
    public static func set<Element>(slice: [Int?], of buffer: Buffer<Element, GPU>, with dstShape: [Int], from source: Buffer<Element, GPU>, with sourceShape: [Int]) {
        fatalError("TODO")
    }
    
    public static func set<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, GPU>, with dstShape: [Int], from source: Buffer<Element, GPU>, with sourceShape: [Int]) {
        fatalError("TODO")
    }
    
    public static func advance<Element>(buffer: Buffer<Element, GPU>, by advancement: Int) -> Buffer<Element, GPU> {
        Buffer(memory: buffer.memory.advanced(by: MemoryLayout<Element>.stride * advancement))
    }
    
    public static func setPointee<Element>(of buffer: Buffer<Element, GPU>, to newValue: Element) {
        fatalError("TODO")
    }
}


public struct VRAMBuffer: Hashable {
    public static func == (lhs: VRAMBuffer, rhs: VRAMBuffer) -> Bool {
        return lhs.buffer.hash == rhs.buffer.hash && lhs.offset == rhs.offset && lhs.length == rhs.length
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(buffer.hash)
        hasher.combine(offset)
    }
    
    var buffer: MTLBuffer
    var offset: Int
    var length: Int
    
    func advanced(by offset: Int) -> VRAMBuffer {
        VRAMBuffer(buffer: buffer, offset: self.offset + offset, length: length - offset)
    }
    
    func prefix(_ length: Int) -> VRAMBuffer {
        precondition(length <= self.length, "Cannot create prefix of length \(length) on buffer of length \(self.length)")
        return VRAMBuffer(buffer: buffer, offset: self.offset + offset, length: length)
    }
}


#endif
