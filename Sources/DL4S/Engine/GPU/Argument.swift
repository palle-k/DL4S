//
//  Argument.swift
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

protocol GPUArgument {
    var slotCount: Int { get }
    func add(to encoder: MTLComputeCommandEncoder, from index: Int)
}

extension ShapedBuffer: GPUArgument where Device == GPU {
    var slotCount: Int {
        return 3
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        let valueBuffer = self.values.memory.buffer
        
        encoder.setBuffer(valueBuffer, offset: 0, index: index)
        encoder.setBytes([Int32(dim)], length: MemoryLayout<Int32>.size, index: index + 1)
        encoder.setBytes(shape.map(Int32.init), length: shape.count * MemoryLayout<Int32>.stride, index: index + 2)
    }
}

extension Buffer: GPUArgument where Device == GPU {
    var slotCount: Int {
        return 2
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        let valueBuffer = self.memory.buffer
        
        encoder.setBuffer(valueBuffer, offset: 0, index: index)
        encoder.setBytes([Int32(count)], length: MemoryLayout<Int32>.size, index: index + 1)
    }
}

extension Float: GPUArgument {
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBytes([self], length: MemoryLayout<Float>.size, index: index)
    }
}

extension Double: GPUArgument {
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBytes([self], length: MemoryLayout<Double>.size, index: index)
    }
}

extension Int32: GPUArgument {
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBytes([self], length: MemoryLayout<Int32>.size, index: index)
    }
}

extension Int: GPUArgument {
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBytes([Int32(self)], length: MemoryLayout<Int32>.size, index: index)
    }
}

extension VRAMBuffer: GPUArgument {
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBuffer(buffer, offset: self.offset, index: index)
    }
}

struct ArrayValuesArgument<Element>: GPUArgument {
    var array: [Element]
    
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBytes(array, length: MemoryLayout<Element>.stride * array.count, index: index)
    }
}

extension Array where Element == Int {
    var valueArg: ArrayValuesArgument<Int32> {
        ArrayValuesArgument(array: self.map(Int32.init))
    }
}

#endif
