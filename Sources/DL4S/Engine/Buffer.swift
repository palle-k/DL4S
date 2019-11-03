//
//  Buffer.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
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


/// A buffer that holds a region of memory with a given length
public struct Buffer<Element, Device: DeviceType>: Hashable {
    let memory: Device.Memory.RawBuffer
    
    var count: Int {
        return Device.Memory.getSize(of: self)
    }
    
    var pointee: Element {
        get {
            return Device.Memory.getValue(from: self)
        }
        
        nonmutating set (newValue) {
            Device.Memory.setPointee(of: self, to: newValue)
        }
    }
    
    func advanced(by offset: Int) -> Buffer<Element, Device> {
        return Device.Memory.advance(buffer: self, by: offset)
    }
    
    subscript(index: Int) -> Element {
        get {
            return advanced(by: index).pointee
        }
        nonmutating set {
            advanced(by: index).pointee = newValue
        }
    }
}

extension Buffer {
    var array: [Element] {
        let b = UnsafeMutableBufferPointer<Element>.allocate(capacity: self.count)
        defer {
            b.deallocate()
        }
        Device.Memory.assign(from: self, to: b, count: self.count)
        return Array(b)
    }
}

extension Buffer: CustomLeafReflectable {
    public var customMirror: Mirror {
        let b = UnsafeMutableBufferPointer<Element>.allocate(capacity: self.count)
        defer {
            b.deallocate()
        }
        Device.Memory.assign(from: self, to: b, count: self.count)
        let a = Array(b)
        return Mirror(self, unlabeledChildren: a, displayStyle: .collection)
    }
}

/// A buffer that holds a region of memory with a given shape
public struct ShapedBuffer<Element, Device: DeviceType>: Hashable {
    var shape: [Int]
    var values: Buffer<Element, Device>
    
    var dim: Int {
        return shape.count
    }
    
    var count: Int {
        return values.count
    }
    
    init(values: Buffer<Element, Device>, shape: [Int]) {
        self.shape = shape
        self.values = values
    }
    
    func reshaped(to shape: [Int]) -> ShapedBuffer<Element, Device> {
        precondition(shape.reduce(1, *) == self.shape.reduce(1, *))
        
        return ShapedBuffer(values: values, shape: shape)
    }
}

extension Buffer: CustomStringConvertible {
    public var description: String {
        return "Buffer(\(generateDescription()))"
    }
    
    func generateDescription() -> String {
        return "[\(array.map {"\($0)"}.joined(separator: ", "))]"
    }
}

extension ShapedBuffer: CustomStringConvertible {
    public var description: String {
        return generateDescription()
    }
    
    private func formatElement(_ element: Element) -> String {
        if let f = element as? Float {
            return f.format(maxDecimals: 3)
        } else if let d = element as? Double {
            return d.format(maxDecimals: 3)
        } else {
            return "\(element)"
        }
    }
    
    func generateDescription() -> String {
        if dim == 0 {
            return "\(formatElement(values.pointee))"
        } else if dim == 1 {
            let dim = self.shape[0]
            return "[\((0 ..< dim).map {"\(formatElement(values[$0]))"}.joined(separator: ", "))]"
        } else {
            let firstDim = shape.first!
            let restDim = Array(shape.dropFirst())
            
            let stride = restDim.reduce(1, *)
            
            let slices = (0 ..< firstDim).map {
                ShapedBuffer(values: values.advanced(by: stride * $0), shape: restDim).generateDescription()
            }
            
            return "[\(slices.joined(separator: ",\n").replacingOccurrences(of: "\n", with: "\n "))]"
        }
    }
}
