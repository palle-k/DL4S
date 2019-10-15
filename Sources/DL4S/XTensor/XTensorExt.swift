//
//  XTensorExt.swift
//  DL4S
//
//  Created by Palle Klewitz on 04.10.19.
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


extension XTensor: CustomStringConvertible, CustomDebugStringConvertible {
    public var description: String {
        values.description
    }
    
    public var debugDescription: String {
        return """
        XTensor<\(Element.self), \(Device.self)>(
            \(values.description.replacingOccurrences(of: "\n", with: "\n    ")),
            context: \(self.context as Any? ?? "nil" as Any)
        )
        """
    }
}

extension XTensor: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.init([Element.init(value)], shape: [])
    }
}

extension XTensor: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.init([Element.init(value)], shape: [])
    }
}

public extension XTensor {
    init(_ value: Element) {
        self.init([value], shape: [])
    }
}

public extension XTensor {
    init(_ source: Tensor<Element, Device>) {
        let copy = Device.Memory.allocateBuffer(withShape: source.shape, type: Element.self)
        Device.Memory.assign(from: source.values, to: copy.values, count: source.count)
        self.init(using: copy, context: nil)
    }
    
    var compatibilityTensor: Tensor<Element, Device> {
        let copy = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Memory.assign(from: values.values, to: copy.values, count: count)
        return Tensor<Element, Device>(values: copy.values, gradient: nil, shape: shape, context: nil)
    }
}

public extension XTensor {
    var item: Element {
        Device.Memory.getValue(from: values.values)
    }
}

public extension XTensor {
    init(_ v: [[Element]], requiresGradient: Bool = false) {
        self.init(Array(v.joined()), shape: [v.count, v.first?.count ?? 0], requiresGradient: requiresGradient)
    }
    
    init(_ v: [[[Element]]], requiresGradient: Bool = false) {
        self.init(
            Array(v.joined().joined()),
            shape: [v.count, v.first?.count ?? 0, v.first?.first?.count ?? 0],
            requiresGradient: requiresGradient
        )
    }
    
    init(_ v: [[[[Element]]]], requiresGradient: Bool = false) {
        self.init(
            Array(v.joined().joined().joined()),
            shape: [
                v.count,
                v.first?.count ?? 0,
                v.first?.first?.count ?? 0,
                v.first?.first?.first?.count ?? 0
            ],
            requiresGradient: requiresGradient
        )
    }
    
    init(_ v: [[[[[Element]]]]], requiresGradient: Bool = false) {
        self.init(
            Array(v.joined().joined().joined().joined()),
            shape: [
                v.count,
                v.first?.count ?? 0,
                v.first?.first?.count ?? 0,
                v.first?.first?.first?.count ?? 0,
                v.first?.first?.first?.first?.count ?? 0
            ],
            requiresGradient: requiresGradient
        )
    }
}

public extension XTensor where Element: RandomizableType {
    init(xavierNormalWithShape shape: [Int], requiresGradient: Bool = false) {
        precondition(shape.count == 2, "Shape must be 2-dimensional")
        self.init(repeating: 0, shape: shape, requiresGradient: requiresGradient)
        
        let tmp = Tensor<Element, Device>(repeating: 0, shape: shape)
        
        Random.fillNormal(tmp, mean: 0, stdev: 2 / Element(shape[0]).sqrt())
        Device.Memory.assign(from: tmp.values, to: self.values.values, count: tmp.count)
    }
}

extension XTensor: Codable where Element: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        requiresGradient = try container.decode(Bool.self, forKey: .requiresGradient)
        shape = try container.decode([Int].self, forKey: .shape)
        let buffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        handle = XTensorHandle(values: buffer.values)
        let data = try container.decode(Data.self, forKey: .data)
        data.withUnsafeBytes { bytes in
            Device.Memory.assign(from: bytes.bindMemory(to: Element.self), to: buffer.values, count: count)
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: count)
        Device.Memory.assign(from: values.values, to: buffer, count: count)
        let data = Data(buffer: buffer)
        
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(requiresGradient, forKey: .requiresGradient)
        try container.encode(data, forKey: .data)
        try container.encode(shape, forKey: .shape)
    }
    
    private enum CodingKeys: String, CodingKey {
        case requiresGradient
        case data
        case shape
    }
}

public extension XTensor {
    var elements: [Element] {
        var array = [Element](repeating: 0, count: count)
        array.withUnsafeMutableBufferPointer { pointer in
            Device.Memory.assign(from: values.values, to: pointer, count: count)
        }
        return array
    }
}
