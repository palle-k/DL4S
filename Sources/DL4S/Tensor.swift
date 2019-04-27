//
//  Tensor.swift
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

public class Tensor<Element: NumericType, Device: DeviceType>: ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral, Codable {
    public typealias FloatLiteralType = Double
    public typealias IntegerLiteralType = Int32
    
    public let shape: [Int]
    
    public var count: Int {
        return shape.reduce(1, *)
    }
    
    public var dim: Int {
        return shape.count
    }
    
    @available(*, deprecated: 0.0, message: "Device specific, do not use")
    var strides: [Int] {
        return CPU.Memory.strides(from: shape)
    }
    
    public var tag: String? = nil
    
    public var requiresGradient: Bool {
        get {
            return gradient != nil
        }
        set (newValue) {
            guard requiresGradient != newValue else {
                return
            }
            if newValue {
                let gradient: Buffer<Element, Device> = Device.Memory.allocateBuffer(withCapacity: count, type: Element.self)
                Device.Engine.fill(value: 0, result: gradient, count: count)
                self.gradient = gradient
            } else {
                gradient.map(Device.Memory.free)
            }
        }
    }
    
    var values: Buffer<Element, Device>
    
    var gradient: Buffer<Element, Device>?
    
    var context: AnyTensorOperation<Element, Device>?
    
    // Specifies, whether the vector is the owner of the value pointers
    var parent: Tensor<Element, Device>?
    
    public init(repeating value: Element, shape: [Int], requiresGradient: Bool = false) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = Device.Memory.allocateBuffer(withCapacity: count, type: Element.self)
        self.gradient = nil
        self.context = nil
        self.parent = nil
        self.requiresGradient = requiresGradient
        
        Device.Engine.fill(value: value, result: values, count: count)
    }
    
    public convenience init(repeating value: Element, shape: Int..., requiresGradient: Bool = false) {
        self.init(repeating: value, shape: shape, requiresGradient: requiresGradient)
    }
    
    public init(_ elements: [Element], shape: [Int], requiresGradient: Bool = false) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = Device.Memory.allocateBuffer(withCapacity: count, type: Element.self)
        self.gradient = nil
        self.context = nil
        self.parent = nil
        
        elements.withUnsafeBufferPointer { ptr in
            Device.Memory.assign(from: ptr, to: self.values, count: self.count)
        }
        
        self.requiresGradient = requiresGradient
    }
    
    public convenience init(_ elements: [Element], shape: Int..., requiresGradient: Bool = false) {
        self.init(elements, shape: shape, requiresGradient: requiresGradient)
    }
    
    public convenience init(_ array: [Element], requiresGradient: Bool = false) {
        self.init(array, shape: array.count, requiresGradient: requiresGradient)
    }
    
    public convenience init(_ array: [[Element]], requiresGradient: Bool = false) {
        precondition(array.allSatisfy {$0.count == array[0].count}, "Invalid shape, all rows of array must be of same length.")
        self.init(Array(array.joined()), shape: array.count, array.first?.count ?? 0, requiresGradient: requiresGradient)
    }
    
    public convenience init(_ array: [[[Element]]], requiresGradient: Bool = false) {
        precondition(array.allSatisfy {
            $0.count == array[0].count && $0.allSatisfy {
                $0.count == array[0][0].count
            }
        }, "Invalid shape, all rows of array must be of same length.")
        self.init(
            Array(array.joined().joined()),
            shape: array.count,
            array.first?.count ?? 0,
            array.first?.first?.count ?? 0,
            requiresGradient: requiresGradient
        )
    }
    
    public convenience init(_ array: [[[[Element]]]], requiresGradient: Bool = false) {
        precondition(array.allSatisfy {
            $0.count == array[0].count && $0.allSatisfy {
                $0.count == array[0][0].count && $0.allSatisfy {
                    $0.count == array[0][0][0].count
                }
            }
        }, "Invalid shape, all rows of array must be of same length.")
        self.init(
            Array(array.joined().joined().joined()),
            shape: array.count,
            array.first?.count ?? 0,
            array.first?.first?.count ?? 0,
            array.first?.first?.first?.count ?? 0,
            requiresGradient: requiresGradient
        )
    }
    
    public convenience init(_ array: [[[[[Element]]]]], requiresGradient: Bool = false) {
        precondition(array.allSatisfy {
            $0.count == array[0].count && $0.allSatisfy {
                $0.count == array[0][0].count && $0.allSatisfy {
                    $0.count == array[0][0][0].count && $0.allSatisfy {
                        $0.count == array[0][0][0][0].count
                    }
                }
            }
        }, "Invalid shape, all rows of array must be of same length.")
        self.init(
            Array(array.joined().joined().joined().joined()),
            shape: array.count,
            array.first?.count ?? 0,
            array.first?.first?.count ?? 0,
            array.first?.first?.first?.count ?? 0,
            array.first?.first?.first?.first?.count ?? 0,
            requiresGradient: requiresGradient
        )
    }
    
    public init<SourceDevice>(_ tensor: Tensor<Element, SourceDevice>) {
        self.shape = tensor.shape
        self.parent = nil
        self.context = nil
        self.tag = tensor.tag
        
        self.values = Device.Memory.allocateBuffer(withCapacity: tensor.count, type: Element.self)
        
        let tmp = UnsafeMutableBufferPointer<Element>.allocate(capacity: tensor.count)
        SourceDevice.Memory.assign(from: tensor.values, to: tmp, count: tensor.count)
        Device.Memory.assign(from: tmp.immutable, to: self.values, count: tensor.count)
        
        if let gradient = tensor.gradient {
            self.gradient = Device.Memory.allocateBuffer(withCapacity: tensor.count, type: Element.self)
            SourceDevice.Memory.assign(from: gradient, to: tmp, count: tensor.count)
            Device.Memory.assign(from: tmp.immutable, to: self.gradient!, count: tensor.count)
        } else {
            self.gradient = nil
        }
        tmp.deallocate()
    }
    
    init(values: Buffer<Element, Device>, gradient: Buffer<Element, Device>?, shape: [Int], parent: Tensor<Element, Device>? = nil, context: AnyTensorOperation<Element, Device>?) {
        self.values = values
        self.gradient = gradient
        self.shape = shape
        self.parent = parent
        self.context = context
    }
    
    init(values: Buffer<Element, Device>, shape: [Int], parent: Tensor<Element, Device>? = nil, context: AnyTensorOperation<Element, Device>?) {
        self.values = values
        self.gradient = nil
        self.shape = shape
        self.parent = parent
        self.context = context
        
        Device.Engine.fill(value: 0, result: self.gradient!, count: count)
        self.requiresGradient = context?.sourceTensors.contains(where: {$0.requiresGradient}) ?? false
    }
    
    public init(_ value: Element, requiresGradient: Bool = false) {
        self.shape = []
        self.values = Device.Memory.allocateBuffer(withCapacity: 1, type: Element.self)
        self.gradient = nil
        self.context = nil
        self.parent = nil
        
        Device.Engine.fill(value: value, result: self.values, count: 1)
        self.requiresGradient = requiresGradient
    }
    
    public required init(floatLiteral value: Double) {
        self.shape = []
        self.values = Device.Memory.allocateBuffer(withCapacity: 1, type: Element.self)
        self.gradient = nil
        self.context = nil
        self.parent = nil
        
        Device.Engine.fill(value: Element(value), result: self.values, count: 1)
    }
    
    public required init(integerLiteral value: Int32) {
        self.shape = []
        self.values = Device.Memory.allocateBuffer(withCapacity: 1, type: Element.self)
        self.gradient = nil
        self.context = nil
        self.parent = nil
        
        Device.Engine.fill(value: Element(value), result: self.values, count: 1)
    }
    
    init(shape: [Int], parent: Tensor<Element, Device>?, context: AnyTensorOperation<Element, Device>?) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = Device.Memory.allocateBuffer(withCapacity: count, type: Element.self)
        self.gradient = nil
        self.context = context
        self.parent = parent
        
        self.requiresGradient = context?.sourceTensors.contains(where: {$0.requiresGradient}) ?? false
    }
    
    private enum CodingKeys: String, CodingKey {
        case values
        case shape
        case gradient
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let shape = try container.decode([Int].self, forKey: .shape)
        let values = try container.decode(Data.self, forKey: .values)
        
        self.shape = shape
        self.values = Device.Memory.allocateBuffer(withCapacity: shape.reduce(1, *), type: Element.self)
        self.context = nil
        self.parent = nil
        
        if let gradient = try container.decode(Data?.self, forKey: .gradient) {
            self.gradient = Device.Memory.allocateBuffer(withCapacity: shape.reduce(1, *), type: Element.self)
            gradient.withUnsafeBytes { ptr in
                Device.Memory.assign(from: ptr.bindMemory(to: Element.self), to: self.gradient!, count: shape.reduce(1, *))
            }
        } else {
            self.gradient = nil
        }

        values.withUnsafeBytes { ptr in
            Device.Memory.assign(from: ptr.bindMemory(to: Element.self), to: self.values, count: shape.reduce(1, *))
        }
    }
    
    deinit {
        if parent == nil {
            Device.Memory.free(values)
            gradient.map(Device.Memory.free)
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        var valueContainer = Data(repeating: 0, count: MemoryLayout<Element>.stride * self.count)
        valueContainer.withUnsafeMutableBytes { ptr in
            Device.Memory.assign(from: self.values, to: ptr.bindMemory(to: Element.self), count: self.count)
        }
        
        try container.encode(shape, forKey: .shape)
        try container.encode(valueContainer, forKey: .values)
        
        if let gradient = self.gradient {
            var gradientContainer = Data(repeating: 0, count: MemoryLayout<Element>.stride * self.count)
            gradientContainer.withUnsafeMutableBytes { ptr in
                Device.Memory.assign(from: gradient, to: ptr.bindMemory(to: Element.self), count: self.count)
            }
            try container.encode(gradientContainer, forKey: .gradient)
        } else {
            try container.encode([Element]?.none, forKey: .gradient)
        }
    }
    
    func resignOwnership(to parent: Tensor<Element, Device>) {
        self.parent = parent
    }
    
    func ensureOwnership() {
        if parent == nil {
            return
        }
        
        let existingValues = self.values
        let existingGradient = self.gradient
        
        self.values = Device.Memory.allocateBuffer(withCapacity: count, type: Element.self)
        Device.Memory.assign(from: existingValues, to: self.values, count: count)
        
        if let gradient = existingGradient {
            self.gradient = Device.Memory.allocateBuffer(withCapacity: count, type: Element.self)
            Device.Memory.assign(from: gradient, to: self.gradient!, count: count)
        }
        
        self.parent = nil
    }
    
    public func zeroGradient() {
        guard let gradient = self.gradient else {
            return
        }
        Device.Engine.fill(value: 0, result: gradient, count: count)
        context?.zeroGradient()
    }
    
    func _backwards() {
        context?.fillSourceGradients(fromResultGradients: self)
    }
    
    public func backwards() {
        guard let gradient = self.gradient else {
            return
        }
        Device.Engine.fill(value: 1, result: gradient, count: count)
        
        let ordering = Tensor.operationOrder(from: self)
        
        for tensor in ordering.reversed() {
            tensor._backwards()
        }
    }
    
    /// Optains the element of the tensor
    /// Only applicable to zero-dimensional tensors (scalars)
    /// The item is not further tracked with automatic differentiation.
    ///
    /// To obtain all the elements of non-scalar tensors, use `tensor.flattenedArray`.
    public var item: Element {
        if dim == 0 {
            return values.pointee
        } else {
            fatalError("Cannot obtain item of vector with dimensionality other than 0")
        }
    }
    
    /// Obtains the gradient of the tensor
    /// Only applicable to zero-dimensional tensors (scalars)
    /// The gradient must have been computed previously using `result.backwards()` on
    /// the result tensor.
    ///
    /// To obtain all the gradient values of non-scalar tensors, use `tensor.flattenedGradientArray`.
    public var gradientItem: Element? {
        if dim == 0 {
            return gradient?.pointee
        } else {
            fatalError("Cannot obtain item of vector with dimensionality other than 0")
        }
    }
    
    public func detached() -> Tensor<Element, Device> {
        let copy = Tensor(values: values, gradient: gradient, shape: shape, parent: self, context: nil)
        copy.ensureOwnership()
        return copy
    }
    
    static func sameIdentity(_ vector1: Tensor<Element, Device>, _ vector2: Tensor<Element, Device>) -> Bool {
        return vector1.values == vector2.values && vector1.gradient == vector2.gradient
    }
    
    static func sameIdentity(_ vector1: Tensor<Element, Device>?, _ vector2: Tensor<Element, Device>?) -> Bool {
        if let vector1 = vector1, let vector2 = vector2 {
            return sameIdentity(vector1, vector2)
        } else {
            return false
        }
    }
}
