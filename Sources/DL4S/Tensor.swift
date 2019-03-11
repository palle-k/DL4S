//
//  Tensor.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
//

import Foundation

public class Tensor<Element: NumericType>: ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral, Codable {
    public typealias FloatLiteralType = Double
    public typealias IntegerLiteralType = Int32
    
    public let shape: [Int]
    
    public var count: Int {
        return shape.reduce(1, *)
    }
    
    public var dim: Int {
        return shape.count
    }
    
    var strides: [Int] {
        return MemoryOps.strides(from: shape)
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
                let gradient: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: count)
                Element.fill(value: 0, result: gradient, count: count)
                self.gradient = gradient
            } else {
                gradient.map(CPUAllocator.free)
            }
        }
    }
    
    var values: UnsafeMutableBufferPointer<Element>
    
    var gradient: UnsafeMutableBufferPointer<Element>?
    
    var context: AnyTensorOperation<Element>?
    
    // Specifies, whether the vector is the owner of the value pointers
    var parent: Tensor<Element>?
    
    public init(repeating value: Element, shape: [Int], requiresGradient: Bool = false) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = CPUAllocator.allocate(count: count)
        self.gradient = nil
        self.context = nil
        self.parent = nil
        self.requiresGradient = requiresGradient
        
        Element.fill(value: value, result: values, count: count)
    }
    
    public convenience init(repeating value: Element, shape: Int..., requiresGradient: Bool = false) {
        self.init(repeating: value, shape: shape, requiresGradient: requiresGradient)
    }
    
    public init(_ elements: [Element], shape: [Int], requiresGradient: Bool = false) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = CPUAllocator.allocate(count: count)
        self.gradient = nil
        self.context = nil
        self.parent = nil
        
        for (i, el) in elements.enumerated() {
            values[i] = el
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
    
    init(values: UnsafeMutableBufferPointer<Element>, gradient: UnsafeMutableBufferPointer<Element>?, shape: [Int], parent: Tensor<Element>? = nil, context: AnyTensorOperation<Element>?) {
        self.values = values
        self.gradient = gradient
        self.shape = shape
        self.parent = parent
        self.context = context
    }
    
    init(values: UnsafeMutableBufferPointer<Element>, shape: [Int], parent: Tensor<Element>? = nil, context: AnyTensorOperation<Element>?) {
        self.values = values
        self.gradient = nil
        self.shape = shape
        self.parent = parent
        self.context = context
        
        Element.fill(value: 0, result: self.gradient!, count: count)
        self.requiresGradient = context?.sourceTensors.contains(where: {$0.requiresGradient}) ?? false
    }
    
    public init(_ value: Element, requiresGradient: Bool = false) {
        self.shape = []
        self.values = CPUAllocator.allocate(count: 1)
        self.gradient = nil
        self.context = nil
        self.parent = nil
        
        self.values.pointee = value
        self.requiresGradient = requiresGradient
    }
    
    public required init(floatLiteral value: Double) {
        self.shape = []
        self.values = CPUAllocator.allocate(count: 1)
        self.gradient = nil
        self.context = nil
        self.parent = nil
        self.values.pointee = Element(value)
    }
    
    public required init(integerLiteral value: Int32) {
        self.shape = []
        self.values = CPUAllocator.allocate(count: 1)
        self.gradient = nil
        self.context = nil
        self.parent = nil
        self.values.pointee = Element(value)
    }
    
    init(shape: [Int], parent: Tensor<Element>?, context: AnyTensorOperation<Element>?) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = CPUAllocator.allocate(count: count)
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
        let values = try container.decode([Element].self, forKey: .values)
        
        self.shape = shape
        self.values = CPUAllocator.allocate(count: shape.reduce(1, *))
        self.context = nil
        self.parent = nil
        
        if let gradient = try container.decode([Element]?.self, forKey: .gradient) {
            self.gradient = CPUAllocator.allocate(count: shape.reduce(1, *))
            gradient.withUnsafeBufferPointer { ptr in
                self.gradient?.assign(from: ptr, count: ptr.count)
            }
        } else {
            self.gradient = nil
        }
        
        values.withUnsafeBufferPointer { ptr in
            self.values.assign(from: ptr, count: ptr.count)
        }
    }
    
    deinit {
        if parent == nil {
            CPUAllocator.free(values)
            gradient.map(CPUAllocator.free)
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encode(shape, forKey: .shape)
        try container.encode(Array(self.values), forKey: .values)
        
        if let gradient = self.gradient {
            try container.encode(Array(gradient), forKey: .gradient)
        } else {
            try container.encode([Element]?.none, forKey: .gradient)
        }
    }
    
    func resignOwnership(to parent: Tensor<Element>) {
        self.parent = parent
    }
    
    func ensureOwnership() {
        if parent == nil {
            return
        }
        
        let existingValues = self.values
        let existingGradient = self.gradient
        
        self.values = CPUAllocator.allocate(count: count)
        self.values.assign(from: existingValues.immutable, count: count)
        
        if let gradient = existingGradient {
            self.gradient = CPUAllocator.allocate(count: count)
            self.gradient!.assign(from: gradient.immutable, count: count)
        }
        
        self.parent = nil
    }
    
    public func zeroGradient() {
        guard let gradient = self.gradient else {
            return
        }
        Element.fill(value: 0, result: gradient, count: count)
        context?.zeroGradient()
    }
    
    func _backwards() {
        context?.fillSourceGradients(fromResultGradients: self)
    }
    
    public func backwards() {
        guard let gradient = self.gradient else {
            return
        }
        Element.fill(value: 1, result: gradient, count: count)
        
        var ordering: [Tensor<Element>] = []
        var visited: Set<Tensor<Element>> = []
        sorted(sorting: &ordering, visited: &visited)
        
        for tensor in ordering.reversed() {
            tensor._backwards()
        }
    }
    
    /// Optains the element of the tensor
    /// Only applicable to zero-dimensional tensors (scalars)
    /// The item is not further tracked with automatic differentiation
    public var item: Element {
        if dim == 0 {
            return values.pointee
        } else {
            fatalError("Cannot obtain item of vector with dimensionality other than 0")
        }
    }
    
    public var gradientItem: Element? {
        if dim == 0 {
            return gradient?.pointee
        } else {
            fatalError("Cannot obtain item of vector with dimensionality other than 0")
        }
    }
    
    public func detached() -> Tensor<Element> {
        let copy = Tensor(values: values, gradient: gradient, shape: shape, parent: self, context: nil)
        copy.ensureOwnership()
        return copy
    }
    
    static func sameIdentity(_ vector1: Tensor<Element>, _ vector2: Tensor<Element>) -> Bool {
        return vector1.values == vector2.values && vector1.gradient == vector2.gradient
    }
    
    static func sameIdentity(_ vector1: Tensor<Element>?, _ vector2: Tensor<Element>?) -> Bool {
        if let vector1 = vector1, let vector2 = vector2 {
            return sameIdentity(vector1, vector2)
        } else {
            return false
        }
    }
}
