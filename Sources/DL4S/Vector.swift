//
//  Vector.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
//

import Foundation

public class Vector<Element: NumericType>: ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral {
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
    
    var values: UnsafeMutableBufferPointer<Element>
    var gradient: UnsafeMutableBufferPointer<Element>
    var context: AnyVectorOperation<Element>?
    
    // Specifies, whether the vector is the owner of the value pointers
    var parent: Vector<Element>?
    
    public init(repeating value: Element, shape: Int...) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = UnsafeMutableBufferPointer.allocate(capacity: count)
        self.gradient = UnsafeMutableBufferPointer.allocate(capacity: count)
        self.context = nil
        self.parent = nil
        
        Element.fill(value: value, result: values, count: count)
        Element.fill(value: 0, result: gradient, count: count)
    }
    
    public init(_ elements: [Element], shape: Int...) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = UnsafeMutableBufferPointer.allocate(capacity: count)
        self.gradient = UnsafeMutableBufferPointer.allocate(capacity: count)
        self.context = nil
        self.parent = nil
        
        for (i, el) in elements.enumerated() {
            values[i] = el
        }
        Element.fill(value: 0, result: gradient, count: count)
    }
    
    public convenience init(_ array: [Element]) {
        self.init(array, shape: array.count)
    }
    
    public convenience init(_ array: [[Element]]) {
        precondition(array.allSatisfy {$0.count == array[0].count}, "Invalid shape, all rows of array must be of same length.")
        self.init(Array(array.joined()), shape: array.count, array.first?.count ?? 0)
    }
    
    public convenience init(_ array: [[[Element]]]) {
        precondition(array.allSatisfy {
            $0.count == array[0].count && $0.allSatisfy {
                $0.count == array[0][0].count
            }
        }, "Invalid shape, all rows of array must be of same length.")
        self.init(
            Array(array.joined().joined()),
            shape: array.count,
            array.first?.count ?? 0,
            array.first?.first?.count ?? 0
        )
    }
    
    public convenience init(_ array: [[[[Element]]]]) {
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
            array.first?.first?.first?.count ?? 0
        )
    }
    
    public convenience init(_ array: [[[[[Element]]]]]) {
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
            array.first?.first?.first?.first?.count ?? 0
        )
    }
    
    init(values: UnsafeMutableBufferPointer<Element>, gradient: UnsafeMutableBufferPointer<Element>, shape: [Int], parent: Vector<Element>? = nil, context: AnyVectorOperation<Element>?) {
        self.values = values
        self.gradient = gradient
        self.shape = shape
        self.parent = parent
        self.context = context
    }
    
    init(values: UnsafeMutableBufferPointer<Element>, shape: [Int], parent: Vector<Element>? = nil, context: AnyVectorOperation<Element>?) {
        self.values = values
        self.gradient = UnsafeMutableBufferPointer<Element>.allocate(capacity: shape.reduce(1, *))
        self.shape = shape
        self.parent = parent
        self.context = context
        
        Element.fill(value: 0, result: self.gradient, count: count)
    }
    
    public init(_ value: Element) {
        self.shape = []
        self.values = UnsafeMutableBufferPointer.allocate(capacity: 1)
        self.gradient = UnsafeMutableBufferPointer.allocate(capacity: 1)
        self.context = nil
        self.parent = nil
        
        self.values.pointee = value
        self.gradient.pointee = 0
    }
    
    public required init(floatLiteral value: Double) {
        self.shape = []
        self.values = UnsafeMutableBufferPointer.allocate(capacity: 1)
        self.gradient = UnsafeMutableBufferPointer.allocate(capacity: 1)
        self.context = nil
        self.parent = nil
        self.values.pointee = Element(value)
        self.gradient.pointee = 0
    }
    
    public required init(integerLiteral value: Int32) {
        self.shape = []
        self.values = UnsafeMutableBufferPointer.allocate(capacity: 1)
        self.gradient = UnsafeMutableBufferPointer.allocate(capacity: 1)
        self.context = nil
        self.parent = nil
        self.values.pointee = Element(value)
        self.gradient.pointee = 0
    }
    
    init(shape: [Int], parent: Vector<Element>?, context: AnyVectorOperation<Element>?) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = UnsafeMutableBufferPointer.allocate(capacity: count)
        self.gradient = UnsafeMutableBufferPointer.allocate(capacity: count)
        self.context = context
        self.parent = parent
        
        Element.fill(value: 0, result: gradient, count: count)
    }
    
    deinit {
        if parent == nil {
            values.deallocate()
            gradient.deallocate()
        }
    }
    
    func resignOwnership(to parent: Vector<Element>) {
        self.parent = parent
    }
    
    func ensureOwnership() {
        if parent == nil {
            return
        }
        
        let existingValues = self.values
        let existingGradient = self.gradient
        
        self.values = UnsafeMutableBufferPointer.allocate(capacity: count)
        self.gradient = UnsafeMutableBufferPointer.allocate(capacity: count)
        
        self.values.assign(from: existingValues.immutable, count: count)
        self.gradient.assign(from: existingGradient.immutable, count: count)
        
        self.parent = nil
    }
    
    func zeroGradient() {
        Element.fill(value: 0, result: gradient, count: count)
        context?.zeroGradient()
    }
    
    func _backwards() {
        context?.backwards(from: self)
    }
    
    public func backwards() {
        Element.fill(value: 1, result: gradient, count: count)
        _backwards()
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
    
    public var gradientItem: Element {
        if dim == 0 {
            return gradient.pointee
        } else {
            fatalError("Cannot obtain item of vector with dimensionality other than 0")
        }
    }
    
    public func detached() -> Vector<Element> {
        let copy = Vector(values: values, gradient: gradient, shape: shape, parent: self, context: nil)
        copy.ensureOwnership()
        return copy
    }
    
    static func sameIdentity(_ vector1: Vector<Element>, _ vector2: Vector<Element>) -> Bool {
        return vector1.values == vector2.values && vector1.gradient == vector2.gradient
    }
    
    static func sameIdentity(_ vector1: Vector<Element>?, _ vector2: Vector<Element>?) -> Bool {
        if let vector1 = vector1, let vector2 = vector2 {
            return sameIdentity(vector1, vector2)
        } else {
            return false
        }
    }
}

extension Vector: CustomStringConvertible, CustomDebugStringConvertible {
    private func generateDescription() -> String {
        
        if dim > 1 {
            let d = (0 ..< shape[0])
                .map {self[$0].generateDescription()}
                .joined(separator: ",\n")
                .replacingOccurrences(of: "\n", with: "\n ")
            return "[\(d)]"
        } else if let count = self.shape.first {
            let d = (0 ..< count)
                .map {"\(self[$0].item)"}
                .joined(separator: ", ")
            return "[\(d)]"
        } else {
            return "\(item)"
        }
    }
    
    private func generateGradientDescription() -> String {
        if dim > 1 {
            let d = (0 ..< shape[0])
                .map {self[$0].generateGradientDescription()}
                .joined(separator: ",\n")
                .replacingOccurrences(of: "\n", with: "\n ")
            return "[\(d)]"
        } else if let count = self.shape.first {
            let d = (0 ..< count)
                .map {"\(self[$0].gradientItem)"}
                .joined(separator: ", ")
            return "[\(d)]"
        } else {
            return "\(gradientItem)"
        }
    }
    
    public var description: String {
        return generateDescription()
    }
    
    public var gradientDescription: String {
        return generateGradientDescription()
    }
    
    public var debugDescription: String {
        return """
        Vector<\(Element.self)>(
            shape: \(self.shape)
            elements: \(generateDescription().replacingOccurrences(of: "\n", with: "\n\t")),
            gradient: \(generateGradientDescription().replacingOccurrences(of: "\n", with: "\n\t"))
        )
        """
    }
}

// Memory operation extensions
extension Vector {
    func buffer(from indices: [Int?]) -> (UnsafeMutableBufferPointer<Element>, Bool, [Int]) {
        return MemoryOps.get(slice: indices, of: values, with: shape)
    }
    
    func setBuffer(at indices: [Int?], source: UnsafeBufferPointer<Element>, sourceShape: [Int]) {
        MemoryOps.set(slice: indices, of: values, with: shape, from: source, with: sourceShape)
    }
    
    func gradient(from indices: [Int?]) -> (UnsafeMutableBufferPointer<Element>, Bool, [Int]) {
        return MemoryOps.get(slice: indices, of: gradient, with: shape)
    }
    
    func setGradient(at indices: [Int?], source: UnsafeBufferPointer<Element>, sourceShape: [Int]) {
        MemoryOps.set(slice: indices, of: gradient, with: shape, from: source, with: sourceShape)
    }
    
    func buffer(from indices: [Range<Int>?]) -> (UnsafeMutableBufferPointer<Element>, Bool, [Int]) {
        return MemoryOps.get(slice: indices, of: values, with: shape)
    }
    
    func setBuffer(at indices: [Range<Int>?], source: UnsafeBufferPointer<Element>, sourceShape: [Int]) {
        MemoryOps.set(slice: indices, of: values, with: shape, from: source, with: sourceShape)
    }
    
    func gradient(from indices: [Range<Int>?]) -> (UnsafeMutableBufferPointer<Element>, Bool, [Int]) {
        return MemoryOps.get(slice: indices, of: gradient, with: shape)
    }
    
    func setGradient(at indices: [Range<Int>?], source: UnsafeBufferPointer<Element>, sourceShape: [Int]) {
        MemoryOps.set(slice: indices, of: gradient, with: shape, from: source, with: sourceShape)
    }
}

func iterate(_ shape: [Int]) -> AnySequence<[Int]> {
    func increment<S: RandomAccessCollection>(_ list: S, shape: S) -> ([Int], Bool) where S.Element == Int {
        guard let first = list.first, let firstDim = shape.first else {
            return ([], true)
        }
        let (rest, overflows) = increment(list.dropFirst(), shape: shape.dropFirst())
        if overflows {
            if first + 1 >= firstDim {
                return ([0] + rest, true)
            } else {
                return ([first + 1] + rest, false)
            }
        } else {
            return ([first] + rest, false)
        }
    }
    
    return AnySequence(sequence(first: Array(repeating: 0, count: shape.count)) { state -> [Int]? in
        let (incremented, overflows) = increment(state, shape: shape)
        return overflows ? nil : incremented
    })
}
