//
//  Buffer.swift
//  DL4S
//
//  Created by Palle Klewitz on 25.02.19.
//

import Foundation


infix operator .* : MultiplicationPrecedence


public protocol Allocator {
    associatedtype BufferType
    
    static func zeros(shape: [Int]) -> BufferType
}

enum CPUAllocator: Allocator {
    public static func zeros(shape: [Int]) -> RAMBuffer<Float> {
        return RAMBuffer(shape: shape, repeating: 0)
    }
    
}


public protocol Buffer: class {
    associatedtype DType: NumericType
    
    var shape: [Int] { get }
    
    func transposed() -> Self
    func reshaped(_ newShape: [Int]) -> Self
    
    subscript(_ index: Int) -> Self { get set }
    subscript(_ index: [Int]) -> DType { get set }
    // subscript(_ index: [(Int, Int)]) -> Self { get set }
    
    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static prefix func - (value: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Self) -> Self
    static func .* (lhs: Self, rhs: Self) -> Self
    
    static func exp(_ value: Self) -> Self
    static func log(_ value: Self) -> Self
    static func relu(_ value: Self) -> Self
    static func tanh(_ value: Self) -> Self
}

public extension Buffer {
    var ndim: Int {
        return shape.count
    }
    
    var count: Int {
        return shape.reduce(1, *)
    }
}


public final class RAMBuffer<DType: NumericType>: Buffer {
    
    // Memory layout is always row major, so first dimension -> highest stride, last dimension -> stride of 1
    
    public let shape: [Int]
    private var values: UnsafeMutablePointer<DType>
    
    private var strides: [Int] {
        return Array(shape.reversed().reduce(into: [1], { acc, dim in
            acc.append(acc.last! * dim)
        }).reversed().dropFirst())
    }
    
    init(shape: [Int], repeating repeatedValue: DType) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = UnsafeMutablePointer.allocate(capacity: count)
        
        DType.fill(value: repeatedValue, result: self.values, count: count)
    }
    
    init(shape: [Int]) {
        let count = shape.reduce(1, *)
        
        self.shape = shape
        self.values = UnsafeMutablePointer.allocate(capacity: count)
    }
    
    init(shape: [Int], values: UnsafeMutablePointer<DType>) {
        self.shape = shape
        self.values = values
    }
    
    deinit {
        values.deallocate()
    }
    
    public subscript(index: Int) -> RAMBuffer<DType> {
        get {
            let count = strides[0]
            let insertionIdx = count * index
            
            let valueBuffer = UnsafeMutablePointer<DType>.allocate(capacity: count)
            memcpy(valueBuffer, self.values.advanced(by: insertionIdx), MemoryLayout<DType>.stride * count)
            
            return RAMBuffer(
                shape: Array(shape.dropFirst()),
                values: valueBuffer
            )
        }
        set (slice) {
            let count = strides[0]
            let insertionIdx = count * index
            memcpy(self.values.advanced(by: insertionIdx), slice.values, MemoryLayout<DType>.stride * count)
        }
    }
    
    public subscript(index: [Int]) -> DType {
        get {
            let idx = zip(index, strides).map(*).reduce(0, +)
            return values[idx]
        }
        set {
            let idx = zip(index, strides).map(*).reduce(0, +)
            values[idx] = newValue
        }
    }
    
    public static func + (lhs: RAMBuffer<DType>, rhs: RAMBuffer<DType>) -> RAMBuffer<DType> {
        precondition(lhs.shape == rhs.shape)
        
        let result = RAMBuffer(shape: lhs.shape)
        DType.vAdd(lhs: lhs.values, rhs: rhs.values, result: result.values, count: lhs.count)
        return result
    }
    
    public static func - (lhs: RAMBuffer<DType>, rhs: RAMBuffer<DType>) -> RAMBuffer<DType> {
        precondition(lhs.shape == rhs.shape)
        
        let result = RAMBuffer(shape: lhs.shape)
        DType.vSub(lhs: lhs.values, rhs: rhs.values, result: result.values, count: lhs.count)
        return result
    }
    
    public static prefix func - (value: RAMBuffer<DType>) -> RAMBuffer<DType> {
        let result = RAMBuffer(shape: value.shape)
        DType.vNeg(val: value.values, result: result.values, count: value.count)
        return result
    }
    
    public static func * (lhs: RAMBuffer<DType>, rhs: RAMBuffer<DType>) -> RAMBuffer<DType> {
        precondition(lhs.shape == rhs.shape)
        
        let result = RAMBuffer(shape: lhs.shape)
        DType.vMul(lhs: lhs.values, rhs: rhs.values, result: result.values, count: lhs.count)
        return result
    }
    
    public static func / (lhs: RAMBuffer<DType>, rhs: RAMBuffer<DType>) -> RAMBuffer<DType> {
        precondition(lhs.shape == rhs.shape)
        
        let result = RAMBuffer(shape: lhs.shape)
        DType.vDiv(lhs: lhs.values, rhs: rhs.values, result: result.values, count: lhs.count)
        return result
    }
    
    public static func .* (lhs: RAMBuffer<DType>, rhs: RAMBuffer<DType>) -> RAMBuffer<DType> {
        precondition(lhs.shape[1] == rhs.shape[0])
        precondition(lhs.ndim == 2)
        precondition(rhs.ndim == 2)
        
        let result = RAMBuffer(shape: [lhs.shape[0], rhs.shape[1]])
        DType.matMul(lhs: lhs.values, rhs: rhs.values, result: result.values, lhsRows: lhs.shape[0], lhsCols: lhs.shape[1], rhsCols: rhs.shape[1])
        return result
    }
    
    public func transposed() -> RAMBuffer<DType> {
        precondition(ndim == 2)
        
        let copy = UnsafeMutablePointer<DType>.allocate(capacity: count)
        DType.transpose(val: values, result: copy, width: shape[1], height: shape[0])
        return RAMBuffer(shape: shape.reversed(), values: copy)
    }
    
    public func reshaped(_ newShape: [Int]) -> RAMBuffer<DType> {
        let copy = UnsafeMutablePointer<DType>.allocate(capacity: count)
        memcpy(copy, values, count * MemoryLayout<DType>.stride)
        return RAMBuffer(shape: newShape, values: copy)
    }
    
    public static func exp(_ value: RAMBuffer<DType>) -> RAMBuffer<DType> {
        let copy = UnsafeMutablePointer<DType>.allocate(capacity: value.count)
        DType.exp(val: value.values, result: copy, count: value.count)
        return RAMBuffer(shape: value.shape, values: copy)
    }
    
    public static func log(_ value: RAMBuffer<DType>) -> RAMBuffer<DType> {
        let copy = UnsafeMutablePointer<DType>.allocate(capacity: value.count)
        DType.log(val: value.values, result: copy, count: value.count)
        return RAMBuffer(shape: value.shape, values: copy)
    }
    
    public static func relu(_ value: RAMBuffer<DType>) -> RAMBuffer<DType> {
        let copy = UnsafeMutablePointer<DType>.allocate(capacity: value.count)
        DType.relu(val: value.values, result: copy, count: value.count)
        return RAMBuffer(shape: value.shape, values: copy)
    }
    
    public static func tanh(_ value: RAMBuffer<DType>) -> RAMBuffer<DType> {
        let copy = UnsafeMutablePointer<DType>.allocate(capacity: value.count)
        DType.tanh(val: value.values, result: copy, count: value.count)
        return RAMBuffer(shape: value.shape, values: copy)
    }
}
