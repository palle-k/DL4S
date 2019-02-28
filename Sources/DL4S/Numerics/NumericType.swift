//
//  VectorNumerics.swift
//  DL4S
//
//  Created by Palle Klewitz on 25.02.19.
//

import Foundation
import Accelerate

public protocol NumericType: Hashable, ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral {
    init(_ floatValue: Double)
    init(_ integerValue: Int32)
    
    static var zero: Self { get }
    static var one: Self { get }
    
    static prefix func - (value: Self) -> Self
    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    
    func sqrt() -> Self
    func log() -> Self
    func exp() -> Self
    func sin() -> Self
    func cos() -> Self
    func tan() -> Self
    func sinh() -> Self
    func cosh() -> Self
    func tanh() -> Self
    
    init(_ float: Float)
    init(_ int: Int)
    init(_ uint: UInt)
    
    static func fill(value: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func transpose(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, srcRows: Int, srcCols: Int)
    
    static func vAdd(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vsAdd(lhs: UnsafeBufferPointer<Self>, rhs: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vNeg(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vSub(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func vMul(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vMA(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, add: UnsafeMutableBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vsMul(lhs: UnsafeBufferPointer<Self>, rhs: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func vDiv(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func svDiv(lhs: Self, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func vSquare(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func matMul(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsRows: Int, lhsCols: Int, rhsCols: Int)
    static func matMulAddInPlace(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), transposeFirst: Bool, transposeSecond: Bool)
    static func dot(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, count: Int) -> Self
    
    static func vMulSA(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, add: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vsMulVAdd(lhs: UnsafeBufferPointer<Self>, rhs: Self, add: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func log(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func exp(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func relu(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func tanh(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func sum(val: UnsafeBufferPointer<Self>, count: Int) -> Self
    
    static func copysign(values: UnsafeBufferPointer<Self>, signs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func argmax(values: UnsafeBufferPointer<Self>, count: Int) -> (Int, Self)
}
