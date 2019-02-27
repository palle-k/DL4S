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
    
    static func fill(value: Self, result: UnsafeMutablePointer<Self>, count: Int)
    static func transpose(val: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, width: Int, height: Int)
    
    static func vAdd(lhs: UnsafePointer<Self>, rhs: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func vsAdd(lhs: UnsafePointer<Self>, rhs: Self, result: UnsafeMutablePointer<Self>, count: Int)
    static func vNeg(val: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func vSub(lhs: UnsafePointer<Self>, rhs: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    
    static func vMul(lhs: UnsafePointer<Self>, rhs: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func vMA(lhs: UnsafePointer<Self>, rhs: UnsafePointer<Self>, add: UnsafeMutablePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func vsMul(lhs: UnsafePointer<Self>, rhs: Self, result: UnsafeMutablePointer<Self>, count: Int)
    
    static func vDiv(lhs: UnsafePointer<Self>, rhs: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func svDiv(lhs: Self, rhs: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    
    static func vSquare(values: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    
    static func matMul(lhs: UnsafePointer<Self>, rhs: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, lhsRows: Int, lhsCols: Int, rhsCols: Int)
    
    static func log(val: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func exp(val: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func relu(val: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func tanh(val: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    
    static func sum(val: UnsafePointer<Self>, count: Int) -> Self
}
