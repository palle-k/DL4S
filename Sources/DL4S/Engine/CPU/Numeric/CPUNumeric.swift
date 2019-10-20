//
//  CPUNumeric.swift
//  DL4S
//
//  Created by Palle Klewitz on 20.10.19.
//

import Foundation

public protocol CPUNumeric {
    static func fill(value: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func fill(value: Self, result: UnsafeMutableBufferPointer<Self>, stride: Int, count: Int)
    
    static func transpose(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, srcRows: Int, srcCols: Int)
    
    static func vAdd(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vsAdd(lhs: UnsafeBufferPointer<Self>, rhs: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vNeg(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vSub(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func vMul(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vsMul(lhs: UnsafeBufferPointer<Self>, rhs: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func vDiv(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func svDiv(lhs: Self, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)

    static func gemm(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), alpha: Self, beta: Self, transposeFirst: Bool, transposeSecond: Bool)
    
    static func log(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func exp(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func relu(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func tanh(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func sqrt(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func sum(val: UnsafeBufferPointer<Self>, count: Int) -> Self
    static func sum(val: UnsafeBufferPointer<Self>, stride: Int, count: Int) -> Self
    
    static func argmax(values: UnsafeBufferPointer<Self>, count: Int) -> (Int, Self)
    static func argmin(values: UnsafeBufferPointer<Self>, count: Int) -> (Int, Self)
    
    static func argmax(values: UnsafeBufferPointer<Self>, stride: Int, count: Int) -> (Int, Self)
    static func argmin(values: UnsafeBufferPointer<Self>, stride: Int, count: Int) -> (Int, Self)
    
    static func max(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func min(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func copy(values: UnsafeBufferPointer<Self>, srcStride: Int, result: UnsafeMutableBufferPointer<Self>, dstStride: Int, count: Int)
    static func arange(start: Self, end: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func sin(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func cos(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func tan(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
}
