//
//  Int32.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation
import Accelerate


extension Int32: NumericType {
    public static func vSquare(values: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = values[i] * values[i]
        }
    }
    
    public init(floatLiteral value: Double) {
        self = Int32(value)
    }
    
    public static func tanh(val: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        fatalError("Tanh not implemented for Int32")
    }
    
    public static func relu(val: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = Swift.max(0, val[i])
        }
    }
    
    public static func transpose(val: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, width: Int, height: Int) {
        for x in 0 ..< width {
            for y in 0 ..< height {
                result[y + x * height] = val[y * width + x]
            }
        }
    }
    
    public static var one: Int32 {
        return 1
    }
    
    public static func vsAdd(lhs: UnsafePointer<Int32>, rhs: Int32, result: UnsafeMutablePointer<Int32>, count: Int) {
        vDSP_vsaddi(lhs, 1, [rhs], result, 1, UInt(count))
    }
    
    public static func vsMul(lhs: UnsafePointer<Int32>, rhs: Int32, result: UnsafeMutablePointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] * rhs
        }
    }
    
    public static func svDiv(lhs: Int32, rhs: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs / rhs[i]
        }
    }
    
    public static func fill(value: Int32, result: UnsafeMutablePointer<Int32>, count: Int) {
        vDSP_vfilli([value], result, 1, UInt(count))
    }
    
    public static func log(val: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        fatalError("Logarithm not supported for type Int32, cast to Float or Double first.")
    }
    
    public static func exp(val: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        fatalError("Exponentiation not supported for type Int32, cast to Float or Double first.")
    }
    
    public static func matMul(lhs: UnsafePointer<Int32>, rhs: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, lhsRows: Int, lhsCols: Int, rhsCols: Int) {
        fatalError("Matrix multiplication not supported for type Int32")
    }
    
    public static func vSub(lhs: UnsafePointer<Int32>, rhs: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] - rhs[i]
        }
    }
    
    public static func vMul(lhs: UnsafePointer<Int32>, rhs: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] * rhs[i]
        }
    }
    
    public static func vMA(lhs: UnsafePointer<Int32>, rhs: UnsafePointer<Int32>, add: UnsafeMutablePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] * rhs[i] + add[i]
        }
    }
    
    public static func vDiv(lhs: UnsafePointer<Int32>, rhs: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] * rhs[i]
        }
    }
    
    public static func vAdd(lhs: UnsafePointer<Int32>, rhs: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        vDSP_vaddi(lhs, 1, rhs, 1, result, 1, UInt(count))
    }
    
    public static func vNeg(val: UnsafePointer<Int32>, result: UnsafeMutablePointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = -val[i]
        }
    }
    
    public static func sum(val: UnsafePointer<Int32>, count: Int) -> Int32 {
        var result: Int32 = 0
        for i in 0 ..< count {
            result += val[i]
        }
        return result
    }
}
