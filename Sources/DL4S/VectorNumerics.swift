//
//  VectorNumerics.swift
//  DL4S
//
//  Created by Palle Klewitz on 25.02.19.
//

import Foundation
import Accelerate

public protocol NumericType: Equatable {
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
    static func vsMul(lhs: UnsafePointer<Self>, rhs: Self, result: UnsafeMutablePointer<Self>, count: Int)
    
    static func vDiv(lhs: UnsafePointer<Self>, rhs: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func svDiv(lhs: Self, rhs: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    
    static func matMul(lhs: UnsafePointer<Self>, rhs: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, lhsRows: Int, lhsCols: Int, rhsCols: Int)
    
    static func log(val: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func exp(val: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func relu(val: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
    static func tanh(val: UnsafePointer<Self>, result: UnsafeMutablePointer<Self>, count: Int)
}

extension Int32: NumericType {
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
}

extension Float: NumericType {
    public static func relu(val: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vthr(val, 1, [0.0], result, 1, UInt(count))
    }
    
    public static func tanh(val: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vvtanhf(result, val, [Int32(count)])
    }
    
    public static func transpose(val: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, width: Int, height: Int) {
        vDSP_mtrans(val, 1, result, 1, UInt(width), UInt(height))
    }
    
    public static var one: Float {
        return 1.0
    }
    
    public static func vsAdd(lhs: UnsafePointer<Float>, rhs: Float, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vsadd(lhs, 1, [rhs], result, 1, UInt(count))
    }
    
    public static func vsMul(lhs: UnsafePointer<Float>, rhs: Float, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vsmul(lhs, 1, [rhs], result, 1, UInt(count))
    }
    
    public static func svDiv(lhs: Float, rhs: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_svdiv([lhs], rhs, 1, result, 1, UInt(count))
    }
    
    public static func fill(value: Float, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vfill([value], result, 1, UInt(count))
    }
    
    public static func exp(val: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vvexpf(result, val, [Int32(count)])
    }
    
    public static func log(val: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vvlogf(result, val, [Int32(count)])
    }
    
    public static func matMul(lhs: UnsafePointer<Float>, rhs: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, lhsRows: Int, lhsCols: Int, rhsCols: Int) {
        vDSP_mmul(lhs, 1, rhs, 1, result, 1, UInt(lhsRows), UInt(rhsCols), UInt(lhsCols))
    }
    
    public static func vAdd(lhs: UnsafePointer<Float>, rhs: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vadd(lhs, 1, rhs, 1, result, 1, UInt(count))
    }
    
    public static func vNeg(val: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vneg(val, 1, result, 1, UInt(count))
    }
    
    public static func vSub(lhs: UnsafePointer<Float>, rhs: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vsub(rhs, 1, lhs, 1, result, 1, UInt(count))
    }
    
    public static func vMul(lhs: UnsafePointer<Float>, rhs: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vmul(lhs, 1, rhs, 1, result, 1, UInt(count))
    }
    
    public static func vDiv(lhs: UnsafePointer<Float>, rhs: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vdiv(rhs, 1, lhs, 1, result, 1, UInt(count))
    }
}

extension Double: NumericType {
    public static func relu(val: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vthrD(val, 1, [0.0], result, 1, UInt(count))
    }
    
    public static func tanh(val: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vvtanh(result, val, [Int32(count)])
    }
    
    public static func transpose(val: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, width: Int, height: Int) {
        vDSP_mtransD(val, 1, result, 1, UInt(width), UInt(height))
    }
    
    public static var one: Double {
        return 1.0
    }
    
    public static func vsAdd(lhs: UnsafePointer<Double>, rhs: Double, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vsaddD(lhs, 1, [rhs], result, 1, UInt(count))
    }
    
    public static func vsMul(lhs: UnsafePointer<Double>, rhs: Double, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vsmulD(lhs, 1, [rhs], result, 1, UInt(count))
    }
    
    public static func svDiv(lhs: Double, rhs: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_svdivD([lhs], rhs, 1, result, 1, UInt(count))
    }
    
    public static func fill(value: Double, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vfillD([value], result, 1, UInt(count))
    }
    
    public static func exp(val: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vvexp(result, val, [Int32(count)])
    }
    
    public static func log(val: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vvlog(result, val, [Int32(count)])
    }
    
    public static func matMul(lhs: UnsafePointer<Double>, rhs: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, lhsRows: Int, lhsCols: Int, rhsCols: Int) {
        vDSP_mmulD(lhs, 1, rhs, 1, result, 1, UInt(lhsRows), UInt(rhsCols), UInt(lhsCols))
    }
    
    public static func vAdd(lhs: UnsafePointer<Double>, rhs: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vaddD(lhs, 1, rhs, 1, result, 1, UInt(count))
    }
    
    public static func vNeg(val: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vnegD(val, 1, result, 1, UInt(count))
    }
    
    public static func vSub(lhs: UnsafePointer<Double>, rhs: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vsubD(rhs, 1, lhs, 1, result, 1, UInt(count))
    }
    
    public static func vMul(lhs: UnsafePointer<Double>, rhs: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vmulD(lhs, 1, rhs, 1, result, 1, UInt(count))
    }
    
    public static func vDiv(lhs: UnsafePointer<Double>, rhs: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vdivD(rhs, 1, lhs, 1, result, 1, UInt(count))
    }
}

