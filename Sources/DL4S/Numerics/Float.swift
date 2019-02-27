//
//  Float.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation
import Accelerate


extension Float: NumericType {
    public static func vSquare(values: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vsq(values, 1, result, 1, UInt(count))
    }
    
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
    
    public static func vMA(lhs: UnsafePointer<Float>, rhs: UnsafePointer<Float>, add: UnsafeMutablePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vma(lhs, 1, rhs, 1, add, 1, result, 1, UInt(count))
    }
    
    public static func vDiv(lhs: UnsafePointer<Float>, rhs: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) {
        vDSP_vdiv(rhs, 1, lhs, 1, result, 1, UInt(count))
    }
    
    public static func sum(val: UnsafePointer<Float>, count: Int) -> Float {
        var result: Float = 0
        vDSP_sve(val, 1, &result, UInt(count))
        return result
    }
}
