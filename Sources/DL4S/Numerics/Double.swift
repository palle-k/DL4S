//
//  Double.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation
import Accelerate


extension Double: NumericType {
    public static func vSquare(values: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vsqD(values, 1, result, 1, UInt(count))
    }
    
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
    
    public static func vMA(lhs: UnsafePointer<Double>, rhs: UnsafePointer<Double>, add: UnsafeMutablePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vmaD(lhs, 1, rhs, 1, add, 1, result, 1, UInt(count))
    }
    
    public static func vDiv(lhs: UnsafePointer<Double>, rhs: UnsafePointer<Double>, result: UnsafeMutablePointer<Double>, count: Int) {
        vDSP_vdivD(rhs, 1, lhs, 1, result, 1, UInt(count))
    }
    
    public static func sum(val: UnsafePointer<Double>, count: Int) -> Double {
        var result: Double = 0
        vDSP_sveD(val, 1, &result, UInt(count))
        return result
    }
}

