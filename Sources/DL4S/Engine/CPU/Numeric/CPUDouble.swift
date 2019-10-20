//
//  CPUDouble.swift
//  DL4S
//
//  Created by Palle Klewitz on 20.10.19.
//

import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

extension Double: CPUNumeric {
    public static func fill(value: Double, result: UnsafeMutableBufferPointer<Double>, stride: Int, count: Int) {
        vDSP_vfillD([value], result.pointer(capacity: count), stride, UInt(count))
    }
    
    public static func relu(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vthrD(val.pointer(capacity: count), 1, [0.0], result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func tanh(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vvtanh(result.pointer(capacity: count), val.pointer(capacity: count), [Int32(count)])
    }
    
    public static func transpose(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, srcRows: Int, srcCols: Int) {
        vDSP_mtransD(val.pointer(capacity: srcRows * srcCols), 1, result.pointer(capacity: srcRows * srcCols), 1, UInt(srcCols), UInt(srcRows))
    }
    
    public static func sqrt(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vvsqrt(result.pointer(capacity: count), val.pointer(capacity: count), [Int32(count)])
    }
    
    public static func vsAdd(lhs: UnsafeBufferPointer<Double>, rhs: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vsaddD(lhs.pointer(capacity: count), 1, [rhs], result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func vsMul(lhs: UnsafeBufferPointer<Double>, rhs: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vsmulD(lhs.pointer(capacity: count), 1, [rhs], result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func svDiv(lhs: Double, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_svdivD([lhs], rhs.pointer(capacity: count), 1, result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func fill(value: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vfillD([value], result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func exp(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vvexp(result.pointer(capacity: count), val.pointer(capacity: count), [Int32(count)])
    }
    
    public static func log(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vvlog(result.pointer(capacity: count), val.pointer(capacity: count), [Int32(count)])
    }
    
    public static func vAdd(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vaddD(lhs.pointer(capacity: count), 1, rhs.pointer(capacity: count), 1, result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func vNeg(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vnegD(val.pointer(capacity: count), 1, result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func vSub(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vsubD(rhs.pointer(capacity: count), 1, lhs.pointer(capacity: count), 1, result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func vMul(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vmulD(lhs.pointer(capacity: count), 1, rhs.pointer(capacity: count), 1, result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func vDiv(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vdivD(rhs.pointer(capacity: count), 1, lhs.pointer(capacity: count), 1, result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func sum(val: UnsafeBufferPointer<Double>, count: Int) -> Double {
        var result: Double = 0
        vDSP_sveD(val.pointer(capacity: count), 1, &result, UInt(count))
        return result
    }
    
    public static func sum(val: UnsafeBufferPointer<Double>, stride: Int, count: Int) -> Double {
        var result: Double = 0
        vDSP_sveD(val.pointer(capacity: count), stride, &result, UInt(count))
        return result
    }
    
    public static func gemm(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), alpha: Self, beta: Self, transposeFirst: Bool, transposeSecond: Bool) {
        precondition((transposeFirst ? lhsShape.1 : lhsShape.0) == resultShape.0)
        precondition((transposeSecond ? rhsShape.0 : rhsShape.1) == resultShape.1)
        precondition((transposeFirst ? lhsShape.0 : lhsShape.1) == (transposeSecond ? rhsShape.1 : rhsShape.0))
        
        cblas_dgemm(
            CblasRowMajor,
            transposeFirst ? CblasTrans : CblasNoTrans,
            transposeSecond ? CblasTrans : CblasNoTrans,
            Int32(resultShape.0),
            Int32(resultShape.1),
            Int32(transposeFirst ? lhsShape.0 : lhsShape.1),
            alpha,
            lhs.pointer(capacity: lhsShape.0 * lhsShape.1),
            Int32(lhsShape.1),
            rhs.pointer(capacity: rhsShape.0 * rhsShape.1),
            Int32(rhsShape.1),
            beta,
            result.pointer(capacity: resultShape.0 * resultShape.1),
            Int32(resultShape.1)
        )
    }
    
    public static func argmax(values: UnsafeBufferPointer<Double>, count: Int) -> (Int, Double) {
        var maxI: UInt = 0
        var maxV: Double = 0
        
        vDSP_maxviD(values.pointer(capacity: count), 1, &maxV, &maxI, UInt(count))
        
        return (Int(maxI), maxV)
    }
    
    public static func argmin(values: UnsafeBufferPointer<Double>, count: Int) -> (Int, Double) {
        var minI: UInt = 0
        var minV: Double = 0
        
        vDSP_minviD(values.pointer(capacity: count), 1, &minV, &minI, UInt(count))
        
        return (Int(minI), minV)
    }
    
    public static func argmax(values: UnsafeBufferPointer<Double>, stride: Int, count: Int) -> (Int, Double) {
        var maxI: UInt = 0
        var maxV: Double = 0
        
        vDSP_maxviD(values.pointer(capacity: count), stride, &maxV, &maxI, UInt(count))
        
        return (Int(maxI) / stride, maxV)
    }
    
    public static func argmin(values: UnsafeBufferPointer<Double>, stride: Int, count: Int) -> (Int, Double) {
        var maxI: UInt = 0
        var maxV: Double = 0
        
        vDSP_minviD(values.pointer(capacity: count), stride, &maxV, &maxI, UInt(count))
        
        return (Int(maxI) / stride, maxV)
    }
    
    public static func copy(values: UnsafeBufferPointer<Double>, srcStride: Int, result: UnsafeMutableBufferPointer<Double>, dstStride: Int, count: Int) {
        cblas_dcopy(Int32(count), values.pointer(capacity: count * srcStride), Int32(srcStride), result.pointer(capacity: dstStride * count), Int32(dstStride))
    }
    
    public static func arange(start: Double, end: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vrampD([start], [end / Double(count)], result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func max(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vmaxD(lhs.pointer(capacity: count), 1, rhs.pointer(capacity: count), 1, result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func min(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vDSP_vminD(lhs.pointer(capacity: count), 1, rhs.pointer(capacity: count), 1, result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func sin(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vvsin(result.pointer(capacity: count), values.pointer(capacity: count), [Int32(count)])
    }
    
    public static func cos(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vvcos(result.pointer(capacity: count), values.pointer(capacity: count), [Int32(count)])
    }
    
    public static func tan(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        vvtan(result.pointer(capacity: count), values.pointer(capacity: count), [Int32(count)])
    }
}
