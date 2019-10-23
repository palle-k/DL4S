//
//  GPUNumeric.swift
//  DL4S
//
//  Created by Palle Klewitz on 24.10.19.
//

import Foundation
#if canImport(Metal) && canImport(MetalPerformanceShaders)
import MetalPerformanceShaders

public protocol GPUNumeric: Numeric {
    static var gpuTypeIdentifier: String { get }
    static var mpsDataType: MPSDataType { get }
}

extension Float: GPUNumeric {
    public static var gpuTypeIdentifier: String {
        return "Float32"
    }
    
    public static var mpsDataType: MPSDataType {
        return MPSDataType.float32
    }
}

extension Double: GPUNumeric {
    public static var gpuTypeIdentifier: String {
        return "DOUBLE_NOT_SUPPORTED"
    }
    
    public static var mpsDataType: MPSDataType {
        fatalError("Double not supported on GPU")
    }
}

extension Int32: GPUNumeric {
    public static var gpuTypeIdentifier: String {
        return "Int32"
    }
    
    public static var mpsDataType: MPSDataType {
        return MPSDataType.int32
    }
}
#else
// Stub protocol if Metal is unavailable
public protocol GPUNumeric {}
#endif
