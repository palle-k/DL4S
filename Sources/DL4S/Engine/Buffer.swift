//
//  Buffer.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

import Foundation


public struct Buffer<Element: NumericType, DeviceType: Device>: Hashable {
    let memory: DeviceType.MemoryOperatorType.RawBufferType
    
    var count: Int {
        return DeviceType.MemoryOperatorType.getSize(of: self)
    }
    
    var pointee: Element {
        get {
            return DeviceType.MemoryOperatorType.getValue(from: self)
        }
        
        nonmutating set (newValue) {
            DeviceType.EngineType.fill(value: newValue, result: self, count: 1)
        }
    }
}
