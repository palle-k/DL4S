//
//  Buffer.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

import Foundation


public struct Buffer<Element: NumericType, Device: DeviceType>: Hashable {
    let memory: Device.Memory.RawBuffer
    
    var count: Int {
        return Device.Memory.getSize(of: self)
    }
    
    var pointee: Element {
        get {
            return Device.Memory.getValue(from: self)
        }
        
        nonmutating set (newValue) {
            Device.Engine.fill(value: newValue, result: self, count: 1)
        }
    }
    
    func advanced(by offset: Int) -> Buffer<Element, Device> {
        return Device.Memory.advance(buffer: self, by: offset)
    }
    
    subscript(index: Int) -> Element {
        get {
            return advanced(by: index).pointee
        }
        nonmutating set {
            advanced(by: index).pointee = newValue
        }
    }
}
