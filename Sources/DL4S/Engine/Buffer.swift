//
//  Buffer.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

import Foundation


struct Buffer<Element: NumericType, DeviceType: Device> {
    let memory: DeviceType.AllocatorType.RawBufferType
}
