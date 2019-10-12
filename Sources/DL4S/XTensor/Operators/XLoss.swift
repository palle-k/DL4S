//
//  XLoss.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
//

import Foundation

public func binaryCrossEntropy<Element: NumericType, Device: DeviceType>(expected: XTensor<Element, Device>, actual: XTensor<Element, Device>) -> XTensor<Element, Device> {
    let e = expected.view(as: [-1])
    let a = actual.view(as: [-1])
    
    let p1 = e * a.log()
    let p2 = (1 - e) * (1 - a).log()
    return (-(p1 + p2)).reduceMean()
}
