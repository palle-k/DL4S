//
//  XUtilOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.10.19.
//

import Foundation

public extension XTensor where Element == Int32 {
    func oneHotEncoded<Target>(dim: Int, type: Target.Type = Target.self) -> XTensor<Target, Device> {
        var result = XTensor<Target, Device>(repeating: 0, shape: self.shape + [dim])
        
        for idx in iterate(self.shape) {
            let target = Int(self[idx].item)
            result[idx + [target]] = 1
        }
        
        return result
    }
}
