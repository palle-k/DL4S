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

public extension XTensor {
    init(linearRampWithLowerBound lowerBound: Element = 0, upperBound: Element, by stride: Element = 1) {
        let buffer = Device.Memory.allocateBuffer(withShape: [((upperBound - lowerBound) / stride).toInt()], type: Element.self)
        Device.Engine.arange(lowerBound: lowerBound, upperBound: upperBound, result: buffer)
        self.init(using: buffer, context: nil)
    }
    
    func repeated(_ times: Int) -> XTensor<Element, Device> {
        return XTensor(stacking: Array(repeating: self, count: count))
    }

    func padded(with value: Element = 0, padding: [(Int, Int)]) -> XTensor<Element, Device> {
        precondition(padding.count == dim)
        
        var result = XTensor<Element, Device>(repeating: value, shape: zip(shape, padding).map {$0 + $1.0 + $1.1})
        let index = zip(shape, padding).map {$1.0 ..< ($0 + $1.0)}
        result[index] = self
        
        return result
    }

    func padded(with value: Element = 0, padding: [Int]) -> XTensor<Element, Device> {
        precondition(padding.count == dim)
        
        var result = XTensor<Element, Device>(repeating: value, shape: zip(shape, padding).map {$0 + $1 * 2})
        let index = zip(shape, padding).map {$1 ..< ($0 + $1)}
        result[index] = self
        
        return result
    }
    
    func reversed() -> XTensor<Element, Device> {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.reverse(values: values, result: resultBuffer)
        
        return XTensor(
            using: resultBuffer,
            context: requiresGradient ? XTensorContext(
                tag: "Reverse",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.reversed()
                }]
            ) : nil
        )
    }

}
