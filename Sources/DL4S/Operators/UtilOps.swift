//
//  UtilOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.04.19.
//

import Foundation


public func `repeat`<Element, Device>(_ tensor: Tensor<Element, Device>, count: Int) -> Tensor<Element, Device> {
    return stack(Array(repeating: tensor, count: count))
}

public func arange<Element, Device>(lowerBound: Element = 0, upperBound: Element, by stride: Element = 1) -> Tensor<Element, Device> {
    let result = Tensor<Element, Device>(repeating: 0, shape: ((upperBound - lowerBound) / stride).toInt())
    Device.Engine.arange(lowerBound: lowerBound, upperBound: upperBound, result: result.shapedValues)
    return result
}

public func pad<Element, Device>(_ tensor: Tensor<Element, Device>, padding: [(Int, Int)], value: Element = 0) -> Tensor<Element, Device> {
    precondition(padding.count == tensor.dim)
    
    let result = Tensor<Element, Device>(repeating: value, shape: zip(tensor.shape, padding).map {$0 + $1.0 + $1.1})
    let index = zip(tensor.shape, padding).map {$1.0 ..< ($0 + $1.0)}
    result[index] = tensor
    
    return result
}

public func pad<Element, Device>(_ tensor: Tensor<Element, Device>, padding: [Int], value: Element = 0) -> Tensor<Element, Device> {
    precondition(padding.count == tensor.dim)
    
    let result = Tensor<Element, Device>(repeating: value, shape: zip(tensor.shape, padding).map {$0 + $1 * 2})
    let index = zip(tensor.shape, padding).map {$1 ..< ($0 + $1)}
    result[index] = tensor
    
    return result
}

public extension Tensor {
    static func repeating(_ tensor: Tensor<Element, Device>, count: Int) -> Tensor<Element, Device> {
        return `repeat`(tensor, count: count)
    }
    
    static func arange(lowerBound: Element = 0, upperBound: Element, by stride: Element = 1) -> Tensor<Element, Device> {
        return DL4S.arange(lowerBound: lowerBound, upperBound: upperBound, by: stride)
    }
    
    func padded(_ padding: [(Int, Int)], value: Element = 0) -> Tensor<Element, Device> {
        return DL4S.pad(self, padding: padding, value: value)
    }
    
    func padded(_ padding: [Int], value: Element = 0) -> Tensor<Element, Device> {
        return DL4S.pad(self, padding: padding, value: value)
    }
}
