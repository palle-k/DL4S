//
//  UtilOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.04.19.
//  Copyright (c) 2019 - Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

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


private struct ReverseContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    var symbol: String {
        return "reverse"
    }
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let srcGradient = source.shapedGradient, let vectorGradient = vector.shapedGradient else {
            return
        }
        Device.Engine.reverseAdd(values: vectorGradient, add: srcGradient, result: srcGradient)
    }
}

public func reverse<Element, Device>(_ vector: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let result = Tensor<Element, Device>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? ReverseContext(source: vector).asAny() : nil
    )
    
    Device.Engine.reverse(values: vector.shapedValues, result: result.shapedValues)
    
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
    
    func reversed() -> Tensor<Element, Device> {
        return DL4S.reverse(self)
    }
}
