//
//  XReduce.swift
//  DL4S
//
//  Created by Palle Klewitz on 03.10.19.
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


//MARK: Summation
public extension XTensor {
    func reduceSum(along axes: [Int]) -> XTensor<Element, Device> {
        if axes.isEmpty {
            return self
        }
        
        var resultShape = shape
        for a in axes.reversed() {
            resultShape.remove(at: a)
        }
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.reduceSum(values: self.values, result: resultBuffer, axes: axes)
        
        if requiresGradient {
            return XTensor(
                using: resultBuffer,
                context: XTensorContext(
                    tag: "sum\(axes)",
                    sources: [self],
                    backpropagateAccumulate: [{ resultGradient, acc in
                        var broadcastShape = self.shape
                        
                        for a in axes {
                            broadcastShape[a] = 1
                        }
                        
                        return (acc ?? XTensor(repeating: 0, shape: self.shape)) + resultGradient.view(as: broadcastShape)
                    }]
                )
            )
        } else {
            return XTensor(using: resultBuffer, context: nil)
        }
    }
    
    @inline(__always)
    func reduceSum(along axes: Int...) -> Self {
        reduceSum(along: axes)
    }
    
    func reduceSum() -> Self {
        reduceSum(along: Array(0 ..< dim))
    }
    
    func reduceMean(along axes: [Int]) -> Self {
        reduceSum(along: axes) / XTensor(integerLiteral: axes.map {shape[$0]}.reduce(1, *))
    }
    
    func reduceMean(along axes: Int...) -> Self {
        reduceMean(along: axes)
    }
    
    func reduceMean() -> Self {
        reduceMean(along: Array(0 ..< dim))
    }
    
    func variance(along axes: [Int]) -> Self {
        let m = self.reduceMean(along: axes)
        return (self * self).reduceMean(along: axes) - m * m
    }
    
    func variance(along axes: Int...) -> Self {
        variance(along: axes)
    }
    
    func variance() -> Self {
        variance(along: Array(0 ..< dim))
    }
    
    func argmax() -> Int {
        Device.Engine.argmax(values: values.values, count: count).0
    }
}

public func sum<Element, Device>(_ tensor: XTensor<Element, Device>) -> XTensor<Element, Device> {
    tensor.reduceSum()
}

public func sum<Element, Device>(_ tensor: XTensor<Element, Device>, axes: [Int]) -> XTensor<Element, Device> {
    tensor.reduceSum(along: axes)
}

public func sum<Element, Device>(_ tensor: XTensor<Element, Device>, axes: Int...) -> XTensor<Element, Device> {
    tensor.reduceSum(along: axes)
}

public func mean<Element, Device>(_ tensor: XTensor<Element, Device>) -> XTensor<Element, Device> {
    tensor.reduceMean()
}

public func mean<Element, Device>(_ tensor: XTensor<Element, Device>, axes: [Int]) -> XTensor<Element, Device> {
    tensor.reduceMean(along: axes)
}

public func mean<Element, Device>(_ tensor: XTensor<Element, Device>, axes: Int...) -> XTensor<Element, Device> {
    tensor.reduceMean(along: axes)
}

public func variance<Element, Device>(_ tensor: XTensor<Element, Device>) -> XTensor<Element, Device> {
    tensor.variance()
}

public func variance<Element, Device>(_ tensor: XTensor<Element, Device>, axes: [Int]) -> XTensor<Element, Device> {
    tensor.variance(along: axes)
}

public func variance<Element, Device>(_ tensor: XTensor<Element, Device>, axes: Int...) -> XTensor<Element, Device> {
    tensor.variance(along: axes)
}

//MARK: Min/Max
public extension XTensor {
    func reduceMax(along axes: [Int]) -> Self {
        var resultShape: [Int] = shape
        for a in axes.reversed() {
            resultShape.remove(at: a)
        }
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        if requiresGradient {
            let contextBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Int32.self)
            if let axis = axes.first, axes.count == 1 {
                Device.Engine.reduceMax(values: values, result: resultBuffer, context: contextBuffer, axis: axis)
            } else {
                Device.Engine.reduceMax(values: values, result: resultBuffer, context: contextBuffer, axes: axes)
            }
            
            let context = XTensor<Int32, Device>(using: contextBuffer, context: nil)
            
            return XTensor(
                using: resultBuffer,
                context: XTensorContext(
                    tag: "max\(axes)",
                    sources: [self],
                    backpropagate: [{ resultGradient in
                        resultGradient.scatter(context: context, axes: axes, shape: self.shape)
                    }]
                )
            )
        } else {
            Device.Engine.reduceMax(values: values, result: resultBuffer, context: nil, axes: axes)
            
            return XTensor(using: resultBuffer, context: nil)
        }
    }
    
    func reduceMax(along axes: Int...) -> Self {
        reduceMax(along: axes)
    }
    
    func reduceMax() -> Self {
        reduceMax(along: Array(0 ..< dim))
    }
    
    private func scatter(context: XTensor<Int32, Device>, axes: [Int], shape: [Int]) -> Self {
        precondition(axes.count == 1, "Scatter is only available along a single axis.")
        let axis = axes.first!
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.expandContext(reduced: values, context: context.values, result: resultBuffer, axis: axis)
        return XTensor(
            using: resultBuffer,
            context: XTensorContext(
                tag: "scatter",
                sources: [self],
                backpropagate: [{ resultGradient in
                    fatalError("Backpropagation is not available for Scatter operation.")
                }]
            )
        )
    }
}
