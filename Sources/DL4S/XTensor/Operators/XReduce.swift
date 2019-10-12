//
//  XReduce.swift
//  DL4S
//
//  Created by Palle Klewitz on 03.10.19.
//

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
                    tag: "ReduceSum(\(axes))",
                    sources: [self],
                    backpropagate: [{ resultGradient in
                        var broadcastShape = self.shape
                        
                        for a in axes {
                            broadcastShape[a] = 1
                        }
                        
                        return XTensor(repeating: 0, shape: self.shape) + resultGradient
                    }]
                )
            )
        } else {
            return XTensor(using: resultBuffer, context: nil)
        }
    }
    
    func reduceSum() -> XTensor<Element, Device> {
        reduceSum(along: Array(0 ..< dim))
    }
    
    func reduceMean(along axes: [Int]) -> XTensor<Element, Device> {
        reduceSum(along: axes) / XTensor(integerLiteral: axes.map {shape[$0]}.reduce(1, *))
    }
    
    func reduceMean() -> XTensor<Element, Device> {
        reduceMean(along: Array(0 ..< dim))
    }
}

//MARK: Min/Max
public extension XTensor {
    func reduceMax(along axes: [Int]) -> XTensor<Element, Device> {
        var resultShape: [Int] = shape
        for a in axes.reversed() {
            resultShape.remove(at: a)
        }
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        if requiresGradient {
            let contextBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Int32.self)
            Device.Engine.reduceMax(values: values, result: resultBuffer, context: contextBuffer, axes: axes)
            let context = XTensor<Int32, Device>(using: contextBuffer, context: nil)
            
            return XTensor(
                using: resultBuffer,
                context: XTensorContext(
                    tag: "ReduceMax(\(axes))",
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
    
    private func scatter(context: XTensor<Int32, Device>, axes: [Int], shape: [Int]) -> XTensor<Element, Device> {
        precondition(axes.count == 1, "Scatter is only available along a single axis.")
        let axis = axes.first!
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.expandContext(reduced: values, context: context.values, result: resultBuffer, axis: axis)
        return XTensor(
            using: resultBuffer,
            context: XTensorContext(
                tag: "Scatter",
                sources: [self],
                backpropagate: [{ resultGradient in
                    fatalError("Backpropagation is not available for Scatter operation.")
                }]
            )
        )
    }
}
