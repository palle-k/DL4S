//
//  ReductionArithmetic.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
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


private struct SumContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    let source: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.gradientItem, let sourceGradient = source.gradient else {
            return
        }
        Device.Engine.vsAdd(lhs: sourceGradient, rhs: vectorGradient, result: sourceGradient, count: source.count)
    }
    
    var symbol: String {
        return "Σ"
    }
}

public func sum<Element, Device>(_ vector: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let result = Tensor<Element, Device>(
        shape: [],
        parent: nil,
        context: vector.requiresGradient ? SumContext(source: vector).asAny() : nil
    )
    result.values.pointee = Device.Engine.sum(val: vector.values, count: vector.count)
    return result
}

private struct ReduceSumContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    var axes: [Int]
    
    var symbol: String {
        return "reduceSum"
    }
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        var broadcastShape = source.shape
        
        for a in axes {
            broadcastShape[a] = 1
        }
        
        guard let gradient = vector.shapedGradient?.reshaped(to: broadcastShape), let sourceGradient = source.shapedGradient else {
            return
        }
        
        Device.Engine.broadcastAdd(lhs: gradient, rhs: sourceGradient, result: sourceGradient)
    }
}

public func sum<Element, Device>(_ vector: Tensor<Element, Device>, axes: [Int]) -> Tensor<Element, Device> {
    if axes.isEmpty {
        return vector
    }
    
    var resultShape = vector.shape
    for a in axes.reversed() {
        resultShape.remove(at: a)
    }
    let result = Tensor<Element, Device>(
        shape: resultShape,
        parent: nil,
        context: ReduceSumContext(source: vector, axes: axes).asAny()
    )
    
    Device.Engine.reduceSum(values: vector.shapedValues, result: result.shapedValues, axes: axes)
    
    return result
}

private struct MaxContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    let maxI: Int
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradientItem else {
            return
        }
        sourceGradient[maxI] = sourceGradient[maxI] + vectorGradient
    }
    
    var symbol: String {
        return "max"
    }
}

private struct MaxAxisContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    let maxIdxs: [Int]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        
    }
    
    var symbol: String {
        return "max"
    }
}

public func max<Element, Device>(_ vector: Tensor<Element, Device>, axis: Int? = nil) -> Tensor<Element, Device> {
    if let axis = axis {
        var resultShape: [Int] = vector.shape
        resultShape.remove(at: axis)
        
        let result: Tensor<Element, Device> = Tensor(repeating: 0, shape: resultShape)
        var maxIdxs: [Int] = []
        
        for idx in iterate(resultShape) {
            var srcIdx: [Int?] = idx.map {$0}
            srcIdx.insert(nil, at: axis)
            
            //let slice = vector[srcIdx]
            let (slice, isCopy, _) = vector.buffer(from: srcIdx)
            
            let (arg, max) = Device.Engine.argmax(values: slice, count: slice.count)
            maxIdxs.append(arg)
            result.values[zip(idx, result.strides).map(*).reduce(0, +)] = max
            
            if isCopy {
                Device.Memory.free(slice)
            }
        }
        
        result.context = vector.requiresGradient ? MaxAxisContext(source: vector, maxIdxs: maxIdxs).asAny() : nil
        
        return result
    } else {
        let (arg, max) = Device.Engine.argmax(values: vector.values, count: vector.count)
        
        let result = Tensor<Element, Device>(max)
        result.context = vector.requiresGradient ? MaxContext(source: vector, maxI: arg).asAny() : nil
        return result
    }
}


public func argmax<Element, Device>(_ vector: Tensor<Element, Device>) -> Int {
    let (arg, _) = Device.Engine.argmax(values: vector.values, count: vector.count)
    return arg
}

private struct StackOperation<Element: NumericType, Device: DeviceType>: TensorOperation {
    var sourceTensors: [Tensor<Element, Device>]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        var offset = 0
        
        for src in sourceTensors {
            guard let srcGradient = src.gradient else {
                continue
            }
            let numElements = src.count
            Device.Engine.vAdd(lhs: srcGradient, rhs: vectorGradient.advanced(by: offset), result: srcGradient, count: numElements)
            offset += numElements
        }
    }
    
    var symbol: String {
        return "stack"
    }
}

public func stack<Element, Device>(_ vectors: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
    precondition(vectors.allSatisfy {$0.shape.dropFirst() == vectors[0].shape.dropFirst()}, "All vectors must have same shape except for first dimension")
    
    let resultShape = [vectors.reduce(0, {$0 + $1.shape[0]})] + Array(vectors[0].shape.dropFirst())
    
    let resultVector = Tensor<Element, Device>(
        shape: resultShape,
        parent: nil,
        context: vectors.contains(where: {$0.requiresGradient}) ? StackOperation(sourceTensors: vectors).asAny() : nil
    )
    
    var offset = 0
    
    for vector in vectors {
        let numElements = vector.count
        Device.Memory.assign(from: vector.values, to: resultVector.values.advanced(by: offset), count: numElements)
        offset += numElements
    }
    
    return resultVector
}

public func stack<Element, Device>(_ vectors: Tensor<Element, Device>...) -> Tensor<Element, Device> {
    return stack(vectors)
}
