//
//  Stack.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.04.19.
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

private struct StackAxisOperation<Element: NumericType, Device: DeviceType>: TensorOperation {
    var sourceTensors: [Tensor<Element, Device>]
    var axis: Int
    
    var symbol: String {
        return "stack"
    }
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        let sourceGradients = sourceTensors.compactMap {$0.shapedGradient}
        guard let vectorGradient = vector.shapedGradient, sourceGradients.count == sourceTensors.count else {
            return
        }
        
        Device.Engine.unstackAdd(stacked: vectorGradient, add: sourceGradients, result: sourceGradients, axis: axis)
    }
}

public func stack<Element, Device>(_ vectors: [Tensor<Element, Device>], axis: Int) -> Tensor<Element, Device> {
    precondition(0 ..< vectors[0].dim ~= axis, "Dimensionality of tensors must be in dimensionality range of source tensors")
    
    precondition(vectors.allSatisfy {
        $0.shape.count == vectors[0].shape.count &&
            zip($0.shape, vectors[0].shape)
                .enumerated()
                .allSatisfy {$0 == axis || $1.0 == $1.1}
    }, "All vector shapes must match except on concatenation axis.")
    
    let requiresGradient: Bool
    if vectors.contains(where: {$0.requiresGradient}) {
        // Backwards pass requires all source tensors to have gradient
        for vector in vectors {
            vector.requiresGradient = true
        }
        requiresGradient = true
    } else {
        requiresGradient = false
    }
    
    let resultStackDimCount = vectors
        .map {$0.shape[axis]}
        .reduce(0, +)
    
    var resultShape = vectors[0].shape
    resultShape[axis] = resultStackDimCount
    
    let result = Tensor<Element, Device>(
        shape: resultShape,
        parent: nil,
        context: requiresGradient ? StackAxisOperation(sourceTensors: vectors, axis: axis).asAny() : nil
    )
    
    Device.Engine.stack(buffers: vectors.map {$0.shapedValues}, result: result.shapedValues, axis: axis)
    
    return result
}

public func stack<Element, Device>(_ vectors: Tensor<Element, Device>...) -> Tensor<Element, Device> {
    return stack(vectors)
}

public func stack<Element, Device>(_ vectors: Tensor<Element, Device>..., axis: Int) -> Tensor<Element, Device> {
    return stack(vectors, axis: axis)
}


public extension Tensor {
    static func stack(_ tensors: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        return DL4S.stack(tensors)
    }
    
    static func stack(_ tensors: [Tensor<Element, Device>], axis: Int) -> Tensor<Element, Device> {
        return DL4S.stack(tensors, axis: axis)
    }
    
    static func stack(_ tensors: Tensor<Element, Device>...) -> Tensor<Element, Device> {
        return DL4S.stack(tensors)
    }
    
    static func stack(_ tensors: Tensor<Element, Device>..., axis: Int) -> Tensor<Element, Device> {
        return DL4S.stack(tensors, axis: axis)
    }
}
