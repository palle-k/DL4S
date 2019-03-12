//
//  UnaryOps.swift
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


private struct ExpContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        
        Device.Engine.vMA(lhs: vector.values, rhs: vectorGradient, add: sourceGradient, result: sourceGradient, count: source.count)
    }
    
    var symbol: String {
        return "exp"
    }
}

private struct LogContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        let temp = Device.Memory.allocateBuffer(withCapacity: vector.count, type: Element.self)
        
        Device.Engine.vDiv(lhs: vectorGradient, rhs: source.values, result: temp, count: source.count)
        Device.Engine.vAdd(lhs: temp, rhs: sourceGradient, result: sourceGradient, count: source.count)
        
        Device.Memory.free(temp)
    }
    
    var symbol: String {
        return "log"
    }
}

private struct TanhContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        let temp = Device.Memory.allocateBuffer(withCapacity: vector.count, type: Element.self)
        
        Device.Engine.vSquare(values: vector.values, result: temp, count: source.count)
        Device.Engine.vNeg(val: temp, result: temp, count: source.count)
        Device.Engine.vsAdd(lhs: temp, rhs: 1, result: temp, count: source.count)
        Device.Engine.vMA(lhs: temp, rhs: vectorGradient, add: sourceGradient, result: sourceGradient, count: source.count)
        
        Device.Memory.free(temp)
    }
    
    var symbol: String {
        return "tanh"
    }
}

private struct ReluContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        
        let temp1 = Device.Memory.allocateBuffer(withCapacity: vector.count, type: Element.self)
        Device.Engine.isPositive(val: source.values, result: temp1, count: source.count)
        Device.Engine.vMA(lhs: temp1, rhs: vectorGradient, add: sourceGradient, result: sourceGradient, count: source.count)
        Device.Memory.free(temp1)
    }
    
    var symbol: String {
        return "relu"
    }
}

private struct SqrtContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        let tmp = Device.Memory.allocateBuffer(withCapacity: vector.count, type: Element.self)
        
        // 1/2*1/sqrt(source)
        Device.Engine.svDiv(lhs: 0.5, rhs: vector.values, result: tmp, count: vector.count)
        Device.Engine.vMA(lhs: tmp, rhs: vectorGradient, add: sourceGradient, result: sourceGradient, count: source.count)
        Device.Memory.free(tmp)
    }
    
    var symbol: String {
        return "√"
    }
}


public func exp<Element, Device>(_ vector: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let result = Tensor<Element, Device>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? ExpContext(source: vector).asAny() : nil
    )
    
    Device.Engine.exp(val: vector.values, result: result.values, count: result.count)
    
    return result
}

public func log<Element, Device>(_ vector: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let result = Tensor<Element, Device>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? LogContext(source: vector).asAny() : nil
    )
    
    Device.Engine.log(val: vector.values, result: result.values, count: result.count)
    
    return result
}

public func sqrt<Element, Device>(_ vector: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let result = Tensor<Element, Device>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? SqrtContext(source: vector).asAny() : nil
    )
    
    Device.Engine.sqrt(val: vector.values, result: result.values, count: result.count)
    
    return result
}


public func log<Element, Device>(_ vector: Tensor<Element, Device>, base: Tensor<Element, Device>) -> Tensor<Element, Device> {
    return log(vector) / log(base)
}

public func tanh<Element, Device>(_ vector: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let result = Tensor<Element, Device>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? TanhContext(source: vector).asAny() : nil
    )
    
    Device.Engine.tanh(val: vector.values, result: result.values, count: result.count)
    
    return result
}

public func sigmoid<Element, Device>(_ vector: Tensor<Element, Device>) -> Tensor<Element, Device> {
    return 1 / (1 + exp(-vector))
}

public func relu<Element, Device>(_ vector: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let result = Tensor<Element, Device>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? ReluContext(source: vector).asAny() : nil
    )
    
    Device.Engine.relu(val: vector.values, result: result.values, count: result.count)
    
    return result
}

public func leakyRelu<Element, Device>(_ vector: Tensor<Element, Device>, leakage: Element) -> Tensor<Element, Device> {
    // -relu(-vector) = min(0, vector)
    return relu(vector) - Tensor(leakage) * relu(-vector)
}


public func binaryCrossEntropy<Element: NumericType, Device: DeviceType>(expected: Tensor<Element, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let e = expected.view(as: -1)
    let a = actual.view(as: -1)
    
    let p1 = e * log(a)
    let p2 = (1 - e) * log(1 - a)
    return mean(-(p1 + p2))
}

public func categoricalCrossEntropy<Element: NumericType, Device: DeviceType>(expected: Tensor<Int32, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let expectedView = expected.view(as: -1)
    let actualView = actual.view(as: expectedView.shape[0], -1)
    
    var result = Tensor<Element, Device>(0)
    
    for i in 0 ..< expectedView.shape[0] {
        // expected is not included in compute graph
        result = result - log(actualView[i, Int(expectedView[i].item)])
    }
    
    return result / Tensor<Element, Device>(Element(expected.count))
}

public func meanSquaredError<Element, Device>(expected: Tensor<Element, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let diff = expected - actual
    let s = sum(diff * diff)
    return s  / Tensor(Element(expected.dim > 1 ? expected.shape[0] : 1))
}

public func mean<Element, Device>(_ vector: Tensor<Element, Device>, axis: Int? = nil) -> Tensor<Element, Device> {
    let s = sum(vector, axis: axis)
    return s / Tensor(Element(axis.map {vector.shape[$0]} ?? vector.count))
}

public func variance<Element, Device>(_ vector: Tensor<Element, Device>, axis: Int? = nil) -> Tensor<Element, Device> {
    let m = mean(vector, axis: axis)
    return mean(vector * vector, axis: axis) - m * m
}

public func l2loss<Element, Device>(_ vector: Tensor<Element, Device>, loss: Element) -> Tensor<Element, Device> {
    return mean(vector * vector) * Tensor(loss)
}

public func l1loss<Element, Device>(_ vector: Tensor<Element, Device>, loss: Element) -> Tensor<Element, Device> {
    // abs(x) = sqrt(x * x)
    return mean(sqrt(vector * vector)) * Tensor(loss)
}


private struct SoftmaxContext<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        precondition(vector.dim == 2)
        
        let diag = Tensor<Element, Device>.diagonal(size: vector.shape[1], value: 1)
        let stride = vector.strides[0]
        let temp = Device.Memory.allocateBuffer(withCapacity: stride, type: Element.self)
        
        for i in 0 ..< vector.shape[0] {
            let filled = diag * vector[i]
            let outer = mmul(vector.view(as: -1, 1), vector.view(as: 1, -1))
            let jac = filled - outer
            
            Device.Engine.matMul(
                lhs: vectorGradient.advanced(by: i * stride),
                rhs: jac.values,
                result: temp,
                lhsRows: 1,
                lhsCols: jac.shape[0],
                rhsCols: jac.shape[1]
            )
            
            Device.Engine.vAdd(lhs: sourceGradient.advanced(by: stride * i), rhs: temp, result: sourceGradient.advanced(by: stride * i), count: stride)
        }
        
        Device.Memory.free(temp)
    }
    
    var symbol: String {
        return "softmax"
    }
}

public func softmax<Element, Device>(_ vector: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let result = Tensor<Element, Device>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? SoftmaxContext(source: vector).asAny() : nil
    )
    
    let (_, max) = Device.Engine.argmax(values: vector.values, count: result.count)
    Device.Engine.vsAdd(lhs: vector.values, rhs: -max, result: result.values, count: result.count)
    Device.Engine.exp(val: result.values, result: result.values, count: result.count)
    
    let stride = vector.strides[0]
    for i in 0 ..< vector.shape[0] {
        let sum = Device.Engine.sum(val: result.values.advanced(by: stride * i), count: stride)
        Device.Engine.vsMul(lhs: result.values.advanced(by: stride * i), rhs: 1 / sum, result: result.values.advanced(by: stride * i), count: stride)
    }
    
    return result
}
