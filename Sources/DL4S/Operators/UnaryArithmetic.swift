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
    // return 1 / (1 + exp(-vector))
    // Using tanh for improved numeric stability
    return 0.5 * tanh(vector * 0.5) + 0.5
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


public extension Tensor {
    func exp() -> Tensor<Element, Device> {
        return DL4S.exp(self)
    }
    
    func log() -> Tensor<Element, Device> {
        return DL4S.log(self)
    }
    
    func sqrt() -> Tensor<Element, Device> {
        return DL4S.sqrt(self)
    }
    
    func log(base: Tensor<Element, Device>) -> Tensor<Element, Device> {
        return DL4S.log(self, base: base)
    }
    
    func tanh() -> Tensor<Element, Device> {
        return DL4S.tanh(self)
    }
    
    func sigmoid() -> Tensor<Element, Device> {
        return DL4S.sigmoid(self)
    }
    
    func relu() -> Tensor<Element, Device> {
        return DL4S.relu(self)
    }
    
    func leakyRelu(_ leak: Element) -> Tensor<Element, Device> {
        return DL4S.leakyRelu(self, leakage: leak)
    }
}
