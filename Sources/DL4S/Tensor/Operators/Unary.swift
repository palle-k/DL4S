//
//  Unary.swift
//  DL4S
//
//  Created by Palle Klewitz on 04.10.19.
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


public extension Tensor {
    func exp() -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.exp(values: values, result: resultBuffer)
        var result = Tensor(using: resultBuffer, context: nil)
        
        if requiresGradient {
            let resultCopy = result
            result.context = TensorContext(
                tag: "exp",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient * resultCopy
                }]
            )
            result.requiresGradient = true
        }
        
        return result
    }
    
    func log() -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.log(values: values, result: resultBuffer)
        var result = Tensor(using: resultBuffer, context: nil)
        
        if requiresGradient {
            result.context = TensorContext(
                tag: "log",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient / self
                }]
            )
            result.requiresGradient = true
        }
        return result
    }
    
    func tanh() -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.tanh(values: values, result: resultBuffer)
        var result = Tensor(using: resultBuffer, context: nil)
        
        if requiresGradient {
            let resultCopy = result
            result.context = TensorContext(
                tag: "tanh",
                sources: [self],
                backpropagate: [{ resultGradient in
                    (1 - resultCopy * resultCopy) * resultGradient
                }]
            )
            result.requiresGradient = true
        }
        return result
    }
    
    func sqrt() -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.sqrt(values: values, result: resultBuffer)
        var result = Tensor(using: resultBuffer, context: nil)
        
        if requiresGradient {
            let resultCopy = result
            result.context = TensorContext(
                tag: "sqrt",
                sources: [self],
                backpropagate: [{ resultGradient in
                    0.5 / resultCopy * resultGradient
                }]
            )
            result.requiresGradient = true
        }
        
        return result
    }
    
    func heaviside() -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.heaviside(values: values, result: resultBuffer)
        
        var result = Tensor(using: resultBuffer, context: nil)
        
        if requiresGradient {
            result.context = TensorContext(
                tag: "heaviside",
                sources: [self],
                backpropagate: [{ resultGradient in
                    Tensor(repeating: 0, shape: resultGradient.shape)
                }]
            )
            result.requiresGradient = true
        }
        
        return result
    }
    
    func rectifiedLinear() -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.relu(values: values, result: resultBuffer)
        
        var result = Tensor(using: resultBuffer, context: nil)
        
        if requiresGradient {
            result.context = TensorContext(
                tag: "relu",
                sources: [self],
                backpropagate: [{ resultGradient in
                    OperationGroup.capture(named: "RectifiedLinearGrad") {
                        self.heaviside() * resultGradient
                    }
                }]
            )
            result.requiresGradient = true
        }
        
        return result
    }
    
    func leakyRectifiedLinear(leakage: Self) -> Self {
        rectifiedLinear() - leakage * (-self).rectifiedLinear()
    }
    
    func sigmoid() -> Self {
        0.5 * (self * 0.5).tanh() + 0.5
    }
    
    func softmax(axis: Int = 1) -> Self {
        let normalizer = detached().reduceMax(along: [axis]).unsqueezed(at: axis)
        let exponentiated = (self - normalizer).exp()
        return exponentiated / exponentiated.reduceSum(along: [axis]).unsqueezed(at: axis)
    }
    
    func sine() -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.sin(values: values, result: resultBuffer)
        var result = Tensor(using: resultBuffer, context: nil)
        if requiresGradient {
            result.context = TensorContext(
                tag: "sin",
                sources: [self],
                backpropagate: [{ resultGradient in
                    self.cosine() * resultGradient
                }]
            )
            result.requiresGradient = true
        }
        return result
    }
    
    func cosine() -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.sin(values: values, result: resultBuffer)
        var result = Tensor(using: resultBuffer, context: nil)
        if requiresGradient {
            result.context = TensorContext(
                tag: "cos",
                sources: [self],
                backpropagate: [{ resultGradient in
                    -self.sine() * resultGradient
                }]
            )
            result.requiresGradient = true
        }
        return result
    }
}

public func exp<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.exp()
}

public func log<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.log()
}

public func sqrt<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.sqrt()
}

public func tanh<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.tanh()
}

public func sigmoid<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.sigmoid()
}

public func sin<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.sine()
}

public func cos<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.cosine()
}

public func relu<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.rectifiedLinear()
}

public func leakyRelu<Element, Device>(_ tensor: Tensor<Element, Device>, leakage: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.leakyRectifiedLinear(leakage: leakage)
}

public func heaviside<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.heaviside()
}

public func softmax<Element, Device>(_ tensor: Tensor<Element, Device>, axis: Int = 1) -> Tensor<Element, Device> {
    tensor.softmax(axis: axis)
}
