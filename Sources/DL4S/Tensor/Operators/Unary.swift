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

// MARK: Element-wise operations

public extension Tensor {
    
    /// Element-wise exponentiates the tensor
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
    
    /// Computes the element-wise logarithm of the tensor.
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
    
    /// Computes the element-wise hyperbolic tangent of the tensor.
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
    
    /// Computes the element-wise square root of the tensor.
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
    
    /// Computes the element-wise heaviside step function of the tensor.
    ///
    /// The heaviside step function is defined as `value > 0 ? 1 : 0`
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
    
    /// Computes the element-wise relu function.
    ///
    /// The relu function is defined as `max(value, 0)`
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
    
    /// Computes the element-wise leaky relu function.
    ///
    /// The leaky relu function is defined as `max(value, leakage * value)`
    func leakyRectifiedLinear(leakage: Self) -> Self {
        rectifiedLinear() - leakage * (-self).rectifiedLinear()
    }
    
    /// Computes the element-wise sigmoid function.
    func sigmoid() -> Self {
        0.5 * (self * 0.5).tanh() + 0.5
    }
    
    /// Computes the softmax function along the given axis.
    /// If no axis is provided, the softmax is computed along axis 1.
    func softmax(axis: Int = 1) -> Self {
        let normalizer = detached().reduceMax(along: [axis]).unsqueezed(at: axis)
        let exponentiated = (self - normalizer).exp()
        return exponentiated / exponentiated.reduceSum(along: [axis]).unsqueezed(at: axis)
    }
    
    /// Computes the logarithm of the softmax function along the given axis.
    /// If no axis is provided, the softmax is computed along axis 1.
    func logSoftmax(axis: Int = 1) -> Self {
        let normalizer = detached().reduceMax(along: [axis]).unsqueezed(at: axis)
        let norm = self - normalizer
        let exponentiated = norm.exp()
        let logSumExp = exponentiated.reduceSum(along: [axis]).log().unsqueezed(at: axis)
        return norm - logSumExp
    }
    
    /// Computes the element-wise sine.
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
    
    /// Computes the element-wise cosine.
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
    
    /// Computes the element-wise GeLU activation
    ///
    /// See [Hendrycks, Gimpel - Gaussian Error Linear Units](https://arxiv.org/pdf/1606.08415.pdf)
    func gaussianErrorLinear() -> Self {
        self * (self * 1.702).sigmoid()
    }
    
    
    /// Computes the element-wise Swish activation
    ///
    /// See [Ramachandran et al. - Searching for Activation Functions](https://arxiv.org/pdf/1710.05941.pdf)
    func swishActivated(beta: Self = 1) -> Self {
        self * DL4S.sigmoid(beta * self)
    }
    
    
    /// Computes the element-wise Mish activation
    ///
    /// See [Diganta Misra - Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/pdf/1908.08681.pdf)
    func mishActivated() -> Self {
        self * DL4S.tanh(DL4S.log(1 + DL4S.exp(self)))
    }
    
    
    /// Computes the element-wise LiSHT activation
    ///
    /// See [Roy et al. - LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent Activation Function for Neural Networks](https://arxiv.org/pdf/1901.05894.pdf)
    func lishtActivated() -> Self {
        self * DL4S.tanh(self)
    }

}

/// Element-wise exponentiates the tensor.
public func exp<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.exp()
}

/// Computes the element-wise logarithm.
public func log<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.log()
}

/// Computes the element-wise square root.
public func sqrt<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.sqrt()
}

/// Computes the element-wise hyperbolic tangent.
public func tanh<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.tanh()
}

/// Computes the element-wise sigmoid function.
public func sigmoid<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.sigmoid()
}

/// Computes the element-wise sine.
public func sin<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.sine()
}

/// Computes the element-wise cosine.
public func cos<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.cosine()
}

/// Computes the element-wise relu function.
///
/// The relu function is defined as `max(value, 0)`
public func relu<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.rectifiedLinear()
}

/// Computes the element-wise leaky relu function.
///
/// The leaky relu function is defined as `max(value, leakage * value)`
public func leakyRelu<Element, Device>(_ tensor: Tensor<Element, Device>, leakage: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.leakyRectifiedLinear(leakage: leakage)
}

/// Computes the element-wise heaviside step function of the tensor.
///
/// The heaviside step function is defined as `value > 0 ? 1 : 0`
public func heaviside<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.heaviside()
}

/// Computes the softmax function along the given axis.
/// If no axis is provided, the softmax is computed along axis 1.
public func softmax<Element, Device>(_ tensor: Tensor<Element, Device>, axis: Int = 1) -> Tensor<Element, Device> {
    tensor.softmax(axis: axis)
}

/// Computes the logarithm of the softmax function along the given axis.
/// If no axis is provided, the log softmax is computed along axis 1.
public func logSoftmax<Element, Device>(_ tensor: Tensor<Element, Device>, axis: Int = 1) -> Tensor<Element, Device> {
    tensor.logSoftmax(axis: axis)
}

/// Computes the element-wise GeLU activation
///
/// See [Hendrycks, Gimpel - Gaussian Error Linear Units](https://arxiv.org/pdf/1606.08415.pdf)
func gelu<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.gaussianErrorLinear()
}


/// Computes the element-wise Swish activation
///
/// See [Ramachandran et al. - Searching for Activation Functions](https://arxiv.org/pdf/1710.05941.pdf)
func swishActivated<Element, Device>(_ tensor: Tensor<Element, Device>, beta: Tensor<Element, Device> = 1) -> Tensor<Element, Device> {
    tensor.swishActivated(beta: beta)
}


/// Computes the element-wise Mish activation
///
/// See [Diganta Misra - Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/pdf/1908.08681.pdf)
func mishActivated<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.mishActivated()
}


/// Computes the element-wise LiSHT activation
///
/// See [Roy et al. - LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent Activation Function for Neural Networks](https://arxiv.org/pdf/1901.05894.pdf)
func lishtActivated<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.lishtActivated()
}
