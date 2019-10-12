//
//  XUnary.swift
//  DL4S
//
//  Created by Palle Klewitz on 04.10.19.
//

import Foundation


public extension XTensor {
    func exp() -> XTensor<Element, Device> {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.exp(values: values, result: resultBuffer)
        var result = XTensor(using: resultBuffer, context: nil)
        
        if requiresGradient {
            result.context = XTensorContext(
                tag: "Exponentiate",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient * result
                }]
            )
            result.requiresGradient = true
        }
        
        return result
    }
    
    func log() -> XTensor<Element, Device> {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.log(values: values, result: resultBuffer)
        var result = XTensor(using: resultBuffer, context: nil)
        
        if requiresGradient {
            result.context = XTensorContext(
                tag: "Logarithm",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient / self
                }]
            )
            result.requiresGradient = true
        }
        return result
    }
    
    func tanh() -> XTensor<Element, Device> {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.tanh(values: values, result: resultBuffer)
        var result = XTensor(using: resultBuffer, context: nil)
        
        if requiresGradient {
            result.context = XTensorContext(
                tag: "Tanh",
                sources: [self],
                backpropagate: [{ resultGradient in
                    (1 - result * result) * resultGradient
                }]
            )
            result.requiresGradient = true
        }
        return result
    }
    
    func sqrt() -> XTensor<Element, Device> {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.sqrt(values: values, result: resultBuffer)
        var result = XTensor(using: resultBuffer, context: nil)
        
        if requiresGradient {
            result.context = XTensorContext(
                tag: "SquareRoot",
                sources: [self],
                backpropagate: [{ resultGradient in
                    0.5 / result * resultGradient
                }]
            )
            result.requiresGradient = true
        }
        
        return result
    }
    
    func sigmoid() -> XTensor<Element, Device> {
        0.5 * (self * 0.5).tanh() + 0.5
    }
}
