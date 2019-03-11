//
//  Norm.swift
//  DL4S
//
//  Created by Palle Klewitz on 01.03.19.
//

import Foundation


public class BatchNorm<Element: NumericType, Device: DeviceType>: Layer, Codable {
    
    public typealias Input = Element
    
    public var trainable: Bool = true
    
    public var parameters: [Tensor<Element, Device>] {
        return [shift, scale]
    }
    
    let shift: Tensor<Element, Device>
    let scale: Tensor<Element, Device>
    
    var runningMean: Tensor<Element, Device>
    var runningVar: Tensor<Element, Device>
    
    public var momentum: Element
    
    public init(inputSize: Int, momentum: Element = 0.1) {
        shift = Tensor(repeating: 0, shape: inputSize, requiresGradient: true)
        scale = Tensor(repeating: 1, shape: inputSize, requiresGradient: true)
        
        runningMean = Tensor(repeating: 0, shape: inputSize)
        runningVar = Tensor(repeating: 1, shape: inputSize)
        
        self.momentum = momentum
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        let x = inputs[0]
        
//        if self.trainable {
//            runningMean = Tensor(momentum) * runningMean + Tensor(1 - momentum) * mean(x, axis: 0)
//            runningVar = Tensor(momentum) * runningVar + Tensor(1 - momentum) * variance(x, axis: 0)
//        }
        
        let normalized = (x - mean(x, axis: 0)) / sqrt(variance(x, axis: 0))
        return normalized * scale + shift
    }
}
