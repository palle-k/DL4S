//
//  Conv.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.04.19.
//

import Foundation


public class Conv2D<Element: RandomizableType, Device: DeviceType>: Layer {
    public var isTrainable: Bool = true
    
    public let filters: Tensor<Element, Device>
    public let bias: Tensor<Element, Device>
    
    public let padding: Int
    public let stride: Int
    
    public var parameters: [Tensor<Element, Device>] {
        return [filters, bias]
    }
    
    public init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int? = nil) {
        self.padding = padding ?? ((kernelSize - 1) / 2)
        self.stride = stride
        
        self.filters = Tensor<Element, Device>(repeating: 0, shape: outputChannels, inputChannels, kernelSize, kernelSize, requiresGradient: true)
        self.bias = Tensor<Element, Device>(repeating: 0, shape: 1, outputChannels, 1, 1)
        
        Random.fillNormal(self.filters, mean: 0, stdev: (2 / Element(kernelSize * kernelSize * inputChannels)).sqrt())
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        return conv2d(images: inputs[0], filters: filters, padding: padding, stride: stride) + bias
    }
}

