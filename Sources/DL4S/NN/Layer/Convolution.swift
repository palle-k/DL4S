//
//  Convolution.swift
//  DL4S
//
//  Created by Palle Klewitz on 17.10.19.
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

public struct Convolution2D<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[
        \.filters,
        \.bias
    ]}
    
    public var filters: Tensor<Element, Device>
    public var bias: Tensor<Element, Device>
    public let stride: Int
    public let padding: Int?
    
    public var parameters: [Tensor<Element, Device>] {
        get {
            [filters, bias]
        }
    }
    
    public init(inputChannels: Int, outputChannels: Int, kernelSize: (width: Int, height: Int), padding: Int? = nil, stride: Int = 1) {
        self.filters = Tensor(
            normalDistributedWithShape: [outputChannels, inputChannels, kernelSize.height, kernelSize.width],
            mean: 0,
            stdev: 2 / Element(kernelSize.height * kernelSize.width * inputChannels).sqrt(),
            requiresGradient: true
        )
        self.bias = Tensor(repeating: 0, shape: [1, outputChannels, 1, 1], requiresGradient: true)
        self.stride = stride
        self.padding = padding
        
        #if DEBUG
        filters.tag = "W"
        bias.tag = "b"
        #endif
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Conv2D") {
            inputs.convolved2d(filters: filters, padding: padding, stride: stride) + bias
        }
    }
}
