//
//  ResNet.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.04.19.
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


public class ResidualBlock<Element: RandomizableType, Device: DeviceType>: Layer {
    let conv1: Conv2D<Element, Device>
    let conv2: Conv2D<Element, Device>
    let bn1: BatchNorm<Element, Device>
    let bn2: BatchNorm<Element, Device>
    let downsample: Sequential<Element, Device>?
    
    public var isTrainable: Bool = true
    
    public var parameters: [Tensor<Element, Device>] {
        return Array([conv1.parameters, conv2.parameters, bn1.parameters, bn2.parameters, downsample?.parameters ?? []].joined())
    }
    
    public init(inputShape: [Int], outPlanes: Int, downsample: Int) {
        conv1 = Conv2D(inputChannels: inputShape[0], outputChannels: outPlanes, kernelSize: 3, stride: downsample)
        conv2 = Conv2D(inputChannels: outPlanes, outputChannels: outPlanes, kernelSize: 3)
        
        let convShape = ConvUtil.outputShape(for: inputShape, kernelCount: outPlanes, kernelWidth: 3, kernelHeight: 3, stride: downsample, padding: 1)
        
        bn1 = BatchNorm(inputSize: convShape)
        bn2 = BatchNorm(inputSize: convShape)
        
        if downsample != 1 {
            self.downsample = Sequential(
                Conv2D(inputChannels: inputShape[0], outputChannels: outPlanes, kernelSize: 1, stride: downsample, padding: 0).asAny(),
                BatchNorm(inputSize: [outPlanes, inputShape[1] / downsample, inputShape[2] / downsample]).asAny()
            )
        } else {
            self.downsample = nil
        }
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        var x = inputs[0]
        let res = downsample?(x) ?? x
        
        x = conv1(x)
        x = bn1(x)
        x = relu(x)
        x = conv2(x)
        x = bn2(x)
        x = x + res
        x = relu(x)
        
        return x
    }
}


public class ResNet<Element: RandomizableType, Device: DeviceType>: Layer {
    public var isTrainable: Bool = true
    
    let start: Sequential<Element, Device>
    let l1: Sequential<Element, Device>
    let l2: Sequential<Element, Device>
    let l3: Sequential<Element, Device>
    let l4: Sequential<Element, Device>
    let end: Sequential<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        return Array([
            start.parameters,
            l1.parameters,
            l2.parameters,
            l3.parameters,
            l4.parameters,
            end.parameters
            ].joined())
    }
    
    public var trainableParameters: [Tensor<Element, Device>] {
        guard isTrainable else {
            return []
        }
        return Array([
            start.trainableParameters,
            l1.trainableParameters,
            l2.trainableParameters,
            l3.trainableParameters,
            l4.trainableParameters,
            end.trainableParameters
        ].joined())
    }
    
    public init(inputShape: [Int], classCount: Int) {
        let startOut = ConvUtil.outputShape(for: inputShape, kernelCount: 64, kernelWidth: 7, kernelHeight: 7, stride: 2, padding: 3)
        
        start = Sequential(
            Conv2D(inputChannels: inputShape[0], outputChannels: 64, kernelSize: 7, stride: 2, padding: 3).asAny(),
            BatchNorm(inputSize: startOut).asAny(),
            Relu().asAny()
        )
        l1 = ResNet<Element, Device>.makeResidualLayer(inputShape: startOut, outPlanes: 64, downsample: 1, blocks: 2)
        l2 = ResNet<Element, Device>.makeResidualLayer(inputShape: [64, startOut[1], startOut[2]], outPlanes: 128, downsample: 2, blocks: 2)
        l3 = ResNet<Element, Device>.makeResidualLayer(inputShape: [128, startOut[1] / 2, startOut[2] / 2], outPlanes: 256, downsample: 2, blocks: 2)
        l4 = ResNet<Element, Device>.makeResidualLayer(inputShape: [256, startOut[1] / 4, startOut[2] / 4], outPlanes: 512, downsample: 2, blocks: 2)
        end = Sequential(
            AdaptiveAvgPool2D(targetSize: 1).asAny(),
            Flatten().asAny(),
            Dense(inputFeatures: 512, outputFeatures: classCount).asAny(),
            Softmax().asAny()
        )
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        var x = inputs[0]
        
        x = start(x)
        print(x.shape)
        x = l1(x)
        print(x.shape)
        x = l2(x)
        print(x.shape)
        x = l3(x)
        print(x.shape)
        x = l4(x)
        print(x.shape)
        x = end(x)
        print(x.shape)
        
        return x
    }
    
    private static func makeResidualLayer(inputShape: [Int], outPlanes: Int, downsample: Int, blocks: Int) -> Sequential<Element, Device> {
        let s = Sequential<Element, Device>()
        
        s.append(ResidualBlock(inputShape: inputShape, outPlanes: outPlanes, downsample: downsample))
        
        for _ in 1 ..< blocks {
            s.append(ResidualBlock(inputShape: [outPlanes, inputShape[1] / downsample, inputShape[2] / downsample], outPlanes: outPlanes, downsample: 1))
        }
        
        return s
    }
}
