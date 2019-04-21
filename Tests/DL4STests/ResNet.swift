//
//  ResNet.swift
//  DL4STests
//
//  Created by Palle Klewitz on 21.04.19.
//

import XCTest
@testable import DL4S


class ResidualBlock<Element: RandomizableType, Device: DeviceType>: Layer {
    let conv1: Conv2D<Element, Device>
    let conv2: Conv2D<Element, Device>
    let bn1: BatchNorm<Element, Device>
    let bn2: BatchNorm<Element, Device>
    let downsample: Sequential<Element, Device>?
    
    var isTrainable: Bool = true
    
    var parameters: [Tensor<Element, Device>] {
        return Array([conv1.parameters, conv2.parameters, bn1.parameters, bn2.parameters, downsample?.parameters ?? []].joined())
    }
    
    init(inputShape: [Int], outPlanes: Int, downsample: Int) {
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
    
    func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
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


class ResNet<Element: RandomizableType, Device: DeviceType>: Layer {
    var isTrainable: Bool = true
    
    let start: Sequential<Element, Device>
    let l1: Sequential<Element, Device>
    let l2: Sequential<Element, Device>
    let l3: Sequential<Element, Device>
    let l4: Sequential<Element, Device>
    let end: Sequential<Element, Device>
    
    var parameters: [Tensor<Element, Device>] {
        return Array([
            start.parameters,
            l1.parameters,
            l2.parameters,
            l3.parameters,
            l4.parameters,
            end.parameters
            ].joined())
    }
    
    var trainableParameters: [Tensor<Element, Device>] {
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
    
    init(inputShape: [Int], classCount: Int) {
        let l1out = ConvUtil.outputShape(for: inputShape, kernelCount: 64, kernelWidth: 7, kernelHeight: 7, stride: 2, padding: 3)
        
        start = Sequential(
            Conv2D(inputChannels: inputShape[0], outputChannels: 64, kernelSize: 7, stride: 2, padding: 3).asAny(),
            BatchNorm(inputSize: l1out).asAny(),
            Relu().asAny()
        )
        l1 = ResNet<Element, Device>.makeResidualLayer(inputShape: l1out, outPlanes: 64, downsample: 1, blocks: 2)
        l2 = ResNet<Element, Device>.makeResidualLayer(inputShape: [64, l1out[1], l1out[2]], outPlanes: 128, downsample: 2, blocks: 2)
        l3 = ResNet<Element, Device>.makeResidualLayer(inputShape: [128, l1out[1] / 2, l1out[2] / 2], outPlanes: 256, downsample: 2, blocks: 2)
        l4 = ResNet<Element, Device>.makeResidualLayer(inputShape: [256, l1out[1] / 4, l1out[2] / 4], outPlanes: 512, downsample: 2, blocks: 2)
        end = Sequential(
            AdaptiveAvgPool2D(targetSize: 1).asAny(),
            Flatten().asAny(),
            Dense(inputFeatures: 512, outputFeatures: classCount).asAny(),
            Softmax().asAny()
        )
    }
    
    func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        var x = inputs[0]
        
        x = start(x)
        x = l1(x)
        x = l2(x)
        x = l3(x)
        x = l4(x)
        x = end(x)
        
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


class ResNetTests: XCTestCase {
    func testResNet() {
        let resnet = ResNet<Float, CPU>(inputShape: [3, 256, 256], classCount: 32)
        let optim = Adam(parameters: resnet.trainableParameters, learningRate: 0.001)
        optim.zeroGradient()
        
        let t = Tensor<Float, CPU>(repeating: 0, shape: 4, 3, 256, 256)
        t.tag = "input"
        Random.fill(t, a: 0, b: 1)
        
        let result = resnet(t)
        let expected = Tensor<Int32, CPU>([23, 15, 10, 6])
        
        let loss = categoricalCrossEntropy(expected: expected, actual: result)
        loss.backwards()
        optim.step()
    }
}
