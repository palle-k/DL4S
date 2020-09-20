//
//  VGG.swift
//  DL4S
//
//  Created by Palle Klewitz on 19.10.19.
//  Copyright (c) 2019 - 2020 - Palle Klewitz
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

public protocol VGGBase: LayerType where Parameter: RandomizableType {
    associatedtype Conv1: LayerType where Conv1.Parameter == Parameter, Conv1.Device == Device, Conv1.Inputs == Tensor<Parameter, Device>, Conv1.Outputs == Tensor<Parameter, Device>
    associatedtype Conv2: LayerType where Conv2.Parameter == Parameter, Conv2.Device == Device, Conv2.Inputs == Tensor<Parameter, Device>, Conv2.Outputs == Tensor<Parameter, Device>
    associatedtype Conv3: LayerType where Conv3.Parameter == Parameter, Conv3.Device == Device, Conv3.Inputs == Tensor<Parameter, Device>, Conv3.Outputs == Tensor<Parameter, Device>
    associatedtype Conv4: LayerType where Conv4.Parameter == Parameter, Conv4.Device == Device, Conv4.Inputs == Tensor<Parameter, Device>, Conv4.Outputs == Tensor<Parameter, Device>
    associatedtype Conv5: LayerType where Conv5.Parameter == Parameter, Conv5.Device == Device, Conv5.Inputs == Tensor<Parameter, Device>, Conv5.Outputs == Tensor<Parameter, Device>
    typealias DenseLayer = Sequential<Sequential<Sequential<Sequential<Dense<Self.Parameter, Self.Device>, BatchNorm<Self.Parameter, Self.Device>>, Sequential<Relu<Self.Parameter, Self.Device>, Dropout<Self.Parameter, Self.Device>>>, Sequential<Sequential<Dense<Self.Parameter, Self.Device>, BatchNorm<Self.Parameter, Self.Device>>, Sequential<Relu<Self.Parameter, Self.Device>, Dropout<Self.Parameter, Self.Device>>>>, Sequential<Dense<Self.Parameter, Self.Device>, LogSoftmax<Self.Parameter, Self.Device>>>
    
    var conv1: Conv1 { get set }
    var conv2: Conv2 { get set }
    var conv3: Conv3 { get set }
    var conv4: Conv4 { get set }
    var conv5: Conv5 { get set }
    var dense: DenseLayer { get set }
}

public extension VGGBase {
    var parameters: [Tensor<Parameter, Self.Device>] {
        get {
            Array([
                conv1.parameters,
                conv2.parameters,
                conv3.parameters,
                conv4.parameters,
                conv5.parameters,
                dense.parameters
            ].joined())
        }
    }
    
    var parameterPaths: [WritableKeyPath<Self, Tensor<Parameter, Device>>] {
        Array([
            conv1.parameterPaths.map((\Self.conv1).appending(path:)),
            conv2.parameterPaths.map((\Self.conv2).appending(path:)),
            conv3.parameterPaths.map((\Self.conv3).appending(path:)),
            conv4.parameterPaths.map((\Self.conv4).appending(path:)),
            conv5.parameterPaths.map((\Self.conv5).appending(path:)),
            dense.parameterPaths.map((\Self.dense).appending(path:))
        ].joined())
    }
    
    var isDropoutActive: Bool {
        get {
            dense.first.first.second.second.isActive || dense.first.second.second.second.isActive
        }
        set {
            dense.first.first.second.second.isActive = newValue
            dense.first.second.second.second.isActive = newValue
        }
    }
    
    func callAsFunction(_ inputs: Tensor<Parameter, Device>) -> Tensor<Parameter, Device> {
        var x = inputs
        x = conv1.callAsFunction(x)
        x = conv2.callAsFunction(x)
        x = conv3.callAsFunction(x)
        x = conv4.callAsFunction(x)
        x = conv5.callAsFunction(x)
        x = dense.callAsFunction(x.view(as: x.shape[0], -1))
        return x
    }
    
    static func makeDense(classes: Int) -> DenseLayer {
        Sequential {
            Dense<Parameter, Device>(inputSize: 512 * 6 * 6, outputSize: 4096)
            BatchNorm<Parameter, Device>(inputSize: [4096])
            Relu<Parameter, Device>()
            Dropout<Parameter, Device>(rate: 0.5)
            
            Dense<Parameter, Device>(inputSize: 4096, outputSize: 4096)
            BatchNorm<Parameter, Device>(inputSize: [4096])
            Relu<Parameter, Device>()
            Dropout<Parameter, Device>(rate: 0.5)
            
            Dense<Parameter, Device>(inputSize: 4096, outputSize: classes)
            LogSoftmax<Parameter, Device>()
        }
    }
}
