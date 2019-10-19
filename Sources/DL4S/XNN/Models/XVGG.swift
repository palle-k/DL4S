//
//  XVGG.swift
//  DL4S
//
//  Created by Palle Klewitz on 19.10.19.
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

public protocol VGGBase: XLayer where Parameter: RandomizableType {
    associatedtype Conv1: XLayer where Conv1.Parameter == Parameter, Conv1.Device == Device
    associatedtype Conv2: XLayer where Conv2.Parameter == Parameter, Conv2.Device == Device, Conv2.Inputs == Conv1.Outputs
    associatedtype Conv3: XLayer where Conv3.Parameter == Parameter, Conv3.Device == Device, Conv3.Inputs == Conv2.Outputs
    associatedtype Conv4: XLayer where Conv4.Parameter == Parameter, Conv4.Device == Device, Conv4.Inputs == Conv3.Outputs
    associatedtype Conv5: XLayer where Conv5.Parameter == Parameter, Conv5.Device == Device, Conv5.Inputs == Conv4.Outputs, Conv5.Outputs == XSequential<
        XSequential<
            XSequential<XDense<Parameter, Device>, XRelu<Parameter, Device>>,
            XSequential<XDropout<Parameter, Device>, XDense<Parameter, Device>>>,
        XSequential<
            XSequential<XRelu<Parameter, Device>, XDropout<Parameter, Device>>,
            XSequential<XDense<Parameter, Device>, XSoftmax<Parameter, Device>>>>.Inputs
    typealias Dense = XSequential<
        XSequential<
            XSequential<XDense<Parameter, Device>, XRelu<Parameter, Device>>,
            XSequential<XDropout<Parameter, Device>, XDense<Parameter, Device>>>,
        XSequential<
            XSequential<XRelu<Parameter, Device>, XDropout<Parameter, Device>>,
            XSequential<XDense<Parameter, Device>, XSoftmax<Parameter, Device>>>>
    
    var conv1: Conv1 { get set }
    var conv2: Conv2 { get set }
    var conv3: Conv3 { get set }
    var conv4: Conv4 { get set }
    var conv5: Conv5 { get set }
    var dense: Dense { get set }
}

public extension VGGBase {
    var parameters: [XTensor<Parameter, Self.Device>] {
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
    
    var parameterPaths: [WritableKeyPath<Self, XTensor<Parameter, Device>>] {
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
            dense.first.second.first.isActive || dense.second.first.second.isActive
        }
        set {
            dense.first.second.first.isActive = newValue
            dense.second.first.second.isActive = newValue
        }
    }
    
    func callAsFunction(_ inputs: Conv1.Inputs) -> Dense.Outputs {
        let c1 = conv1.callAsFunction(inputs)
        let c2 = conv2.callAsFunction(c1)
        let c3 = conv3.callAsFunction(c2)
        let c4 = conv4.callAsFunction(c3)
        let c5 = conv5.callAsFunction(c4)
        let d = dense.callAsFunction(c5)
        return d
    }
    
    static func makeDense(classes: Int) -> Dense {
        XSequential {
            XDense<Parameter, Device>(inputSize: 512 * 7 * 7, outputSize: 4096)
            XRelu<Parameter, Device>()
            XDropout<Parameter, Device>(rate: 0.5)
            
            XDense<Parameter, Device>(inputSize: 4096, outputSize: 4096)
            XRelu<Parameter, Device>()
            XDropout<Parameter, Device>(rate: 0.5)
            
            XDense<Parameter, Device>(inputSize: 4096, outputSize: classes)
            XSoftmax<Parameter, Device>()
        }
    }
}

public struct VGG11<E: RandomizableType, D: DeviceType>: VGGBase {
    public typealias Parameter = E
    public typealias Device = D
    
    public var conv1: XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XMaxPool2D<E, D>>
    public var conv2: XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XMaxPool2D<E, D>>
    public var conv3: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var conv4: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var conv5: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var dense: XSequential<XSequential<XSequential<XDense<E, D>, XRelu<E, D>>, XSequential<XDropout<E, D>, XDense<E, D>>>, XSequential<XSequential<XRelu<E, D>, XDropout<E, D>>, XSequential<XDense<E, D>, XSoftmax<E, D>>>>

    public init(inputChannels: Int, classes: Int) {
        conv1 = XSequential {
            XConvolution2D<E, D>(inputChannels: inputChannels, outputChannels: 64, kernelSize: (3, 3))
            XRelu<E, D>()
            XMaxPool2D<E, D>()
        }
        
        conv2 = XSequential {
            XConvolution2D<E, D>(inputChannels: 64, outputChannels: 128, kernelSize: (3, 3))
            XRelu<E, D>()
            XMaxPool2D<E, D>()
        }
        
        conv3 = XSequential {
            XConvolution2D<E, D>(inputChannels: 128, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv4 = XSequential {
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv5 = XSequential {
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        dense = Self.makeDense(classes: classes)
    }
}

public struct VGG13<E: RandomizableType, D: DeviceType>: VGGBase {
    public typealias Parameter = E
    public typealias Device = D
    
    public var conv1: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var conv2: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var conv3: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var conv4: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var conv5: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var dense: XSequential<XSequential<XSequential<XDense<E, D>, XRelu<E, D>>, XSequential<XDropout<E, D>, XDense<E, D>>>, XSequential<XSequential<XRelu<E, D>, XDropout<E, D>>, XSequential<XDense<E, D>, XSoftmax<E, D>>>>
    
    public init(inputChannels: Int, classes: Int) {
        conv1 = XSequential {
            XConvolution2D<E, D>(inputChannels: inputChannels, outputChannels: 64, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 64, outputChannels: 64, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv2 = XSequential {
            XConvolution2D<E, D>(inputChannels: 64, outputChannels: 128, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 128, outputChannels: 128, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv3 = XSequential {
            XConvolution2D<E, D>(inputChannels: 128, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv4 = XSequential {
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv5 = XSequential {
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        dense = Self.makeDense(classes: classes)
    }
}

public struct VGG16<E: RandomizableType, D: DeviceType>: VGGBase {
    public typealias Parameter = E
    public typealias Device = D
    
    public var conv1: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var conv2: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var conv3: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XSequential<XConvolution2D<E, D>, XRelu<E, D>>>, XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XMaxPool2D<E, D>>>
    public var conv4: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XSequential<XConvolution2D<E, D>, XRelu<E, D>>>, XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XMaxPool2D<E, D>>>
    public var conv5: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XSequential<XConvolution2D<E, D>, XRelu<E, D>>>, XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XMaxPool2D<E, D>>>
    public var dense: XSequential<XSequential<XSequential<XDense<E, D>, XRelu<E, D>>, XSequential<XDropout<E, D>, XDense<E, D>>>, XSequential<XSequential<XRelu<E, D>, XDropout<E, D>>, XSequential<XDense<E, D>, XSoftmax<E, D>>>>

    public init(inputChannels: Int, classes: Int) {
        conv1 = XSequential {
            XConvolution2D<E, D>(inputChannels: inputChannels, outputChannels: 64, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 64, outputChannels: 64, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv2 = XSequential {
            XConvolution2D<E, D>(inputChannels: 64, outputChannels: 128, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 128, outputChannels: 128, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv3 = XSequential {
            XConvolution2D<E, D>(inputChannels: 128, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv4 = XSequential {
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv5 = XSequential {
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        dense = Self.makeDense(classes: classes)
    }
}

public struct VGG19<E: RandomizableType, D: DeviceType>: VGGBase {
    public typealias Parameter = E
    public typealias Device = D
    
    public var conv1: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var conv2: XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XConvolution2D<E, D>>, XSequential<XRelu<E, D>, XMaxPool2D<E, D>>>
    public var conv3: XSequential<XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XSequential<XConvolution2D<E, D>, XRelu<E, D>>>, XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XSequential<XConvolution2D<E, D>, XRelu<E, D>>>>, XMaxPool2D<E, D>>
    public var conv4: XSequential<XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XSequential<XConvolution2D<E, D>, XRelu<E, D>>>, XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XSequential<XConvolution2D<E, D>, XRelu<E, D>>>>, XMaxPool2D<E, D>>
    public var conv5: XSequential<XSequential<XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XSequential<XConvolution2D<E, D>, XRelu<E, D>>>, XSequential<XSequential<XConvolution2D<E, D>, XRelu<E, D>>, XSequential<XConvolution2D<E, D>, XRelu<E, D>>>>, XMaxPool2D<E, D>>
    public var dense: XSequential<XSequential<XSequential<XDense<E, D>, XRelu<E, D>>, XSequential<XDropout<E, D>, XDense<E, D>>>, XSequential<XSequential<XRelu<E, D>, XDropout<E, D>>, XSequential<XDense<E, D>, XSoftmax<E, D>>>>

    public init(inputChannels: Int, classes: Int) {
        conv1 = XSequential {
            XConvolution2D<E, D>(inputChannels: inputChannels, outputChannels: 64, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 64, outputChannels: 64, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv2 = XSequential {
            XConvolution2D<E, D>(inputChannels: 64, outputChannels: 128, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 128, outputChannels: 128, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv3 = XSequential {
            XConvolution2D<E, D>(inputChannels: 128, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 256, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv4 = XSequential {
            XConvolution2D<E, D>(inputChannels: 256, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        conv5 = XSequential {
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XConvolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            XRelu<E, D>()
            
            XMaxPool2D<E, D>()
        }
        
        dense = Self.makeDense(classes: classes)
    }
}
