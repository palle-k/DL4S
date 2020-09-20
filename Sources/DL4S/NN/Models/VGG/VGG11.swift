//
//  VGG11.swift
//  DL4S
//
//  Created by Palle Klewitz on 20.09.20.
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

public struct VGG11<E: RandomizableType, D: DeviceType>: VGGBase {
    public typealias Parameter = E
    public typealias Device = D
    
    public var conv1: Sequential<Sequential<Convolution2D<E, D>, BatchNorm<E, D>>, Sequential<Relu<E, D>, MaxPool2D<E, D>>>
    public var conv2: Sequential<Sequential<Convolution2D<E, D>, BatchNorm<E, D>>, Sequential<Relu<E, D>, MaxPool2D<E, D>>>
    public var conv3: Sequential<Sequential<Sequential<Convolution2D<E, D>, BatchNorm<E, D>>, Sequential<Relu<E, D>, Convolution2D<E, D>>>, Sequential<Sequential<BatchNorm<E, D>, Relu<E, D>>, MaxPool2D<E, D>>>
    public var conv4: Sequential<Sequential<Sequential<Convolution2D<E, D>, BatchNorm<E, D>>, Sequential<Relu<E, D>, Convolution2D<E, D>>>, Sequential<Sequential<BatchNorm<E, D>, Relu<E, D>>, MaxPool2D<E, D>>>
    public var conv5: Sequential<Sequential<Sequential<Convolution2D<E, D>, BatchNorm<E, D>>, Sequential<Relu<E, D>, Convolution2D<E, D>>>, Sequential<Sequential<BatchNorm<E, D>, Relu<E, D>>, MaxPool2D<E, D>>>
    public var dense: DenseLayer

    public init(inputChannels: Int, classes: Int) {
        conv1 = Sequential {
            Convolution2D<E, D>(inputChannels: inputChannels, outputChannels: 64, kernelSize: (3, 3))
            BatchNorm<E, D>(inputSize: [64, 1, 1])
            Relu<E, D>()
            MaxPool2D<E, D>()
        }
        
        conv2 = Sequential {
            Convolution2D<E, D>(inputChannels: 64, outputChannels: 128, kernelSize: (3, 3))
            BatchNorm<E, D>(inputSize: [128, 1, 1])
            Relu<E, D>()
            MaxPool2D<E, D>()
        }
        
        conv3 = Sequential {
            Convolution2D<E, D>(inputChannels: 128, outputChannels: 256, kernelSize: (3, 3))
            BatchNorm<E, D>(inputSize: [256, 1, 1])
            Relu<E, D>()
            
            Convolution2D<E, D>(inputChannels: 256, outputChannels: 256, kernelSize: (3, 3))
            BatchNorm<E, D>(inputSize: [256, 1, 1])
            Relu<E, D>()
            
            MaxPool2D<E, D>()
        }
        
        conv4 = Sequential {
            Convolution2D<E, D>(inputChannels: 256, outputChannels: 512, kernelSize: (3, 3))
            BatchNorm<E, D>(inputSize: [512, 1, 1])
            Relu<E, D>()
            
            Convolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            BatchNorm<E, D>(inputSize: [512, 1, 1])
            Relu<E, D>()
            
            MaxPool2D<E, D>()
        }
        
        conv5 = Sequential {
            Convolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            BatchNorm<E, D>(inputSize: [512, 1, 1])
            Relu<E, D>()
            
            Convolution2D<E, D>(inputChannels: 512, outputChannels: 512, kernelSize: (3, 3))
            BatchNorm<E, D>(inputSize: [512, 1, 1])
            Relu<E, D>()
            
            MaxPool2D<E, D>()
        }
        
        dense = Self.makeDense(classes: classes)
    }
}
