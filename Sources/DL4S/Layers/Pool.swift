//
//  Pool.swift
//  DL4S
//
//  Created by Palle Klewitz on 19.04.19.
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


public class MaxPool2D<Element: NumericType, Device: DeviceType>: Layer {
    public let stride: Int
    public let padding: Int
    public let windowSize: Int
    
    public var isTrainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public init(windowSize: Int, stride: Int? = nil, padding: Int? = nil) {
        self.stride = stride ?? windowSize
        self.padding = padding ?? ((windowSize - 1) / 2)
        self.windowSize = windowSize
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        return maxPool2d(images: inputs[0], windowSize: windowSize, padding: padding, stride: stride)
    }
}


public class AvgPool2D<Element: NumericType, Device: DeviceType>: Layer {
    public let stride: Int
    public let padding: Int
    public let windowSize: Int
    
    public var isTrainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public init(windowSize: Int, stride: Int? = nil, padding: Int? = nil) {
        self.stride = stride ?? windowSize
        self.padding = padding ?? ((windowSize - 1) / 2)
        self.windowSize = windowSize
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        return avgPool2d(images: inputs[0], windowSize: windowSize, padding: padding, stride: stride)
    }
}


public class AdaptiveMaxPool2D<Element: NumericType, Device: DeviceType>: Layer {
    public let targetSize: Int
    
    public var isTrainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public init(targetSize: Int) {
        self.targetSize = targetSize
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        let x = inputs[0]
        let height = x.shape[2]
        let windowSize = height / targetSize
        return maxPool2d(images: x, windowSize: windowSize, padding: 0, stride: windowSize)
    }
}

public class AdaptiveAvgPool2D<Element: NumericType, Device: DeviceType>: Layer {
    public let targetSize: Int
    
    public var isTrainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public init(targetSize: Int) {
        self.targetSize = targetSize
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        let x = inputs[0]
        let height = x.shape[2]
        let windowSize = height / targetSize
        return avgPool2d(images: x, windowSize: windowSize, padding: 0, stride: windowSize)
    }
}
