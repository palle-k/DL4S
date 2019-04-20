//
//  Pool.swift
//  DL4S
//
//  Created by Palle Klewitz on 19.04.19.
//

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
