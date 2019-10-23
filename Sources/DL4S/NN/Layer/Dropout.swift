//
//  Dropout.swift
//  DL4S
//
//  Created by Palle Klewitz on 17.10.19.
//

import Foundation


public struct Dropout<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    public var rate: Float
    public var isActive: Bool = true
    
    public init(rate: Float) {
        self.rate = rate
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        if isActive {
            return OperationGroup.capture(named: "Dropout") {
                inputs * Tensor(bernoulliDistributedWithShape: Array(inputs.shape.dropFirst()), probability: (1 - rate))
            }
        } else {
            return inputs
        }
    }
}
