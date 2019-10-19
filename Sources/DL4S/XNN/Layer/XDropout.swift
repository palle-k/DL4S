//
//  XDropout.swift
//  DL4S
//
//  Created by Palle Klewitz on 17.10.19.
//

import Foundation


public struct XDropout<Element: RandomizableType, Device: DeviceType>: XLayer, Codable {
    public static var parameters: [WritableKeyPath<Self, XTensor<Element, Device>>] {[]}
    public var parameters: [XTensor<Element, Device>] { get {[]} set {} }
    
    public var rate: Float
    
    public init(rate: Float) {
        self.rate = rate
    }
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        OperationGroup.capture(named: "Dropout") {
            inputs * XTensor(bernoulliDistributedWithShape: Array(inputs.shape.dropFirst()), probability: (1 - rate))
        }
    }
}
