//
//  XBidirectionalRNN.swift
//  DL4S
//
//  Created by Palle Klewitz on 17.10.19.
//

import Foundation


public struct XBidirectional<RNNLayer: XRNN>: XLayer {
    public typealias Inputs = RNNLayer.Inputs
    public typealias Outputs = (forward: RNNLayer.Outputs, backward: RNNLayer.Outputs)
    
    public static var parameters: [WritableKeyPath<Self, XTensor<RNNLayer.Parameter, RNNLayer.Device>>] {
        [\Self.forwardLayer, \Self.backwardLayer].flatMap { basePath in
            RNNLayer.parameters.map(basePath.appending(path:))
        }
    }
    
    public var parameters: [XTensor<RNNLayer.Parameter, RNNLayer.Device>] {
        get {forwardLayer.parameters + backwardLayer.parameters}
        set {
            forwardLayer.parameters = Array(newValue[..<(newValue.count / 2)])
            backwardLayer.parameters = Array(newValue[(newValue.count / 2)...])
        }
    }
    
    public var forwardLayer: RNNLayer
    public var backwardLayer: RNNLayer
    
    public init(forward: RNNLayer, backward: RNNLayer) {
        precondition(forward.direction == .forward, "Forward RNN layer must have forward direction")
        precondition(backward.direction == .backward, "Backward RNN layer must have backward direction")
                
        self.forwardLayer = forward
        self.backwardLayer = backward
    }
    
    public func callAsFunction(_ inputs: RNNLayer.Inputs) -> (forward: (RNNLayer.State, () -> RNNLayer.StateSequence), backward: (RNNLayer.State, () -> RNNLayer.StateSequence)) {
        OperationGroup.capture(named: "BidirectionalRNN") {
            (forwardLayer.callAsFunction(inputs), backwardLayer.callAsFunction(inputs))
        }
    }
}
