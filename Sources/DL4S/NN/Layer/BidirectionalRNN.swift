//
//  BidirectionalRNN.swift
//  DL4S
//
//  Created by Palle Klewitz on 17.10.19.
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


/// A bidirectional RNN
public struct Bidirectional<RNNLayer: RNN>: LayerType {
    public typealias Inputs = RNNLayer.Inputs
    public typealias Outputs = (forward: RNNLayer.Outputs, backward: RNNLayer.Outputs)
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<RNNLayer.Parameter, RNNLayer.Device>>] {
        let forwardPaths = forwardLayer.parameterPaths.map {
            (\Self.forwardLayer).appending(path: $0)
        }
        let backwardPaths = backwardLayer.parameterPaths.map {
            (\Self.backwardLayer).appending(path: $0)
        }
        return forwardPaths + backwardPaths
    }
    
    public var parameters: [Tensor<RNNLayer.Parameter, RNNLayer.Device>] {
        get {forwardLayer.parameters + backwardLayer.parameters}
    }
    
    /// RNN for forwards direction
    public var forwardLayer: RNNLayer
    
    /// RNN for backwards direction
    public var backwardLayer: RNNLayer
    
    /// Creates a bidirectional RNN with the given RNNs for the forward and backward pass.
    /// - Parameters:
    ///   - forward: RNN for forward pass. Must have `direction == .forward`.
    ///   - backward: RNN for backward pass. Must have `direction == .backward`.
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

extension Bidirectional: Codable where RNNLayer: Codable {}
