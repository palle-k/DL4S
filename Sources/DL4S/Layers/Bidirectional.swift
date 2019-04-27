//
//  BidirectionalRNN.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.04.19.
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

/// Bidirectional RNN implementation
/// 
/// Combines two RNNs, one for forward, one for backwards into a single RNN
/// and fuses the resulting states together.
public class BidirectionalRNN<Element: RandomizableType, Device, RNNLayer: RNN>: Layer where RNNLayer.Input == Element, RNNLayer.Element == Element, RNNLayer.Device == Device {
    public let forwardLayer: RNNLayer
    public let backwardLayer: RNNLayer
    
    public var isTrainable: Bool {
        get {
            return forwardLayer.isTrainable && backwardLayer.isTrainable
        }
        set {
            forwardLayer.isTrainable = newValue
            backwardLayer.isTrainable = newValue
        }
    }
    
    public var parameters: [Tensor<Element, Device>] {
        return Array([forwardLayer.parameters, backwardLayer.parameters].joined())
    }
    
    public var trainableParameters: [Tensor<Element, Device>] {
        return Array([forwardLayer.trainableParameters, backwardLayer.trainableParameters].joined())
    }
    
    public init(forwardLayer: RNNLayer, backwardLayer: RNNLayer) {
        precondition(forwardLayer.direction == .forward, "Forward layer must have forward direction")
        precondition(backwardLayer.direction == .backward, "Backward layer must have backward direction")
        
        self.forwardLayer = forwardLayer
        self.backwardLayer = backwardLayer
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        let x = inputs[0]
        
        let forwardResult = forwardLayer.forward(x)
        let backwardResult = backwardLayer.forward(x)
        
        if forwardLayer.shouldReturnFullSequence && backwardLayer.shouldReturnFullSequence {
            // output of RNNs: [seqlen, batchSize, stateSize]
            return stack(forwardResult, backwardResult, axis: 2)
        } else if !forwardLayer.shouldReturnFullSequence && !backwardLayer.shouldReturnFullSequence {
            // output of RNNs: [batchSize, stateSize]
            return stack(forwardResult, backwardResult, axis: 1)
        } else {
            fatalError("RNNs must either both return full state sequences or both return final state")
        }
    }
}
