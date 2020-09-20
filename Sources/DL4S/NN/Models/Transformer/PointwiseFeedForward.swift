//
//  File.swift
//  
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

/// Pointwise feed forward layer as introduced in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
///
/// The layer sequences a dense layer, GeLU activation, another dense layer and a dropout layer.
/// Furthermore, it has a residual connection and the output is layer normalized.
public struct PointwiseFeedForward<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var dense1: Dense<Element, Device>
    public var dense2: Dense<Element, Device>
    public var norm: LayerNorm<Element, Device>
    public var dropout: Dropout<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        Array([dense1.parameters, dense2.parameters, norm.parameters].joined())
    }
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            dense1.parameterPaths.map((\Self.dense1).appending(path:)),
            dense2.parameterPaths.map((\Self.dense2).appending(path:)),
            norm.parameterPaths.map((\Self.norm).appending(path:))
        ].joined())
    }
    
    /// Creates a pointwise forward layer to be used in a transformer as introduced in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// The block sequences a dense layer, gelu activation, another dense layer, dropout, a residual connection and layer normalization.
    ///
    /// - Parameters:
    ///   - size: Size of last dimension of inputs and outputs of the block
    ///   - hiddenSize: Hidden size between dense layers of the block
    ///   - dropoutRate: Rate, with which dropout is applied between activation and the second dense layer. Can be enabled and disabled using `isDropoutActive`.
    public init(size: Int, hiddenSize: Int, dropoutRate: Float) {
        dense1 = Dense(inputSize: size, outputSize: hiddenSize)
        dense2 = Dense(inputSize: hiddenSize, outputSize: size)
        norm = LayerNorm(inputSize: [size])
        dropout = Dropout(rate: dropoutRate)
    }
    
    /// Applies the pointwise feed forward layer to the provided inputs
    /// - Parameter inputs: tensor of shape [batch size, sequence length, size]
    /// - Returns: tensor of shape [batch size, sequence length, size]
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "PointwiseFeedForward") {
            // inputs: [batchSize, timeSteps, size]
            let batchSize = inputs.shape[0]
            let seqlen = inputs.shape[1]
            let size = inputs.shape[2]
            
            let denseInputs = inputs.view(as: batchSize * seqlen, size)
            let tmp1 = dense1(denseInputs)
            let tmp2 = tmp1.gaussianErrorLinear()
            let tmp3 = dense2(tmp2)
            let tmp4 = dropout(tmp3)
            let n = norm(tmp4 + denseInputs)
            return n.view(as: batchSize, seqlen, size)
        }
    }
}

