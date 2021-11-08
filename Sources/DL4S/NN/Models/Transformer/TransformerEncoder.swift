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

/// Transformer encoder sequencing positional encoding and token embedding and multiple transformer encoder layers, as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
public struct TransformerEncoder<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var encoderLayers: [TransformerEncoderBlock<Element, Device>]
    
    public var parameters: [Tensor<Element, Device>] {Array([
        encoderLayers.flatMap {$0.parameters},
    ].joined())}
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {Array([
        encoderLayers.enumerated().flatMap { (idx, layer) in
            layer.parameterPaths.map((\Self.encoderLayers[idx]).appending(path:))
        }
    ].joined())}
    
    /// Creates a transformer encoder sequencing positional encoding and token embedding and multiple transformer encoder layers, as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - vocabSize: Number of distinct tokens that can occur in input
    ///   - layerCount: Number of transformer encoder layers
    ///   - heads: Number of attention heads in each encoder layer
    ///   - keyDim: Size of keys in multi-head attention layers
    ///   - valueDim: Size of values in multi-head attention layers
    ///   - modelDim: Size of embedding vectors as well as hidden layer activations and outputs
    ///   - forwardDim: Size of hidden layer activations within pointwise feed forward layers
    ///   - dropout: Rate of dropout applied within pointwise feed forward and multi-head attention layers
    public init(layerCount: Int, heads: Int, keyDim: Int, valueDim: Int, modelDim: Int, forwardDim: Int, dropout: Float) {
        encoderLayers = (0 ..< layerCount).map { i in
            TransformerEncoderBlock(hiddenDim: modelDim, forwardDim: forwardDim, heads: heads, keyDim: keyDim, valueDim: valueDim, dropout: dropout)
        }
    }
    
    /// Forwards the given batch of token sequences through the encoder.
    /// - Parameter inputs: Token sequences
    /// - Returns: Batch of encoder outputs with shape [inputs.count, maxLen, hiddenSize]
    public func callAsFunction(_ inputs: (input: Tensor<Element, Device>, sequenceLengths: [Int])) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Encoder") {
            let mask: Tensor<Element, Device> = makeEncoderMasks(sequenceLengths: inputs.sequenceLengths)
            
            let encoderOutput = encoderLayers.reduce(inputs.input) { acc, layer in
                layer((inputs: acc, mask: mask))
            } // [batchSize, maxLen, hiddenDim]
            
            return encoderOutput // [batchSize, maxLen, hiddenDim]
        }
    }
}
