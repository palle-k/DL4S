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
public struct TransformerDecoder<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var decoderLayers: [TransformerDecoderBlock<Element, Device>]
    
    public var parameters: [Tensor<Element, Device>] {Array([
        decoderLayers.flatMap {$0.parameters},
    ].joined())}
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {Array([
        decoderLayers.enumerated().flatMap { (idx, layer) in
            layer.parameterPaths.map((\Self.decoderLayers[idx]).appending(path:))
        }
    ].joined())}
    
    /// Creates aransformer encoder sequencing positional encoding and token embedding and multiple transformer encoder layers, as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    public init(layerCount: Int, heads: Int, keyDim: Int, valueDim: Int, modelDim: Int, forwardDim: Int, dropout: Float) {
        decoderLayers = (0 ..< layerCount).map { _ in
            TransformerDecoderBlock(hiddenDim: modelDim, forwardDim: forwardDim, heads: heads, keyDim: keyDim, valueDim: valueDim, dropout: dropout)
        }
    }
    
    public func callAsFunction(_ inputs: (decoderInput: Tensor<Element, Device>, encoderStates: Tensor<Element, Device>, encoderInputLengths: [Int], decoderInputLengths: [Int])) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Decoder") {
            let (decoderInput, encoderStates, encoderInputLengths, decoderInputLengths) = inputs

            let encoderMask: Tensor<Element, Device> = makeEncoderMasks(sequenceLengths: encoderInputLengths)
            let decoderMask: Tensor<Element, Device> = makeDecoderMasks(sequenceLengths: decoderInputLengths)
            
            let decoderOutput = decoderLayers.reduce(decoderInput) { acc, layer in
                layer((acc, encoderStates, encoderMask, decoderMask))
            } // [batchSize, maxLen, hiddenDim]
            
            return decoderOutput
        }
    }
}
