//
//  Transformer.swift
//  DL4S
//
//  Created by Palle Klewitz on 07.11.19.
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

/// Transformer as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
///
/// The transformer model shares an embedding matrix between the encoder and decoder and reuses the embedding weights to compute the decoder output distribution.
/// Outputs of the transformer are normalized using log softmax.
public struct Transformer<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public typealias Outputs = Tensor<Element, Device> // disambiguates callAsFunction protocol requirement
    
    public var embedding: Embedding<Element, Device>
    public var positionalEncoding: PositionalEncoding<Element, Device>
    public var dropout: Dropout<Element, Device>
    
    public var encoder: TransformerEncoder<Element, Device>
    public var decoder: TransformerDecoder<Element, Device>
    
    public var outputBias: Tensor<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {Array([
        embedding.parameters,
        encoder.parameters,
        decoder.parameters,
        [outputBias]
    ].joined())}
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {Array([
        embedding.parameterPaths.map((\Self.embedding).appending(path:)),
        encoder.parameterPaths.map((\Self.encoder).appending(path:)),
        decoder.parameterPaths.map((\Self.decoder).appending(path:)),
        [\Self.outputBias]
    ].joined())}
    
    /// Creates a new transformer, which follows [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - encoderLayers: Number of encoder layers
    ///   - decoderLayers: Number of decoder layers
    ///   - vocabSize: Number of tokens in the vocabulary of the transformer
    ///   - hiddenDim: Size of transformer layer outputs
    ///   - heads: Number of attention heads in multi-head attention layers
    ///   - keyDim: Size of key vectors in multi-head attention layers
    ///   - valueDim: Size of value vectors in muti-head attention layers
    ///   - forwardDim: Size of activations in poitnwise feed forward layers
    ///   - dropout: Dropout rate
    public init(encoderLayers: Int, decoderLayers: Int, vocabSize: Int, hiddenDim: Int, heads: Int, keyDim: Int, valueDim: Int, forwardDim: Int, dropout: Float = 0.1) {
        embedding = Embedding(inputFeatures: vocabSize, outputSize: hiddenDim, ignoreIndex: -1)
        self.dropout = Dropout(rate: dropout)
        positionalEncoding = PositionalEncoding(hiddenSize: hiddenDim)
        encoder = TransformerEncoder(layerCount: encoderLayers, heads: heads, keyDim: keyDim, valueDim: valueDim, modelDim: hiddenDim, forwardDim: forwardDim, dropout: dropout)
        decoder = TransformerDecoder(layerCount: decoderLayers, heads: heads, keyDim: keyDim, valueDim: valueDim, modelDim: hiddenDim, forwardDim: forwardDim, dropout: dropout)
        outputBias = Tensor(repeating: 0, shape: [vocabSize], requiresGradient: true)
        
        #if DEBUG
        outputBias.tag = "outBias"
        #endif
    }
    
    private func prepareInputs(_ inputs: Tensor<Int32, Device>) -> Tensor<Element, Device> {
        let embedded = embedding(inputs.flattened())
            .view(as: inputs.shape[0], inputs.shape[1], -1) // [batchSize, maxLen, embedDim]
        let encoderPositions = positionalEncoding(inputs.shape[1]) // [maxLen, embedDim]
        
        return dropout(embedded * Tensor(Element(embedded.shape[2]).sqrt()) + encoderPositions) // [batchSize, maxLen, embedDim]
    }
    
    /// Computes the outputs of the decoder given the inputs for the encoder and decoder.
    /// - Parameter inputs: Tuple containing:
    ///         - Padded encoder inputs using -1 as a padding token.
    ///         - Padded decoder inputs using -1 as padding token.
    ///
    /// - Returns: Batch of sequences of log-softmax normalized distributions over the vocabulary of the transformer with shape [batchSize, seqlen, vocabDim]
    public func callAsFunction(_ inputs: (encoderInput: Tensor<Int32, Device>, decoderInput: Tensor<Int32, Device>, encoderInputLengths: [Int], decoderInputLengths: [Int])) -> Tensor<Element, Device> {
        let (encoderInput, decoderInput, encInLens, decInLens) = inputs
        
        let embeddedEncoderInput = prepareInputs(encoderInput)
        let embeddedDecoderInput = prepareInputs(decoderInput)
        
        let encoderStates = encoder((embeddedEncoderInput, encInLens))
        let decoded = decoder((embeddedDecoderInput, encoderStates, encInLens, decInLens)) // [batchSize, maxLen, hiddenSize]
        
        // [batchSize, maxLen, hiddenSize] x [vocabSize, hiddenSize]^T --> [batchSize, maxLen, vocabSize]
        let deembedded = decoded.broadcastMatrixMultiplied(with: embedding.embeddingMatrix, transposeOther: true) + outputBias
        
        return logSoftmax(deembedded, axis: 2)
    }
    
    /// Greedily decodes the most probable sequence of output symbols given a sequence of input tokens
    /// - Parameters:
    ///   - inputSequence: Input tokens
    ///   - startToken: First token to feed into the decoder. Subsequent tokens are generated autoregressively.
    ///   - endToken: Token, which ends decoding (end of sequence marker)
    ///   - maxLength: Maximum length of the decoded sequence. If no endToken occurs after maxLength tokens, decoding is aborted.
    /// - Returns: Most probable output sequence determined by greedy decoding.
    public func callAsFunction(inputSequence: [Int32], startToken: Int32, endToken: Int32, maxLength: Int) -> [Int32] {
        let encIn = prepareInputs(Tensor([inputSequence]))
        let encoded = encoder((encIn, [inputSequence.count]))
        
        var tokens: [Int32] = []
        for _ in 0 ..< maxLength {
            let tokenInput = [[startToken] + tokens]
            let decIn = prepareInputs(Tensor(tokenInput))
            let output = decoder((decIn, encoded, [inputSequence.count], [tokenInput[0].count]))
            let deembedded = output.broadcastMatrixMultiplied(with: embedding.embeddingMatrix, transposeOther: true)
            
            let nextToken = deembedded[0, -1].argmax()
            tokens.append(Int32(nextToken))
            if nextToken == endToken {
                break
            }
        }
        
        return tokens
    }
}
