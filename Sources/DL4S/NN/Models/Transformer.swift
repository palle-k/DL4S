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

fileprivate func makeEncoderMasks<Element, Device>(sequenceLengths: [Int]) -> Tensor<Element, Device> {
    let maxInLen = sequenceLengths.reduce(0, max)
    let batchSize = sequenceLengths.count
    
    return Tensor<Element, Device>(sequenceLengths.map {
        Array(repeating: 0, count: $0) + Array(repeating: 1, count: maxInLen - $0)
    }).view(as: batchSize, 1, 1, maxInLen) // TODO: Check if maxLen in 3rd or 4th position
}

fileprivate func makeDecoderMasks<Element, Device>(sequenceLengths: [Int]) -> Tensor<Element, Device> {
    let batchSize = sequenceLengths.count
    let maxLen = sequenceLengths.reduce(0, max)
    
    let decoderSeqMask = Tensor<Element, Device>(sequenceLengths.map {
        Array(repeating: 0, count: $0) + Array(repeating: 1, count: maxLen - $0)
    })
    
    let decoderCausalMask = Tensor<Element, Device>(repeating: 1, shape: maxLen, maxLen)
        .bandMatrix(belowDiagonal: -1, aboveDiagonal: nil) // [maxLen, maxLen]
    
    return 1 - relu(1 - decoderSeqMask.view(as: batchSize, 1, 1, maxLen) - decoderCausalMask)
}

/// Positional Encoding layer using the encoding method proposed in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
///
/// The layer takes an array of Ints as an input, which indicate the number of elements in each sequence of the minibatch.
/// It returns a tensor with the shape [max(inputs), hiddenSize].
/// It does not mask out positional encodings for padding elements.
public struct PositionalEncoding<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public typealias Parameter = Element
    
    /// Number of elements in the positional encoding output tensor.
    public let hiddenSize: Int
    
    /// Creates a Positional Encoding layer using the encoding method proposed in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameter hiddenSize: Number of elements in the positional encoding output tensor.
    public init(hiddenSize: Int) {
        precondition(hiddenSize.isMultiple(of: 2), "Hidden size must be multiple of 2")
        self.hiddenSize = hiddenSize
    }
    
    public var parameters: [Tensor<Element, Device>] {[]}
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    
    /// Creates a positional encoding matrix
    /// - Parameter maxLen: Maximum sequence length in the current minibatch
    /// - Returns: Tensor of shape [max(inputs), hiddenSize]
    public func callAsFunction(_ maxLen: Int) -> Tensor<Element, Device> {
        let inputRange = Tensor<Element, Device>((0 ..< maxLen).map(Element.init))
        
        let hiddenRange = Tensor<Element, Device>((0 ..< hiddenSize / 2).map(Element.init))
        let frequencies = Tensor(10000).raised(toPowerOf: hiddenRange / Tensor(Element(hiddenSize / 2)))
        let samplePoints = inputRange.unsqueezed(at: 1) / frequencies.unsqueezed(at: 0) // [seqlen, hiddenSize / 2]
        
        var samples = Tensor(
            stacking: [
                sin(samplePoints).unsqueezed(at: 2),
                cos(samplePoints).unsqueezed(at: 2)
            ],
            along: 2
        ).view(as: -1, hiddenSize) // [seqlen, hiddenSize]
        
        #if DEBUG
        samples.tag = "PositionEncodings"
        #endif
        return samples // [seqlen, hiddensize]
    }
}


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


/// Computes Scaled Multi-Head Dot Product Attention as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
public struct ScaledDotProductAttention<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var temperature: Element
    
    public init(temperature: Element) {
        self.temperature = temperature
    }
    
    public var parameters: [Tensor<Element, Device>] {[]}
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    
    /// Performs scaled dot product attention.
    /// - Parameter inputs: Tuple containing queries of shape [batchSize, heads, queryCount, keyDim], keys of shape [batchSize, heads, keyCount, keyDim] and values of shape [batchSize, heads, valueCount, valueDim]
    ///       as well as an optional mask that may be used to prevent attention to certain elements outside of the batch or in future timesteps. Mask must be broadcastable to shape [batchSize, heads, queryCount, keyCount]
    /// - Returns: Attended values tensor of shape [batchSize, heads, queryCount, valueDim]
    public func callAsFunction(_ inputs: (q: Tensor<Element, Device>, k: Tensor<Element, Device>, v: Tensor<Element, Device>, mask: Tensor<Element, Device>?)) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "ScaledDotProductAttention") {
            let (q, k, v, mask) = inputs
            precondition(k.dim == 4)
            
            // q: [batchSize, heads, queryCount, keyDim]
            // k: [batchSize, heads, keyCount, keyDim]
            // v: [batchSize, heads, valueCount, valueDim]
            
            // [batchSize, heads, queryCount, keyDim] x [batchSize, heads, [keyCount, keyDim]^T] -> [batchSize, heads, queryCount, keyCount]
            var attn = (q / Tensor(temperature)).broadcastMatrixMultiplied(with: k, transposeSelf: false, transposeOther: true)
            
            if let mask = mask {
                attn = attn - mask * 1e9 // mask contains 1 for all entries that should be masked away ==> softmax zeros them out.
            }
            
            attn = softmax(attn, axis: 3) // softmax over last axis (keyCount)
            // [batchSize, heads, queryCount, keyCount] x [batchSize, heads, valueCount, valueDim] -> [batchSize, heads, queryCount, valueDim] (constraint: keyCount == valueCount]
            let output = attn.broadcastMatrixMultiplied(with: v)
            return output // [batchSize, heads, queryCount, valueDim]
        }
    }
}


/// Multi-Head Attention Layer following [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
public struct MultiHeadAttention<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    /// Matrix multiplied with queries before dot product attention
    public var qDense: Tensor<Element, Device>
    /// Matrix multiplied with keys before dot product attention
    public var kDense: Tensor<Element, Device>
    /// Matrix multiplied with values before dot product attention
    public var vDense: Tensor<Element, Device>
    /// Matrix multiplied with result from dot product attention layer
    public var fc: Tensor<Element, Device>
    public var attn: ScaledDotProductAttention<Element, Device>
    public var norm: LayerNorm<Element, Device>
    public var dropout: Dropout<Element, Device>
    
    /// Number of attention heads
    public let heads: Int
    /// Dimensionality of query and key vectors
    public let keyDim: Int
    /// Dimensionality of value vectors
    public let valueDim: Int
    /// Lat dimension of keys, queries and values before matrix multiplication
    public let hiddenDim: Int
    
    public var parameters: [Tensor<Element, Device>] {Array([
        [qDense, kDense, vDense, fc],
        norm.parameters
    ].joined())}
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {Array([
        [\.qDense, \.kDense, \.vDense, \.fc],
        norm.parameterPaths.map((\Self.norm).appending(path:))
    ].joined())}
    
    /// Multi-Head Attention Layer following [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - heads: Number of attention heads
    ///   - hiddenDim: Last dimension of keys, queries and values
    ///   - keyDim: Last dimesion of keys
    ///   - valueDim: Intermediate last dimension of values
    ///   - dropout: Dropout rate
    public init(heads: Int, hiddenDim: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1) {
        self.heads = heads
        self.keyDim = keyDim
        self.valueDim = valueDim
        self.hiddenDim = hiddenDim
        
        attn = ScaledDotProductAttention(temperature: Element(keyDim).sqrt())
        qDense = Tensor(xavierNormalWithShape: hiddenDim, keyDim * heads, requiresGradient: true)
        kDense = Tensor(xavierNormalWithShape: hiddenDim, keyDim * heads, requiresGradient: true)
        vDense = Tensor(xavierNormalWithShape: hiddenDim, valueDim * heads, requiresGradient: true)
        fc = Tensor(xavierNormalWithShape: valueDim * heads, hiddenDim, requiresGradient: true)
        self.dropout = Dropout(rate: dropout)
        norm = LayerNorm(inputSize: [hiddenDim])
        
        #if DEBUG
        qDense.tag = "qDense"
        kDense.tag = "kDense"
        vDense.tag = "vDense"
        fc.tag = "FC"
        #endif
    }
    
    /// Computes multi-head scaled dot product attention using the provided query, key and value vector as well as the provided mask.
    ///
    /// Additionally applies dropout, a residual connection and layer normalization.
    ///
    /// - Parameter inputs: Tuple containing queries of shape [batchSize, queryCount, hiddenDim], keys of shape [batchSize, keyCount, hiddenDim] and values of shape [batchSize, valueCount, hiddenDim]
    ///       as well as an optional mask that may be used to prevent attention to certain elements outside of the batch or in future timesteps. Mask must be broadcastable to shape [batchSize, heads, queryCount, keyCount] and have 1 entries for all elements that should be blocked.
    /// - Returns: Normalized scaled dot product attended values
    public func callAsFunction(_ inputs: (q: Tensor<Element, Device>, k: Tensor<Element, Device>, v: Tensor<Element, Device>, mask: Tensor<Element, Device>?)) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "MultiHeadAttention") {
            let (q, k, v, mask) = inputs // q, k, v: [batchSize, maxLen, hiddenDim]
            
            let batchSize = q.shape[0]
            let queryCount = q.shape[1] // == maxLen
            let keyCount = k.shape[1]
            let valueCount = v.shape[1]
            
            let res = q
            
            // [batchSize, queryCount, hiddenDim] x [hiddenDim, keyDim * heads] --> [batchSize, maxLen, keyDim * heads]
            let q_prep = q.broadcastMatrixMultiplied(with: qDense).view(as: batchSize, queryCount, heads, keyDim) // [batchSize, queryCount
            let k_prep = k.broadcastMatrixMultiplied(with: kDense).view(as: batchSize, keyCount, heads, keyDim)
            let v_prep = v.broadcastMatrixMultiplied(with: vDense).view(as: batchSize, valueCount, heads, valueDim)
            
            let q_trans = q_prep.permuted(to: 0, 2, 1, 3) // [batchSize, heads, queryCount, keyDim]
            let k_trans = k_prep.permuted(to: 0, 2, 1, 3) // [batchSize, heads, keyCount, keyDim]
            let v_trans = v_prep.permuted(to: 0, 2, 1, 3) // [batchSize, heads, valueCount, valueDim]
            
            let q_attn = self.attn((q: q_trans, k: k_trans, v: v_trans, mask: mask)) // [batchSize, heads, queryCount, valueDim]
            
            let out = q_attn.permuted(to: 0, 2, 1, 3) // [batchSize, queryCount, heads, valueDim]
                .view(as: batchSize, queryCount, -1) // [batchSize, queryCount, heads * valueDim]
                .broadcastMatrixMultiplied(with: fc) // [batchSize, queryCount, heads * valueDim] x [valueDim * heads, hiddenDim] --> [batchSize, queryCount, hiddenDim]
            let out_drop = dropout(out)
            let q_res = out_drop + res
            let normalized = norm(q_res)
            return normalized // [batchSize, queryCount, hiddenDim]
        }
    }
}

/// Transformer encoder layer consisting of a self-attention and a pointwise feed forward layer as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
public struct TransformerEncoderBlock<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var selfAttention: MultiHeadAttention<Element, Device>
    public var pointwiseFeedForward: PointwiseFeedForward<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {Array([
        selfAttention.parameters, pointwiseFeedForward.parameters
    ].joined())}
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {Array([
        selfAttention.parameterPaths.map((\Self.selfAttention).appending(path:)),
        pointwiseFeedForward.parameterPaths.map((\Self.pointwiseFeedForward).appending(path:))
    ].joined())}
    
    /// Creates Transformer encoder layer consisting of a self-attention and a pointwise feed forward layer as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - hiddenDim: Last dimension of inputs and outputs
    ///   - forwardDim: Size of value vectors within pointwise feed forward layer
    ///   - heads: Number of attention heads
    ///   - keyDim: Size of key and query vectors within multi-head attention layer
    ///   - valueDim: Size of value vectors within multi-head attention layer
    ///   - dropout: Dropout rate for dropout applied within self-attention and pointwise feed forward layer
    public init(hiddenDim: Int, forwardDim: Int, heads: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1) {
        selfAttention = MultiHeadAttention(heads: heads, hiddenDim: hiddenDim, keyDim: keyDim, valueDim: valueDim, dropout: dropout)
        pointwiseFeedForward = PointwiseFeedForward(size: hiddenDim, hiddenSize: forwardDim, dropoutRate: dropout)
    }
    
    /// Applies multi-head self attention and a pointwise feed forward layer to the inputs
    /// - Parameter inputs: Layer input with shape [batchSize, maxLen, hiddenSize] and padding mask broadcastable to [batchSize, heads, queryCount, keyCount] with 1 entries for all elements that should be blocked.
    /// - Returns: Result of layer operations with shape [batchSize, maxLen, hiddenSize]
    public func callAsFunction(_ inputs: (inputs: Tensor<Element, Device>, mask: Tensor<Element, Device>)) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "EncoderLayer") {
            // inputs: [batchSize, maxLen == queryCount, hiddenDim]
            let (inputs, mask) = inputs
            let attn = selfAttention((q: inputs, k: inputs, v: inputs, mask: mask))
            return pointwiseFeedForward(attn)
        }
    }
}

/// Transformer decoder layer consisting of a self attention, encoder attention and a pointwise feed forward layer as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
public struct TransformerDecoderBlock<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var selfAttention: MultiHeadAttention<Element, Device>
    public var encoderAttention: MultiHeadAttention<Element, Device>
    public var pointwiseFeedForward: PointwiseFeedForward<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {Array([
        selfAttention.parameters, encoderAttention.parameters, pointwiseFeedForward.parameters
    ].joined())}
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {Array([
        selfAttention.parameterPaths.map((\Self.selfAttention).appending(path:)),
        encoderAttention.parameterPaths.map((\Self.encoderAttention).appending(path:)),
        pointwiseFeedForward.parameterPaths.map((\Self.pointwiseFeedForward).appending(path:))
    ].joined())}
    
    /// Creates Transformer encoder layer consisting of a self-attention and a pointwise feed forward layer as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - hiddenDim: Last dimension of inputs and outputs
    ///   - forwardDim: Size of value vectors within pointwise feed forward layer
    ///   - heads: Number of attention heads
    ///   - keyDim: Size of key and query vectors within multi-head attention layer
    ///   - valueDim: Size of value vectors within multi-head attention layer
    ///   - dropout: Dropout rate for dropout applied within self-attention and pointwise feed forward layer
    public init(hiddenDim: Int, forwardDim: Int, heads: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1) {
        selfAttention = MultiHeadAttention(heads: heads, hiddenDim: hiddenDim, keyDim: keyDim, valueDim: valueDim, dropout: dropout)
        encoderAttention = MultiHeadAttention(heads: heads, hiddenDim: hiddenDim, keyDim: keyDim, valueDim: valueDim, dropout: dropout)
        pointwiseFeedForward = PointwiseFeedForward(size: hiddenDim, hiddenSize: forwardDim, dropoutRate: dropout)
    }
    
    /// Applies multi-head self attention and a pointwise feed forward layer to the inputs
    /// - Parameter inputs: Layer input with shape [batchSize, maxLen, hiddenSize], encoder outputs with shape [batchSize, maxLen, hiddenSize] and masks broadcastable to [batchSize, heads, queryCount, keyCount] with 1 entries for all elements that should be blocked for encoder and decoder states.
    /// - Returns: Result of layer operations with shape [batchSize, maxLen, hiddenSize]
    public func callAsFunction(_ inputs: (decoderInput: Tensor<Element, Device>, encoderOutput: Tensor<Element, Device>, encoderMask: Tensor<Element, Device>, decoderMask: Tensor<Element, Device>)) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "DecoderLayer") {
            let (decoderInput, encoderOutput, encoderMask, decoderMask) = inputs
            let decAttn = selfAttention((q: decoderInput, k: decoderInput, v: decoderInput, mask: decoderMask))
            let encAttn = encoderAttention((q: decAttn, k: encoderOutput, v: encoderOutput, mask: encoderMask))
            return pointwiseFeedForward(encAttn)
        }
    }
}

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
        let deembedded = (decoded.broadcastMatrixMultiplied(with: embedding.embeddingMatrix, transposeOther: true) + outputBias)
        
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
