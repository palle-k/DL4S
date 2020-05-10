//
//  Transformer.swift
//  DL4S
//
//  Created by Palle Klewitz on 07.11.19.
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

/// A feed forward block for a transformer.
///
/// The block sequences a dense layer, gelu activation, dropout and another dense layer.
///
public struct TransformerFeedForwardBlock<Element: RandomizableType, Device: DeviceType>: LayerType {
    public var dense1: Dense<Element, Device>
    public var dense2: Dense<Element, Device>
    public var dropout: Dropout<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        Array([dense1.parameters, dense2.parameters].joined())
    }
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            dense1.parameterPaths.map((\Self.dense1).appending(path:)),
            dense2.parameterPaths.map((\Self.dense2).appending(path:))
        ].joined())
    }
    
    /// Determines and controls, whether dropout is applied between the activation and the second dense layer.
    public var isDropoutActive: Bool {
        get {dropout.isActive}
        set {dropout.isActive = newValue}
    }
    
    
    /// Creates a feed forward block to be used in a transformer.
    /// The block sequences a dense layer, gelu activation, dropout and another dense layer.
    ///
    /// - Parameters:
    ///   - size: Size of last dimension of inputs and outputs of the block
    ///   - hiddenSize: Hidden size between dense layers of the block
    ///   - dropoutRate: Rate, with which dropout is applied between activation and the second dense layer. Can be enabled and disabled using `isDropoutActive`.
    public init(size: Int, hiddenSize: Int, dropoutRate: Float) {
        dense1 = Dense(inputSize: size, outputSize: hiddenSize)
        dense2 = Dense(inputSize: hiddenSize, outputSize: size)
        dropout = Dropout(rate: dropoutRate)
    }
    
    /// Applies the feed forward block to the provided inputs
    /// - Parameter inputs: tensor of shape [batch size, sequence length, size]
    /// - Returns: tensor of shape [batch size, sequence length, size]
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "TransformerFeedForward") {
            // inputs: [batchSize, timeSteps, size]
            let tmp1 = dense1(inputs.view(as: -1, inputs.shape[2]))
            let tmp2 = gelu(tmp1)
            let tmp3 = dropout(tmp2)
            let tmp4 = dense2(tmp3)
            return tmp4.view(as: inputs.shape[0], inputs.shape[1], -1)
        }
    }
}


/// Input to a scaled dot product attention layer
public struct ScaledDotProductAttentionInput<Element: NumericType, Device: DeviceType> {
    var key: Tensor<Element, Device>
    var value: Tensor<Element, Device>
    var query: Tensor<Element, Device>
}


/// Scaled dot product attention layer
public struct ScaledDotProductAttention<Element: RandomizableType, Device: DeviceType>: LayerType {
    public var parameters: [Tensor<Element, Device>] {[]}
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    
    /// Dropout applied before value retreival
    public var dropout: Dropout<Element, Device>
    /// Scale applied after key * query
    public let scale: Tensor<Element, Device>
    /// Determines, whether attention can only span preceding timesteps
    public let isCausal: Bool
    
    /// Determines and controls, whether dropout is applied before value retreival
    public var isDropoutActive: Bool {
        get {dropout.isActive}
        set {dropout.isActive = newValue}
    }
    
    
    /// Creates a scaled dot product attention layer
    /// - Parameters:
    ///   - size: Size of key, value and query vector
    ///   - dropoutRate: Rate of dropout applied before value retreival
    ///   - isCausal: whether attention can only span preceding timesteps
    public init(size: Int, dropoutRate: Float, isCausal: Bool) {
        scale = Tensor(repeating: 1 / Element(size).sqrt(), shape: [])
        dropout = Dropout(rate: dropoutRate)
        self.isCausal = isCausal
    }
    
    public func callAsFunction(_ inputs: ScaledDotProductAttentionInput<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "ScaledDotProductAttention") {
            // inputs: [batchSize, timeSteps, size]
            let tmp1 = inputs.query.broadcastMatrixMultiplied(with: inputs.key, transposeOther: true) * scale
            
            let tmp2: Tensor<Element, Device>
            if isCausal {
                let (queryTimeSteps, keyTimeSteps) = (tmp1.shape[1], tmp1.shape[2])
                let mask = Tensor<Element, Device>(repeating: 1, shape: [queryTimeSteps, keyTimeSteps])
                    .bandMatrix(belowDiagonal: nil, aboveDiagonal: queryTimeSteps - keyTimeSteps)
                    .unsqueezed(at: 0)
                
                tmp2 = tmp1 * mask - (1 - mask) * 1e-10
            } else {
                tmp2 = tmp1
            }
            let tmp3 = softmax(tmp2, axis: 2)
            let tmp4 = dropout(tmp3)
            
            let tmp5 = tmp4.broadcastMatrixMultiplied(with: inputs.value)
            return tmp5
        }
    }
}

/// Attention with multiple heads
public struct MultiHeadAttention<Element: RandomizableType, Device: DeviceType>: LayerType {
    public var parameters: [Tensor<Element, Device>] {
        Array([attention.parameters, inputTransform.parameters, outputTransform.parameters].joined())
    }
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            attention.parameterPaths.map((\Self.attention).appending(path:)),
            inputTransform.parameterPaths.map((\Self.inputTransform).appending(path:)),
            outputTransform.parameterPaths.map((\Self.outputTransform).appending(path:))
        ].joined())
    }
    
    var attention: ScaledDotProductAttention<Element, Device>
    var inputTransform: Dense<Element, Device>
    var outputTransform: Dense<Element, Device>
    
    /// Number of attention heads
    public let headCount: Int
    
    /// Determines and controls, whether dropout is applied before value retreival
    public var isDropoutActive: Bool {
        get {attention.isDropoutActive}
        set {attention.isDropoutActive = newValue}
    }
    
    /// Creates a multi-head attention layer
    /// - Parameters:
    ///   - size: Size of input and output vectors
    ///   - headCount: Number of attention heads
    ///   - dropoutRate: Rate of dropout applied before value retreival
    ///   - isCausal: Determines, whether attention can only span preceding timesteps
    public init(size: Int, headCount: Int, dropoutRate: Float, isCausal: Bool) {
        self.attention = ScaledDotProductAttention(size: size, dropoutRate: dropoutRate, isCausal: isCausal)
        self.inputTransform = Dense(inputSize: size, outputSize: size * 3)
        self.outputTransform = Dense(inputSize: size, outputSize: size)
        self.headCount = headCount
    }
    
    func splitHeads(_ input: Tensor<Element, Device>, headCount: Int) -> Tensor<Element, Device> {
        let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
        let featuresPerHead = features / headCount
        let splitLastDim = input.view(as: [batchSize, timeSteps, headCount, featuresPerHead])
        let movedToFront = splitLastDim.permuted(to: [0, 2, 1, 3])
        return movedToFront.view(as: [batchSize * headCount, timeSteps, featuresPerHead])
    }

    func joinHeads(_ input: Tensor<Element, Device>, headCount: Int) -> Tensor<Element, Device> {
        let (generalizedBatch, timeSteps, featuresPerHead) = (
            input.shape[0], input.shape[1], input.shape[2]
        )
        let batchSize = generalizedBatch / headCount
        let features = featuresPerHead * headCount
        let splitFirstDim = input.view(as: [batchSize, headCount, timeSteps, featuresPerHead])
        let movedToBack = splitFirstDim.permuted(to: [0, 2, 1, 3])
        return movedToBack.view(as: [batchSize, timeSteps, features])
    }
    
    func splitQKV(_ input: Tensor<Element, Device>) -> ScaledDotProductAttentionInput<Element, Device> {
        let featuresPerHead = input.shape[2] / 3
        let query = input[nil, nil, 0 ..< featuresPerHead]
        let key = input[nil, nil, featuresPerHead ..< 2 * featuresPerHead]
        let value = input[nil, nil, 2 * featuresPerHead ..< 3 * featuresPerHead]
        
        return ScaledDotProductAttentionInput(key: key, value: value, query: query)
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "MultiHeadAttention") {
            let tmp1 = inputTransform(inputs.view(as: [-1, inputs.shape[2]])).view(as: [inputs.shape[0], inputs.shape[1], -1])
            let split = splitHeads(tmp1, headCount: headCount)
            let attentionInput = splitQKV(split)
            let attentionOutputs = attention(attentionInput)
            let joined = joinHeads(attentionOutputs, headCount: headCount)
            let output = outputTransform(joined.view(as: [-1, joined.shape[2]])).view(as: [joined.shape[0], joined.shape[1], -1])
            return output
        }
    }
}

public struct TransformerLayer<Element: RandomizableType, Device: DeviceType> {
    public var parameters: [Tensor<Element, Device>] {
        Array([
            selfAttention.parameters,
            selfAttentionDropout.parameters,
            selfAttentionNorm.parameters,
            feedForward.parameters,
            feedForwardDropout.parameters,
            feedForwardNorm.parameters
        ].joined())
    }
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            selfAttention.parameterPaths.map((\Self.selfAttention).appending(path:)),
            selfAttentionDropout.parameterPaths.map((\Self.selfAttentionDropout).appending(path:)),
            selfAttentionNorm.parameterPaths.map((\Self.selfAttentionNorm).appending(path:)),
            feedForward.parameterPaths.map((\Self.feedForward).appending(path:)),
            feedForwardDropout.parameterPaths.map((\Self.feedForwardDropout).appending(path:)),
            feedForwardNorm.parameterPaths.map((\Self.feedForwardNorm).appending(path:))
        ].joined())
    }
    
    var selfAttention: MultiHeadAttention<Element, Device>
    var selfAttentionDropout: Dropout<Element, Device>
    var selfAttentionNorm: LayerNorm<Element, Device>
    var feedForward: TransformerFeedForwardBlock<Element, Device>
    var feedForwardDropout: Dropout<Element, Device>
    var feedForwardNorm: LayerNorm<Element, Device>
    
    public var isDropoutActive: Bool {
        get {
            selfAttention.isDropoutActive
        }
        set {
            selfAttention.isDropoutActive = newValue
            selfAttentionDropout.isActive = newValue
            feedForward.isDropoutActive = newValue
            feedForwardDropout.isActive = newValue
        }
    }

    public init(size: Int, headCount: Int, dropoutRate: Float, isCausal: Bool) {
        selfAttention = MultiHeadAttention(
            size: size,
            headCount: headCount,
            dropoutRate: dropoutRate,
            isCausal: isCausal
        )
        selfAttentionDropout = Dropout(rate: dropoutRate)
        selfAttentionNorm = LayerNorm(inputSize: [size])
        feedForward = TransformerFeedForwardBlock(size: size, hiddenSize: 4 * size, dropoutRate: dropoutRate)
        feedForwardDropout = Dropout(rate: dropoutRate)
        feedForwardNorm = LayerNorm(inputSize: [size])
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Transformer Block") {
            let tmp1 = selfAttentionNorm(inputs.view(as: -1, inputs.shape[2])).view(as: inputs.shape)
            let tmp2 = selfAttention(tmp1)
            let tmp3 = selfAttentionDropout(tmp2)
            let attended = inputs + tmp3
            
            let tmp4 = feedForwardNorm(attended)
            let tmp5 = feedForward(tmp4)
            let tmp6 = feedForwardDropout(tmp5)
            
            let result = attended + tmp6
            return result
        }
    }
}
