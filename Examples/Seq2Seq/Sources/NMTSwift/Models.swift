//
//  Encoder.swift
//  NMTSwift
//
//  Created by Palle Klewitz on 14.03.19.
//

import Foundation
import DL4S

class Encoder<Element: RandomizableType, Device: DeviceType>: Layer {
    var isTrainable: Bool = true
    
    var parameters: [Tensor<Element, Device>] {
        return embedding.parameters + rnn.parameters
    }
    
    private let embedding: Embedding<Element, Device>
    private let rnn: GRU<Element, Device>
    
    init(inputSize: Int, hiddenSize: Int) {
        self.embedding = Embedding(inputFeatures: inputSize, outputSize: hiddenSize)
        self.rnn = GRU(inputSize: hiddenSize, hiddenSize: hiddenSize, shouldReturnFullSequence: false)
    }
    
    
    func forward(_ inputs: [Tensor<Int32, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        
        let batchSize = 1
        let length = inputs[0].shape[0]
        
        let embedded = self.embedding.forward(inputs[0])
        let rnnIn = embedded.view(as: length, batchSize, -1)
        
        let rnnOut = self.rnn.forward(rnnIn)
        
        return rnnOut
    }
}


class Decoder<Element: RandomizableType, Device: DeviceType>: Layer {
    typealias Input = Int32
    
    var isTrainable: Bool = true
    
    var parameters: [Tensor<Element, Device>] {
        return Array([
            embedding.parameters,
            rnn.parameters,
            dense.parameters,
            softmax.parameters
        ].joined())
    }
    
    var trainableParameters: [Tensor<Element, Device>] {
        return isTrainable ? Array([
            embedding.trainableParameters,
            rnn.trainableParameters,
            dense.trainableParameters,
            softmax.trainableParameters
        ].joined()) : []
    }
    
    private let embedding: Embedding<Element, Device>
    private let rnn: GRU<Element, Device>
    private let dense: Dense<Element, Device>
    private let softmax: Softmax<Element, Device>
    
    init(inputSize: Int, hiddenSize: Int) {
        self.embedding = Embedding(inputFeatures: inputSize, outputSize: hiddenSize)
        self.rnn = GRU(inputSize: hiddenSize, hiddenSize: hiddenSize, shouldReturnFullSequence: true)
        self.dense = Dense(inputFeatures: hiddenSize, outputFeatures: inputSize)
        self.softmax = Softmax()
    }
    
    func forward(_ inputs: [Tensor<Int32, Device>]) -> Tensor<Element, Device> {
        let initialHidden = Tensor<Element, Device>(repeating: 0, shape: 1, self.rnn.hiddenSize) // batchSize x hiddenSize
        return forwardFullSequence(input: inputs[0], initialHidden: initialHidden)
    }
    
    func forward(input: Tensor<Int32, Device>, previousHidden: Tensor<Element, Device>) -> (output: Tensor<Element, Device>, hidden: Tensor<Element, Device>) {
        let embedded = embedding.forward(input.view(as: -1)).view(as: 1, 1, -1)
        let nextHidden = rnn.forward(embedded, previousHidden).view(as: 1, -1)
        let deembedded = dense.forward(nextHidden)
        let probs = softmax.forward(deembedded)
        
        return (probs, nextHidden)
    }
    
    func forwardFullSequence(input: Tensor<Int32, Device>, initialHidden: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let seqlen = input.shape[0]
        let embedded = embedding.forward(input.view(as: -1)).view(as: seqlen, 1, -1)
        // let rectified = relu(embedded)
        let hidden = rnn.forward(embedded, initialHidden).squeeze() // get rid of batch size dimension -> [seqlen x hiddenSize]
        let deembedded = dense.forward(hidden)
        let probs = softmax.forward(deembedded)
        
        return probs
    }
}


