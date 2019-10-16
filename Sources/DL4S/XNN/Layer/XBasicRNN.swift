//
//  XBasicRNN.swift
//  DL4S
//
//  Created by Palle Klewitz on 17.10.19.
//

import Foundation

public struct XBasicRNN<Element: RandomizableType, Device: DeviceType>: XRNN, Codable {
    public typealias Inputs = XTensor<Element, Device>
    public typealias Outputs = (XTensor<Element, Device>, () -> XTensor<Element, Device>)
    
    public static var parameters: [WritableKeyPath<Self, XTensor<Element, Device>>] {[
        \.W, \.U, \.b
    ]}
    
    public let direction: RNNDirection
    
    public var W: XTensor<Element, Device>
    public var U: XTensor<Element, Device>
    public var b: XTensor<Element, Device>
    
    public var inputSize: Int {
        return W.shape[0]
    }
    public var hiddenSize: Int {
        return W.shape[1]
    }
    
    public var parameters: [XTensor<Element, Device>] {
        get {[W, U, b]}
        set {(W, U, b) = (newValue[0], newValue[1], newValue[2])}
    }
    
    public init(inputSize: Int, hiddenSize: Int, direction: RNNDirection = .forward) {
        self.direction = direction
        
        W = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        U = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        b = XTensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
    }
    
    public func initialState(for inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        XTensor(repeating: 0, shape: [inputs.shape[1], hiddenSize]) // [batchSize, hiddenSize]
    }
    
    public func prepare(inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        OperationGroup.capture(named: "BasicRNNPrepare") {
            let seqlen = inputs.shape[0]
            let batchSize = inputs.shape[1]
            
            let multiplied = inputs
                .view(as: [seqlen * batchSize, inputSize])
                .matMul(W)
                .view(as: [seqlen, batchSize, hiddenSize])
            
            return multiplied + b
        }
    }
    
    public func input(at step: Int, using preparedInput: XTensor<Element, Device>) -> XTensor<Element, Device> {
        preparedInput[step]
    }
    
    public func step(_ preparedInput: XTensor<Element, Device>, previousState: XTensor<Element, Device>) -> XTensor<Element, Device> {
        OperationGroup.capture(named: "BasicRNNCell") {
            tanh(preparedInput + previousState.matMul(U))
        }
    }
    
    public func concatenate(_ states: [XTensor<Element, Device>]) -> XTensor<Element, Device> {
        XTensor(stacking: states.map {$0.unsqueezed(at: 0)}, along: 0)
    }
    
    public func numberOfSteps(for inputs: XTensor<Element, Device>) -> Int {
        inputs.shape[0]
    }
}
