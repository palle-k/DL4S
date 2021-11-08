//
//  GRU.swift
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

public struct GRU<Element: RandomizableType, Device: DeviceType>: RNN, Codable {
    public typealias Inputs = Tensor<Element, Device>
    public typealias Outputs = (Tensor<Element, Device>, () -> Tensor<Element, Device>)
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[
        \.Wz, \.Wr, \.Wh,
        \.Uz, \.Ur, \.Uh,
        \.bz, \.br, \.bh
    ]}
    
    public let direction: RNNDirection
    
    public var Wz: Tensor<Element, Device>
    public var Wr: Tensor<Element, Device>
    public var Wh: Tensor<Element, Device>
    public var Uz: Tensor<Element, Device>
    public var Ur: Tensor<Element, Device>
    public var Uh: Tensor<Element, Device>
    public var bz: Tensor<Element, Device>
    public var br: Tensor<Element, Device>
    public var bh: Tensor<Element, Device>
    
    /// Size of inputs of the layer
    public var inputSize: Int {
        return Wz.shape[0]
    }
    
    /// Size of outputs of the layer
    public var hiddenSize: Int {
        return Wz.shape[1]
    }
    
    public var parameters: [Tensor<Element, Device>] {
        get {[Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh]}
    }
    
    /// Creates a Gated Recurrent Unit layer.
    ///
    /// The RNN expects inputs to have a shape of [sequence length, batch size, input size].
    /// 
    /// - Parameters:
    ///   - inputSize: Number of elements at each timestep of the input
    ///   - hiddenSize: Number of elements at each timestep in the output
    ///   - direction: Direction, in which the RNN consumes the input sequence.
    public init(inputSize: Int, hiddenSize: Int, direction: RNNDirection = .forward) {
        self.direction = direction
        
        Wz = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Wr = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Wh = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Uz = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        Ur = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        Uh = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        bz = Tensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        br = Tensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        bh = Tensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        
        #if DEBUG
        Wz.tag = "W_z"
        Wr.tag = "W_r"
        Wh.tag = "W_h"
        Uz.tag = "U_z"
        Ur.tag = "U_r"
        Uh.tag = "U_h"
        bz.tag = "b_z"
        br.tag = "b_r"
        bh.tag = "b_h"
        #endif
    }
    
    public func numberOfSteps(for inputs: Tensor<Element, Device>) -> Int {
        inputs.shape[0]
    }
    
    public func initialState(for inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        Tensor(repeating: 0, shape: [inputs.shape[1], hiddenSize]) // [batchSize, hiddenSize]
    }
    
    public func prepare(inputs: Tensor<Element, Device>) -> (Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>) {
        OperationGroup.capture(named: "GRUPrepare") {
            let seqlen = inputs.shape[0]
            let batchSize = inputs.shape[1]
            
            let preMulView = [seqlen * batchSize, inputSize]
            let postMulView = [seqlen, batchSize, hiddenSize]
            
            return (
                inputs.view(as: preMulView).matrixMultiplied(with: Wz).view(as: postMulView) + bz,
                inputs.view(as: preMulView).matrixMultiplied(with: Wr).view(as: postMulView) + br,
                inputs.view(as: preMulView).matrixMultiplied(with: Wh).view(as: postMulView) + bh
            )
        }
    }
    
    public func input(at step: Int, using preparedInput: (Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>)) -> (Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>) {
        let (x_z, x_r, x_h) = preparedInput
        return (x_z[step], x_r[step], x_h[step])
    }
    
    public func step(_ preparedInput: (Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>), previousState: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "GRUCell") {
            let (x_z, x_r, x_h) = preparedInput
            
            let h_p = previousState.view(as: [x_z.shape[0], hiddenSize])
            
            let z_t = sigmoid(x_z + matMul(h_p, Uz))
            let r_t = sigmoid(x_r + matMul(h_p, Ur))
            
            let h_t_partial_1 = (1 - z_t) * h_p
            let h_t_partial_2 = tanh(x_h + matMul(r_t * h_p, Uh))
            
            let h_t = h_t_partial_1 + z_t * h_t_partial_2
            
            return h_t
        }
    }
    
    public func concatenate(_ states: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        Tensor(stacking: states.map {$0.unsqueezed(at: 0)}, along: 0)
    }
}
