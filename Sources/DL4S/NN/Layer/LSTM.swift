//
//  LSTM.swift
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


public struct LSTM<Element: RandomizableType, Device: DeviceType>: RNN, Codable {
    public typealias Inputs = Tensor<Element, Device>
    public typealias Outputs = (State, () -> State)
    public typealias State = (hiddenState: Tensor<Element, Device>, cellState: Tensor<Element, Device>)
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[
        \.Wi, \.Wo, \.Wf, \.Wc,
        \.Ui, \.Uo, \.Uf, \.Uc,
        \.bi, \.bo, \.bf, \.bc
    ]}
    
    public let direction: RNNDirection
    
    public var Wi: Tensor<Element, Device>
    public var Wo: Tensor<Element, Device>
    public var Wf: Tensor<Element, Device>
    public var Wc: Tensor<Element, Device>
    public var Ui: Tensor<Element, Device>
    public var Uo: Tensor<Element, Device>
    public var Uf: Tensor<Element, Device>
    public var Uc: Tensor<Element, Device>
    public var bi: Tensor<Element, Device>
    public var bo: Tensor<Element, Device>
    public var bf: Tensor<Element, Device>
    public var bc: Tensor<Element, Device>
    
    public var inputSize: Int {
        return Wi.shape[0]
    }
    public var hiddenSize: Int {
        return Wi.shape[1]
    }
    
    public var parameters: [Tensor<Element, Device>] {
        get {[Wi, Wo, Wf, Wc, Ui, Uo, Uf, Uc, bi, bo, bf, bc]}
    }
    
    /// Creates a Long Short-Term Memory (LSTM) layer.
    ///
    /// The RNN expects inputs to have a shape of [sequence length, batch size, input size].
    ///
    /// - Parameters:
    ///   - inputSize: Number of elements at each timestep of the input
    ///   - hiddenSize: Number of elements at each timestep in the output
    ///   - direction: Direction, in which the RNN consumes the input sequence.
    public init(inputSize: Int, hiddenSize: Int, direction: RNNDirection = .forward) {
        self.direction = direction
        
        Wi = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Wo = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Wf = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Wc = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Ui = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        Uo = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        Uf = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        Uc = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        bi = Tensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        bo = Tensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        bf = Tensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        bc = Tensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        
        #if DEBUG
        Wi.tag = "W_i"
        Wo.tag = "W_o"
        Wf.tag = "W_f"
        Wc.tag = "W_c"
        Ui.tag = "U_i"
        Uo.tag = "U_o"
        Uf.tag = "U_f"
        Uc.tag = "U_c"
        bi.tag = "b_i"
        bo.tag = "b_o"
        bf.tag = "b_f"
        bc.tag = "b_c"
        #endif
    }
    
    public func numberOfSteps(for inputs: Tensor<Element, Device>) -> Int {
        inputs.shape[0]
    }
    
    public func initialState(for inputs: Tensor<Element, Device>) -> (hiddenState: Tensor<Element, Device>, cellState: Tensor<Element, Device>) {
        (Tensor(repeating: 0, shape: [inputs.shape[1], hiddenSize]), Tensor(repeating: 0, shape: [inputs.shape[1], hiddenSize]))
    }
    
    public func prepare(inputs: Tensor<Element, Device>) -> (Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>) {
        OperationGroup.capture(named: "LSTMPrepare") {
            let seqlen = inputs.shape[0]
            let batchSize = inputs.shape[1]
            
            let preMulView = [seqlen * batchSize, inputSize]
            let postMulView = [seqlen, batchSize, hiddenSize]
            
            return (
                inputs.view(as: preMulView).matrixMultiplied(with: Wi).view(as: postMulView) + bi,
                inputs.view(as: preMulView).matrixMultiplied(with: Wo).view(as: postMulView) + bo,
                inputs.view(as: preMulView).matrixMultiplied(with: Wf).view(as: postMulView) + bf,
                inputs.view(as: preMulView).matrixMultiplied(with: Wc).view(as: postMulView) + bc
            )
        }
    }
    
    public func input(at step: Int, using preparedInput: (Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>)) -> (Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>) {
        let (x_i, x_o, x_f, x_c) = preparedInput
        return (x_i[step], x_o[step], x_f[step], x_c[step])
    }
    
    public func step(_ preparedInput: (Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>), previousState: State) -> State {
        OperationGroup.capture(named: "LSTMCell") {
            let (x_i, x_o, x_f, x_c) = preparedInput
            
            let h_p = previousState.hiddenState
            let c_p = previousState.cellState
            
            // TODO: Unify W_* matrics, U_* matrices and b_* vectors, perform just two matrix multiplications and one addition, then select slices
            let f_t = sigmoid(x_f + matMul(h_p, Uf))
            let i_t = sigmoid(x_i + matMul(h_p, Ui))
            let o_t = sigmoid(x_o + matMul(h_p, Uo))
            
            let c_t_partial_1 = f_t * c_p + i_t
            let c_t_partial_2 = tanh(x_c + matMul(h_p, Uc))
            let c_t = c_t_partial_1 * c_t_partial_2
            let h_t = o_t * tanh(c_t)
            
            return (h_t, c_t)
        }
    }
    
    public func concatenate(_ states: [State]) -> State {
        (Tensor(stacking: states.map {$0.hiddenState.unsqueezed(at: 0)}, along: 0), Tensor(stacking: states.map {$0.cellState.unsqueezed(at: 0)}, along: 0))
    }
}
