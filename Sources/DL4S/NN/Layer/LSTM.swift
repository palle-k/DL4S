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

/// State of a Long Short-Term Memory (LSTM) layer.
public struct LSTMState<Element: NumericType, Device: DeviceType>: Sendable {
    /// Hidden state, with the shape [batch size, hidden size] for one step or [sequence length, batch size, hidden size] for a sequence
    public var hiddenState: Tensor<Element, Device>

    /// Cell state, with the shape of the hidden state
    public var cellState: Tensor<Element, Device>

    /// Creates an LSTM state.
    /// - Parameters:
    ///   - hiddenState: Hidden state
    ///   - cellState: Cell state, with the shape of the hidden state
    public init(hiddenState: Tensor<Element, Device>, cellState: Tensor<Element, Device>) {
        self.hiddenState = hiddenState
        self.cellState = cellState
    }
}

/// A Long Short-Term Memory (LSTM) layer.
///
/// For every step, the layer computes `c_t = f_t * c_(t-1) + i_t * tanh(x_t W_c + h_(t-1) U_c + b_c)` and
/// `h_t = o_t * tanh(c_t)`, with the forget gate `f_t`, the input gate `i_t`, and the output gate `o_t`.
@Layer
public struct LSTM<Element: RandomizableType, Device: DeviceType>: RNN, Codable, Sendable {
    public typealias Inputs = Tensor<Element, Device>
    public typealias Outputs = (State, () -> State)
    public typealias State = LSTMState<Element, Device>

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
        Wi.shape[0]
    }

    public var hiddenSize: Int {
        Wi.shape[1]
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
        var generator = WyHash()
        self.init(inputSize: inputSize, hiddenSize: hiddenSize, direction: direction, using: &generator)
    }

    /// Creates a Long Short-Term Memory (LSTM) layer.
    ///
    /// The RNN expects inputs to have a shape of [sequence length, batch size, input size].
    ///
    /// - Parameters:
    ///   - inputSize: Number of elements at each timestep of the input
    ///   - hiddenSize: Number of elements at each timestep in the output
    ///   - direction: Direction, in which the RNN consumes the input sequence.
    ///   - generator: Random number generator that provides the initial weights.
    public init<Generator: RandomNumberGenerator>(inputSize: Int, hiddenSize: Int, direction: RNNDirection = .forward, using generator: inout Generator) {
        self.direction = direction

        Wi = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true, using: &generator)
        Wo = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true, using: &generator)
        Wf = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true, using: &generator)
        Wc = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true, using: &generator)
        Ui = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true, using: &generator)
        Uo = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true, using: &generator)
        Uf = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true, using: &generator)
        Uc = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true, using: &generator)
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

    public func initialState(for inputs: Tensor<Element, Device>) -> State {
        State(hiddenState: Tensor(repeating: 0, shape: [inputs.shape[1], hiddenSize]), cellState: Tensor(repeating: 0, shape: [inputs.shape[1], hiddenSize]))
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
                inputs.view(as: preMulView).matrixMultiplied(with: Wc).view(as: postMulView) + bc,
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

            let c_t = f_t * c_p + i_t * tanh(x_c + matMul(h_p, Uc))
            let h_t = o_t * tanh(c_t)

            return State(hiddenState: h_t, cellState: c_t)
        }
    }

    public func concatenate(_ states: [State]) -> State {
        State(
            hiddenState: Tensor(stacking: states.map { $0.hiddenState.unsqueezed(at: 0) }, along: 0),
            cellState: Tensor(stacking: states.map { $0.cellState.unsqueezed(at: 0) }, along: 0),
        )
    }
}
