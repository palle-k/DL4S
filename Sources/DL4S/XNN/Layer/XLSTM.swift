//
//  XLSTM.swift
//  DL4S
//
//  Created by Palle Klewitz on 17.10.19.
//

import Foundation


public struct XLSTM<Element: RandomizableType, Device: DeviceType>: XRNN, Codable {
    public typealias Inputs = XTensor<Element, Device>
    public typealias Outputs = (State, () -> State)
    public typealias State = (hiddenState: XTensor<Element, Device>, cellState: XTensor<Element, Device>)
    
    public static var parameters: [WritableKeyPath<Self, XTensor<Element, Device>>] {[
        \.Wi, \.Wo, \.Wf, \.Wc,
        \.Ui, \.Uo, \.Uf, \.Uc,
        \.bi, \.bo, \.bf, \.bc
    ]}
    
    public let direction: RNNDirection
    
    public var Wi: XTensor<Element, Device>
    public var Wo: XTensor<Element, Device>
    public var Wf: XTensor<Element, Device>
    public var Wc: XTensor<Element, Device>
    public var Ui: XTensor<Element, Device>
    public var Uo: XTensor<Element, Device>
    public var Uf: XTensor<Element, Device>
    public var Uc: XTensor<Element, Device>
    public var bi: XTensor<Element, Device>
    public var bo: XTensor<Element, Device>
    public var bf: XTensor<Element, Device>
    public var bc: XTensor<Element, Device>
    
    public var inputSize: Int {
        return Wi.shape[0]
    }
    public var hiddenSize: Int {
        return Wi.shape[1]
    }
    
    public var parameters: [XTensor<Element, Device>] {
        get {[Wi, Wo, Wf, Wc, Ui, Uo, Uf, Uc, bi, bo, bf, bc]}
        set {(Wi, Wo, Wf, Wc, Ui, Uo, Uf, Uc, bi, bo, bf, bc) = (newValue[0], newValue[1], newValue[2], newValue[3], newValue[4], newValue[5], newValue[6], newValue[7], newValue[8], newValue[9], newValue[10], newValue[11])}
    }
    
    public init(inputSize: Int, hiddenSize: Int, direction: RNNDirection = .forward) {
        self.direction = direction
        
        Wi = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Wo = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Wf = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Wc = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Ui = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        Uo = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        Uf = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        Uc = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        bi = XTensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        bo = XTensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        bf = XTensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        bc = XTensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        
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
    
    public func numberOfSteps(for inputs: XTensor<Element, Device>) -> Int {
        inputs.shape[0]
    }
    
    public func initialState(for inputs: XTensor<Element, Device>) -> (hiddenState: XTensor<Element, Device>, cellState: XTensor<Element, Device>) {
        (XTensor(repeating: 0, shape: [inputs.shape[1], hiddenSize]), XTensor(repeating: 0, shape: [inputs.shape[1], hiddenSize]))
    }
    
    public func prepare(inputs: XTensor<Element, Device>) -> (XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>) {
        OperationGroup.capture(named: "LSTMPrepare") {
            let seqlen = inputs.shape[0]
            let batchSize = inputs.shape[1]
            
            let preMulView = [seqlen * batchSize, inputSize]
            let postMulView = [seqlen, batchSize, hiddenSize]
            
            return (
                inputs.view(as: preMulView).matMul(Wi).view(as: postMulView) + bi,
                inputs.view(as: preMulView).matMul(Wo).view(as: postMulView) + bo,
                inputs.view(as: preMulView).matMul(Wf).view(as: postMulView) + bf,
                inputs.view(as: preMulView).matMul(Wc).view(as: postMulView) + bc
            )
        }
    }
    
    public func input(at step: Int, using preparedInput: (XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>)) -> (XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>) {
        let (x_i, x_o, x_f, x_c) = preparedInput
        return (x_i[step], x_o[step], x_f[step], x_c[step])
    }
    
    public func step(_ preparedInput: (XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>), previousState: State) -> State {
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
        (XTensor(stacking: states.map {$0.hiddenState.unsqueezed(at: 0)}, along: 0), XTensor(stacking: states.map {$0.cellState.unsqueezed(at: 0)}, along: 0))
    }
}
