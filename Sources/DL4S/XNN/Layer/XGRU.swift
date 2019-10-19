//
//  XGRU.swift
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

public struct XGRU<Element: RandomizableType, Device: DeviceType>: XRNN, Codable {
    public typealias Inputs = XTensor<Element, Device>
    public typealias Outputs = (XTensor<Element, Device>, () -> XTensor<Element, Device>)
    
    public var parameterPaths: [WritableKeyPath<Self, XTensor<Element, Device>>] {[
        \.Wz, \.Wr, \.Wh,
        \.Uz, \.Ur, \.Uh,
        \.bz, \.br, \.bh
    ]}
    
    public let direction: RNNDirection
    
    public var Wz: XTensor<Element, Device>
    public var Wr: XTensor<Element, Device>
    public var Wh: XTensor<Element, Device>
    public var Uz: XTensor<Element, Device>
    public var Ur: XTensor<Element, Device>
    public var Uh: XTensor<Element, Device>
    public var bz: XTensor<Element, Device>
    public var br: XTensor<Element, Device>
    public var bh: XTensor<Element, Device>
    
    public var inputSize: Int {
        return Wz.shape[0]
    }
    public var hiddenSize: Int {
        return Wz.shape[1]
    }
    
    public var parameters: [XTensor<Element, Device>] {
        get {[Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh]}
    }
    
    public init(inputSize: Int, hiddenSize: Int, direction: RNNDirection = .forward) {
        self.direction = direction
        
        Wz = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Wr = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Wh = XTensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        Uz = XTensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        Ur = XTensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        Uh = XTensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        bz = XTensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        br = XTensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        bh = XTensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
        
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
    
    public func numberOfSteps(for inputs: XTensor<Element, Device>) -> Int {
        inputs.shape[0]
    }
    
    public func initialState(for inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        XTensor(repeating: 0, shape: [inputs.shape[1], hiddenSize]) // [batchSize, hiddenSize]
    }
    
    public func prepare(inputs: XTensor<Element, Device>) -> (XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>) {
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
    
    public func input(at step: Int, using preparedInput: (XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>)) -> (XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>) {
        let (x_z, x_r, x_h) = preparedInput
        return (x_z[step], x_r[step], x_h[step])
    }
    
    public func step(_ preparedInput: (XTensor<Element, Device>, XTensor<Element, Device>, XTensor<Element, Device>), previousState: XTensor<Element, Device>) -> XTensor<Element, Device> {
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
    
    public func concatenate(_ states: [XTensor<Element, Device>]) -> XTensor<Element, Device> {
        XTensor(stacking: states.map {$0.unsqueezed(at: 0)}, along: 0)
    }
}
