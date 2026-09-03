//
//  File.swift
//  
//
//  Created by Palle Klewitz on 20.09.20.
//  Copyright (c) 2020 - Palle Klewitz
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

/// A layer that normalizes its inputs along the trailing dimensions given by `inputSize`.
///
/// Every leading dimension of the input is treated as an independent sample. With `inputSize: [hidden]`,
/// a `[batch, sequence, hidden]` input is normalized per sequence element.
public struct LayerNorm<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<LayerNorm<Element, Device>, Tensor<Element, Device>> & Sendable] {
        [\.shift, \.scale]
    }
    public var parameters: [Tensor<Element, Device>] {
        get {[shift, scale]}
    }

    /// Learned shift vector
    public var shift: Tensor<Element, Device>

    /// Learned scale vector
    public var scale: Tensor<Element, Device>

    /// A layer that normalizes its inputs along the trailing dimensions given by `inputSize`.
    /// - Parameter inputSize: Shape of the normalized trailing dimensions of the input.
    public init(inputSize: [Int]) {
        shift = Tensor(repeating: 0, shape: inputSize, requiresGradient: true)
        scale = Tensor(repeating: 1, shape: inputSize, requiresGradient: true)

        #if DEBUG
        shift.tag = "shift"
        scale.tag = "scale"
        #endif
    }

    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "LayerNorm") {
            let x = inputs
            let sampleDim = x.dim - shift.dim
            let axes = Array(sampleDim ..< x.dim)
            let statisticsShape = Array(x.shape.prefix(sampleDim)) + Array(repeating: 1, count: shift.dim)
            let mean = x
                .reduceMean(along: axes)
                .view(as: statisticsShape)

            let variance = x
                .variance(along: axes)
                .view(as: statisticsShape)

            let normalized = (x - mean) / (sqrt(variance) + 1e-5)
            return normalized * scale + shift
        }
    }
}
