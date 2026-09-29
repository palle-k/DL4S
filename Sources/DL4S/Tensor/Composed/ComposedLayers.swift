//
//  ComposedLayers.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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

// MARK: Composed gradients

extension Composed {
    static func linearBackward<N, Device>(
        input: Tensor<N, Device>,
        weights: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        inputGradient: inout GradientAccumulator<N, Device>,
        weightGradient: inout GradientAccumulator<N, Device>,
        biasGradient: inout GradientAccumulator<N, Device>,
    ) {
        // Without a gradient graph, the products are added to the accumulated gradients in place,
        // so a weight that is used several times needs no temporary gradient.
        if inputGradient.isRequested {
            inputGradient.add(outputGradient.matrixMultiplied(with: weights, transposeOther: true))
        }
        if weightGradient.isRequested {
            weightGradient.add(input.matrixMultiplied(with: outputGradient, transposeSelf: true))
        }
        if biasGradient.isRequested {
            biasGradient.add(outputGradient.reduceSum(along: [0]))
        }
    }

    /// The mask is a constant, so the gradient is differentiable with respect to the gradient of the result only.
    static func dropoutBackward<N, Device>(mask: Tensor<N, Device>, outputGradient: Tensor<N, Device>, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        inputGradient.add(outputGradient * mask)
    }
}
