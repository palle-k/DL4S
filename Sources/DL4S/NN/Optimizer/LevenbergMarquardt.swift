//
//  LevenbergMarquardt.swift
//  DL4S
//
//  Created by Palle Klewitz on 13.11.19.
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


/// Levenberg Marquardt optizer that uses second derivatives for optimized step sizes.
public struct LevenbergMarquardt<Model: LayerType>: Optimizer {
    public private(set) var model: Model
    private var paths: [WritableKeyPath<Model, Tensor<Model.Parameter, Model.Device>>]
    public var lambda: Tensor<Model.Parameter, Model.Device> = 1
    
    public init(model: Model) {
        self.model = model
        self.paths = model.parameterPaths
    }
    
    public mutating func update(along gradients: [Tensor<LevenbergMarquardt<Model>.Layer.Parameter, LevenbergMarquardt<Model>.Layer.Device>]) {
        precondition(
            gradients.allSatisfy({$0.requiresGradient}),
            "Newton optimizer requires computation graph for backwards pass. Compute gradients with retainBackwardsGraph set to true."
        )
        let secondGrads = zip(model.parameters, gradients).map {
            $1.gradients(of: [$0])[0]
        }
        
        for (path, (firstDerivative, secondDerivative)) in zip(paths, zip(gradients, secondGrads)) {
            let delta = firstDerivative / (secondDerivative * secondDerivative + lambda)
            model[keyPath: path] -= delta
            model[keyPath: path].discardContext()
        }
    }
    
    public mutating func reset() {
        lambda = 1
    }
}
