//
//  XContext.swift
//  DL4S
//
//  Created by Palle Klewitz on 19.10.19.
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

struct TensorContext<Element: NumericType, Device: DeviceType> {
    var tag: String?
    var sources: [Tensor<Element, Device>]
    var backpropagate: [(Tensor<Element, Device>, Tensor<Element, Device>?) -> Tensor<Element, Device>]
    #if DEBUG
    var operationStack = OperationGroup.operationStack
    #endif
    
    init(tag: String?, sources: [Tensor<Element, Device>], backpropagate: [(Tensor<Element, Device>) -> Tensor<Element, Device>]) {
        self.init(tag: tag, sources: sources, backpropagateAccumulate: backpropagate.map { function in
            { resultGradient, accumulator in
                if let acc = accumulator {
                    return acc + function(resultGradient)
                } else {
                    return function(resultGradient)
                }
            }
        })
    }
    
    init(tag: String?, sources: [Tensor<Element, Device>], backpropagateAccumulate: [(Tensor<Element, Device>, Tensor<Element, Device>?) -> Tensor<Element, Device>]) {
        self.tag = tag
        self.sources = sources
        self.backpropagate = backpropagateAccumulate
    }
}
