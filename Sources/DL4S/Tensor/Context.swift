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
//

import Foundation

@usableFromInline
struct TensorContext<Element: NumericType, Device: DeviceType> {
    /// Describes how the gradient of the result flows back to the sources of an operation.
    @usableFromInline
    enum BackpropagateFunction {
        /// One closure per source.
        ///
        /// Each closure receives the gradient of the result and the accumulated gradient of its source.
        /// It adds the gradient of the result with respect to its source to the accumulator and returns the sum.
        case perSource([(Tensor<Element, Device>, consuming Tensor<Element, Device>?) -> Tensor<Element, Device>])
        
        /// One closure for all sources.
        ///
        /// The closure receives the gradient of the result and one accumulator per source.
        /// It returns the accumulated gradient of every source in source order.
        /// Operations use this form when one kernel produces the gradients of all sources at once,
        /// so the kernel runs once per backward pass and no state needs to be shared between closures.
        case allSources((Tensor<Element, Device>, consuming [Tensor<Element, Device>?]) -> [Tensor<Element, Device>])
    }
    
    var tag: String?
    var sources: [Tensor<Element, Device>]
    var backpropagate: BackpropagateFunction
    #if DEBUG
    var operationStack = OperationGroup.operationStack
    #endif
    
    init(tag: String?, sources: [Tensor<Element, Device>], backpropagate: [(Tensor<Element, Device>) -> Tensor<Element, Device>]) {
        self.init(tag: tag, sources: sources, backpropagateAccumulate: backpropagate.map { function in
            { resultGradient, accumulator in
                let gradient = function(resultGradient)
                return accumulator.map { $0 + gradient } ?? gradient
            }
        })
    }
    
    init(tag: String?, sources: [Tensor<Element, Device>], backpropagateAccumulate: [(Tensor<Element, Device>, consuming Tensor<Element, Device>?) -> Tensor<Element, Device>]) {
        self.tag = tag
        self.sources = sources
        self.backpropagate = .perSource(backpropagateAccumulate)
    }
    
    /// Creates a context with one backpropagation closure for all sources.
    ///
    /// - Parameters:
    ///   - tag: Name of the operation for graph output.
    ///   - sources: Tensors that the operation reads.
    ///   - backpropagateAll: Closure that receives the gradient of the result and owns one accumulator per source, and returns the accumulated gradient of every source.
    init(tag: String?, sources: [Tensor<Element, Device>], backpropagateAll: @escaping (Tensor<Element, Device>, consuming [Tensor<Element, Device>?]) -> [Tensor<Element, Device>]) {
        self.tag = tag
        self.sources = sources
        self.backpropagate = .allSources(backpropagateAll)
    }
}
