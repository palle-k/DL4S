//
//  File.swift
//  
//
//  Created by Palle Klewitz on 20.09.20.
//  Copyright (c) 2019 - 2020 - Palle Klewitz
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

/// Computes Scaled Multi-Head Dot Product Attention as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
public struct ScaledDotProductAttention<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var temperature: Element
    
    public init(temperature: Element) {
        self.temperature = temperature
    }
    
    public var parameters: [Tensor<Element, Device>] {[]}
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    
    /// Performs scaled dot product attention.
    /// - Parameter inputs: Tuple containing queries of shape [batchSize, heads, queryCount, keyDim], keys of shape [batchSize, heads, keyCount, keyDim] and values of shape [batchSize, heads, valueCount, valueDim]
    ///       as well as an optional mask that may be used to prevent attention to certain elements outside of the batch or in future timesteps. Mask must be broadcastable to shape [batchSize, heads, queryCount, keyCount]
    /// - Returns: Attended values tensor of shape [batchSize, heads, queryCount, valueDim]
    public func callAsFunction(_ inputs: (q: Tensor<Element, Device>, k: Tensor<Element, Device>, v: Tensor<Element, Device>, mask: Tensor<Element, Device>?)) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "ScaledDotProductAttention") {
            let (q, k, v, mask) = inputs
            precondition(k.dim == 4)
            
            // q: [batchSize, heads, queryCount, keyDim]
            // k: [batchSize, heads, keyCount, keyDim]
            // v: [batchSize, heads, valueCount, valueDim]
            
            // [batchSize, heads, queryCount, keyDim] x [batchSize, heads, [keyCount, keyDim]^T] -> [batchSize, heads, queryCount, keyCount]
            var attn = (q / Tensor(temperature)).broadcastMatrixMultiplied(with: k, transposeSelf: false, transposeOther: true)
            
            if let mask = mask {
                attn = attn - mask * 1e9 // mask contains 1 for all entries that should be masked away ==> softmax zeros them out.
            }
            
            attn = softmax(attn, axis: 3) // softmax over last axis (keyCount)
            // [batchSize, heads, queryCount, keyCount] x [batchSize, heads, valueCount, valueDim] -> [batchSize, heads, queryCount, valueDim] (constraint: keyCount == valueCount]
            let output = attn.broadcastMatrixMultiplied(with: v)
            return output // [batchSize, heads, queryCount, valueDim]
        }
    }
}
