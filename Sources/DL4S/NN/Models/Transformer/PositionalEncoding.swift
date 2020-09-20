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

/// Positional Encoding layer using the encoding method proposed in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
///
/// The layer takes an array of Ints as an input, which indicate the number of elements in each sequence of the minibatch.
/// It returns a tensor with the shape [max(inputs), hiddenSize].
/// It does not mask out positional encodings for padding elements.
public struct PositionalEncoding<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public typealias Parameter = Element
    
    /// Number of elements in the positional encoding output tensor.
    public let hiddenSize: Int
    
    /// Creates a Positional Encoding layer using the encoding method proposed in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameter hiddenSize: Number of elements in the positional encoding output tensor.
    public init(hiddenSize: Int) {
        precondition(hiddenSize.isMultiple(of: 2), "Hidden size must be multiple of 2")
        self.hiddenSize = hiddenSize
    }
    
    public var parameters: [Tensor<Element, Device>] {[]}
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    
    /// Creates a positional encoding matrix
    /// - Parameter maxLen: Maximum sequence length in the current minibatch
    /// - Returns: Tensor of shape [max(inputs), hiddenSize]
    public func callAsFunction(_ maxLen: Int) -> Tensor<Element, Device> {
        let inputRange = Tensor<Element, Device>((0 ..< maxLen).map(Element.init))
        
        let hiddenRange = Tensor<Element, Device>((0 ..< hiddenSize / 2).map(Element.init))
        let frequencies = Tensor(10000).raised(toPowerOf: hiddenRange / Tensor(Element(hiddenSize / 2)))
        let samplePoints = inputRange.unsqueezed(at: 1) / frequencies.unsqueezed(at: 0) // [seqlen, hiddenSize / 2]
        
        var samples = Tensor(
            stacking: [
                sin(samplePoints).unsqueezed(at: 2),
                cos(samplePoints).unsqueezed(at: 2)
            ],
            along: 2
        ).view(as: -1, hiddenSize) // [seqlen, hiddenSize]
        
        #if DEBUG
        samples.tag = "PositionEncodings"
        #endif
        return samples // [seqlen, hiddensize]
    }
}
