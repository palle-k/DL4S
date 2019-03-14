//
//  Embedding.swift
//  DL4S
//
//  Created by Palle Klewitz on 01.03.19.
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


/// Transforms discrete values, such as word indices, into a lower dimensional embedding.
public class Embedding<Element: RandomizableType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Int32
    
    public var parameters: [Tensor<Element, Device>] {
        return [embeddingMatrix]
    }
    
    public var isTrainable: Bool = true
    
    let embeddingMatrix: Tensor<Element, Device>
    
    
    /// Number of input features
    public let inputFeatures: Int
    /// Size of embedded input features
    public let outputSize: Int
    
    public init(inputFeatures: Int, outputSize: Int) {
        self.inputFeatures = inputFeatures
        self.outputSize = outputSize
        self.embeddingMatrix = Tensor<Element, Device>(repeating: 0, shape: [inputFeatures, outputSize])
        
        Random.fillNormal(embeddingMatrix, mean: 0, stdev: (2 / Element(outputSize)).sqrt())
    }
    
    public func forward(_ inputs: [Tensor<Int32, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        
        let x = inputs[0]
        
        let embedded = (0 ..< x.shape[0]).map { i in
            embeddingMatrix[Int(x[i].item)].unsqueeze(at: 0)
        }
        
        return stack(embedded)
    }
}
