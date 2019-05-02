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
    
    /// Loads pretrained word embeddings from the space / tab separated values file at the given path
    /// and arranges them according to the order of words provided.
    ///
    /// The embeddings are expected to be arranged using the following format:
    ///
    ///     word1 num num num ... num
    ///     word2 num num num ... num
    ///     ...
    ///
    /// If a word is not found in the pretrained embeddings, it is randomly created using Xavier initialization
    ///
    /// - Parameters:
    ///   - words: Provided word order.
    ///   - embeddingsURL: Path to pretrained embeddings
    ///   - verbose: If set to true, print out loading progress
    public init?(words: [String], embeddingsURL: URL, verbose: Bool = false) {
        let wordToIndex = Dictionary(uniqueKeysWithValues: words.enumerated().map{($1, $0)})
        
        var tensors: [Tensor<Element, Device>?] = Array(repeating: nil, count: words.count)
        
        var embedDim: Int? = nil
        
        var progress = verbose ? ProgressBar<()>(totalUnitCount: words.count, formatUserInfo: {""}, label: "loading embeddings") : nil
        
        var completedCount = 0
        
        for line in File(url: embeddingsURL) {
            autoreleasepool {
                let components = line.split(whereSeparator: {$0.isWhitespace})
                
                guard components.count >= 2 else {
                    return
                }
                let word = String(components[0])
                guard let index = wordToIndex[word] else {
                    return
                }
                
                let values = Tensor<Element, Device>(components[1...].compactMap(Double.init).map(Element.init))
                tensors[index] = values.unsqueeze(at: 0)
                embedDim = values.count
                
                completedCount += 1
                
                progress?.next(userInfo: ())
            }
            
            if completedCount == words.count {
                break
            }
        }
        
        progress?.complete()
        
        if verbose {
            let unknownCount = tensors.count(where: {$0 == nil})
            print("Unknown: \(unknownCount) of \(words.count)")
            print("Embedding size: \(embedDim ?? -1)")
        }
        
        guard let shape = embedDim else {
            print("No word from wordlist found in embedding file.")
            return nil
        }
        self.embeddingMatrix = Tensor.stack(
            tensors.map { t in
                if let t = t {
                    return t
                } else {
                    let t = Tensor<Element, Device>(repeating: 0, shape: 1, shape)
                    Random.fillNormal(t, mean: 0, stdev: (2 / Element(shape)).sqrt())
                    return t
                }
            }
        )
        self.embeddingMatrix.requiresGradient = true
        self.inputFeatures = words.count
        self.outputSize = shape
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

