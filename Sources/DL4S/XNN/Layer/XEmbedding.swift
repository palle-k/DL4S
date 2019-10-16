//
//  XEmbedding.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.10.19.
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
public struct XEmbedding<Element: RandomizableType, Device: DeviceType>: XLayer, Codable {
    public static var parameters: [WritableKeyPath<XEmbedding<Element, Device>, XTensor<XEmbedding<Element, Device>.Parameter, XEmbedding<Element, Device>.Device>>] {[
        \.embeddingMatrix
    ]}
    
    public var parameters: [XTensor<Element, Device>] {
        get {[embeddingMatrix]}
        set {embeddingMatrix = newValue[0]}
    }
    
    public var embeddingMatrix: XTensor<Element, Device>
    
    /// Number of input features
    public var inputFeatures: Int {
        embeddingMatrix.shape[0]
    }
    
    /// Size of embedded feature vectors
    public var outputSize: Int {
        embeddingMatrix.shape[1]
    }
    
    public init(inputFeatures: Int, outputSize: Int) {
        self.embeddingMatrix = XTensor<Element, Device>(xavierNormalWithShape: [inputFeatures, outputSize], requiresGradient: true)
        #if DEBUG
        self.embeddingMatrix.tag = "W"
        #endif
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
        
        var tensors: [XTensor<Element, Device>?] = Array(repeating: nil, count: words.count)
        
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
                
                let values = XTensor<Element, Device>(components[1...].compactMap(Double.init).map(Element.init))
                tensors[index] = values.unsqueezed(at: 0)
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
        self.embeddingMatrix = XTensor(
            stacking: tensors.map { t in
                if let t = t {
                    return t
                } else {
                    return XTensor<Element, Device>(xavierNormalWithShape: [1, shape])
                }
            },
            along: 0
        )
        self.embeddingMatrix.requiresGradient = true
        #if DEBUG
        self.embeddingMatrix.tag = "W"
        #endif
    }
    
    public func callAsFunction(_ inputs: XTensor<Int32, Device>) -> XTensor<Element, Device> {
        OperationGroup.capture(named: "Embedding") {
            precondition(inputs.count == 1)
            
            let embedded = (0 ..< inputs.shape[0]).map { i in
                embeddingMatrix[Int(inputs[i].item)].unsqueezed(at: 0)
            }
            
            return XTensor(stacking: embedded, along: 0)
        }
    }
}

