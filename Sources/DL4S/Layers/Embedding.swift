//
//  Embedding.swift
//  DL4S
//
//  Created by Palle Klewitz on 01.03.19.
//

import Foundation


public class Embedding<Element: RandomizableType>: Layer, Codable {
    public typealias Input = Int32
    
    public var parameters: [Tensor<Element>] {
        return trainable ? [embeddingMatrix] : []
    }
    
    public var trainable: Bool = true
    
    let embeddingMatrix: Tensor<Element>
    
    public let inputFeatures: Int
    public let outputSize: Int
    
    public init(inputFeatures: Int, outputSize: Int) {
        self.inputFeatures = inputFeatures
        self.outputSize = outputSize
        self.embeddingMatrix = Tensor<Element>(repeating: 0, shape: [inputFeatures, outputSize])
        
        Random.fillNormal(embeddingMatrix, mean: 0, stdev: (2 / Element(outputSize)).sqrt())
    }
    
    public func forward(_ inputs: [Tensor<Int32>]) -> Tensor<Element> {
        precondition(inputs.count == 1)
        
        let x = inputs[0]
        
        let embedded = (0 ..< x.shape[0]).map { i in
            embeddingMatrix[Int(x[i].item)].unsqueeze(at: 0)
        }
        
        return stack(embedded)
    }
}
