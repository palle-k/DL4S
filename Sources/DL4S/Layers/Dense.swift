//
//  Dense.swift
//  DL4S
//
//  Created by Palle Klewitz on 01.03.19.
//

import Foundation


/// Dense / Linear / Fully connected layer
public class Dense<Element: RandomizableType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    /// Weight matrix
    let w: Tensor<Element, Device>
    
    /// Bias vector
    let b: Tensor<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        return [w, b]
    }
    
    public var isTrainable: Bool = true
    
    /// Number of features in each input
    public var inputFeatures: Int {
        return w.shape[0]
    }
    
    /// Number of features in each output of the layer
    public var outputFeatures: Int {
        return w.shape[1]
    }
    
    
    /// Initializes a dense layer with the given number of input and output features and initializes the weights using Xavier 2/n initialization.
    ///
    /// - Parameters:
    ///   - inputFeatures: Number of input features
    ///   - outputFeatures: Number of output features
    public init(inputFeatures: Int, outputFeatures: Int) {
        w = Tensor(repeating: 0.5, shape: [inputFeatures, outputFeatures], requiresGradient: true)
        b = Tensor(repeating: 0, shape: [outputFeatures], requiresGradient: true)
        
        Random.fillNormal(w, mean: 0, stdev: (2 / Element(inputFeatures)).sqrt())
        
        w.tag = "W"
        b.tag = "b"
    }
    
    
    /// Performs a feed forward operation on a batch of samples with the shape [batchSize x inputSize] and returns
    /// a batch of samples with the shape [batchSize x outputSize].
    ///
    /// - Parameter inputs: Batch of input samples
    /// - Returns: Batch of forwarded samples.
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        let out = mmul(inputs[0], w) + b
        return out
    }
}
