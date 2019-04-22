//
//  Layers.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.02.19.
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



/// Protocol for an arbitrary neural network layer
@dynamicCallable
public protocol Layer {
    /// Type of elements of input tensor
    associatedtype Input: NumericType
    /// Type of elements of output tensor
    associatedtype Element: NumericType
    /// Device type
    associatedtype Device: DeviceType
    
    /// Parameters of the layer.
    var parameters: [Tensor<Element, Device>] { get }
    
    
    /// Parameters of the layer that should be optimized.
    ///
    /// If `isTrainable` is false, the layer should not return any parameters here.
    var trainableParameters: [Tensor<Element, Device>] { get }
    
    /// Determines or sets whether the layer is trainable and its weights will be updated
    /// in the optimization phase.
    var isTrainable: Bool { get nonmutating set }
    
    
    /// Performs the layer operations on the given inputs.
    /// For most layers, inputs typically contains only one Tensor.
    ///
    /// - Parameter inputs: Inputs to the layer
    /// - Returns: Computed output value
    func forward(_ inputs: [Tensor<Input, Device>]) -> Tensor<Element, Device>
}

public extension Layer {
    /// Performs the layer operations on the given inputs.
    /// For most layers, inputs typically contains only one Tensor.
    ///
    /// - Parameter inputs: Inputs to the layer
    /// - Returns: Computed output value
    func forward(_ inputs: Tensor<Input, Device>...) -> Tensor<Element, Device> {
        return forward(inputs)
    }
    
    /// Erases the type of the layer so that it can be used in a sequential network together with other layer types.
    ///
    /// - Returns: Type erased layer
    func asAny() -> AnyLayer<Input, Element, Device> {
        return AnyLayer(self)
    }
    
    /// Trainable parameters of the layer
    var trainableParameters: [Tensor<Element, Device>] {
        return isTrainable ? parameters : []
    }
    
    /// Writes all parameters of the layer to the provided URL as a JSON file
    ///
    /// - Parameter url: Location to store the parameters at
    /// - Throws: An error if the parameters could not be encoded (EncodingError) or if the data could not be written to the given URL.
    func saveWeights(to url: URL) throws {
        try autoreleasepool {
            let params = self.parameters
            let encoder = JSONEncoder()
            let encoded = try encoder.encode(params)
            try encoded.write(to: url, options: .atomic)
        }
    }
    
    
    /// Loads all parameters of the layer from a JSON file at the provided URL
    ///
    /// - Parameter url: Location to read the parameters from
    /// - Throws: An error if the parametes could not be decoded (Decodingerror) or if no data could be read from the given URL.
    func loadWeights(from url: URL) throws {
        let params = self.parameters
        
        let decodedParams = try autoreleasepool { () -> [Tensor<Element, Device>] in
            let decoder = JSONDecoder()
            let encoded = try Data(contentsOf: url)
            return try decoder.decode([Tensor<Element, Device>].self, from: encoded)
        }
        
        for (dst, src) in zip(params, decodedParams) {
            Device.Memory.assign(from: src.values, to: dst.values, count: dst.count)
            
            if let dstGrad = dst.gradient, let srcGrad = src.gradient {
                Device.Memory.assign(from: srcGrad, to: dstGrad, count: dst.count)
            }
        }
    }
    
    func dynamicallyCall(withArguments args: [Tensor<Input, Device>]) -> Tensor<Element, Device> {
        return forward(args)
    }
}


/// Type erased layer that enables the creation of heterogenous sequential models
public struct AnyLayer<Input: NumericType, Element: NumericType, Device: DeviceType>: Layer {
    private let getParams: () -> [Tensor<Element, Device>]
    private let getTrainableParams: () -> [Tensor<Element, Device>]
    private let performForward: ([Tensor<Input, Device>]) -> (Tensor<Element, Device>)
    private let getTrainable: () -> Bool
    private let setTrainable: (Bool) -> ()
    
    public var parameters: [Tensor<Element, Device>] {
        return getParams()
    }
    
    public var trainableParameters: [Tensor<Element, Device>] {
        return getTrainableParams()
    }
    
    public var isTrainable: Bool {
        get {
            return getTrainable()
        }
        nonmutating set {
            setTrainable(newValue)
        }
    }
    
    /// Initializes a wrapper around the provided layer to hide its real type.
    ///
    /// - Parameters: layer: Wrapped layer
    public init<L: Layer>(_ layer: L) where L.Element == Element, L.Input == Input, L.Device == Device {
        self.getParams = {layer.parameters}
        self.getTrainableParams = {layer.trainableParameters}
        self.performForward = {layer.forward($0)}
        self.getTrainable = {layer.isTrainable}
        self.setTrainable = {layer.isTrainable = $0}
    }
    
    public func forward(_ inputs: [Tensor<Input, Device>]) -> Tensor<Element, Device> {
        return performForward(inputs)
    }
}


/// Layer that concatenates the forward operations of multiple wrapped layers.
public class Sequential<Element: NumericType, Device: DeviceType>: Layer {
    public typealias Input = Element
    
    
    /// Layers that are combined by the sequential layer
    public var layers: [AnyLayer<Element, Element, Device>]
    
    public var isTrainable: Bool {
        get {
            return layers.contains(where: {$0.isTrainable})
        }
        set {
            layers.forEach {$0.isTrainable = newValue}
        }
    }
    
    public var parameters: [Tensor<Element, Device>] {
        return layers.flatMap {$0.parameters}
    }
    
    
    public var trainableParameters: [Tensor<Element, Device>] {
        return layers.flatMap {$0.trainableParameters}
    }
    
    
    /// Initializes a sequential layer that concatenates the forward operations of the provided source layers.
    ///
    /// Layers have to be type erased by using the `.asAny()` operator (because Swift does not support generalized existentials (yet)).
    ///
    /// - Parameter layers: Wrapped layers.
    public init(_ layers: AnyLayer<Element, Element, Device>...) {
        self.layers = layers
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        return layers.reduce(inputs) {[$1.forward($0)]}[0]
    }
    
    public func append<L: Layer>(_ layer: L) where L.Input == Element, L.Device == Device, L.Element == Element {
        self.layers.append(layer.asAny())
    }
}


/// Auxiliary layer that logs every n batches forwarded through a network
public class Logging<Element: NumericType, Device: DeviceType>: Layer {
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var isTrainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    
    /// Logging rate
    public var rate: Int
    private var count: Int = 0
    
    
    /// Creates a logging layer that logs the provided input every `rate` iterations
    ///
    /// - Parameter rate: Logging rate
    public init(rate: Int = 1) {
        self.rate = rate
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        count += 1
        if count % rate == 0 {
            print(inputs[0])
            count = 0
        }
        return inputs[0]
    }
}


/// Wraps an arbitrary transform into a layer.
public class Lambda<Element: NumericType, Input: NumericType, Device: DeviceType>: Layer {
    public var isTrainable: Bool = true
    
    public var parameters: [Tensor<Element, Device>]
    
    
    /// Transformation that is applied to every input of the layer.
    public var transform: (Tensor<Input, Device>) -> Tensor<Element, Device>
    
    
    /// Initializes a Lambda layer that performs an arbitrary given transformation to its first input tensor
    ///
    /// - Parameters:
    ///   - parameters: Parameters, that should be optimized during the training phase
    ///   - transform: Transformation that is applied to the first input tensor of the layer for each forward operation.
    public init(parameters: [Tensor<Element, Device>] = [], _ transform: @escaping (Tensor<Input, Device>) -> Tensor<Element, Device>) {
        self.transform = transform
        self.parameters = parameters
    }
    
    public func forward(_ inputs: [Tensor<Input, Device>]) -> Tensor<Element, Device> {
        return transform(inputs[0])
    }
}


public class Debug<Element: NumericType, Device: DeviceType>: Layer {
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var isTrainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public var checkNaN: Bool
    public var checkInfinity: Bool
    
    public init(checkNaN: Bool = true, checkInfinity: Bool = true) {
        self.checkNaN = checkNaN
        self.checkInfinity = checkInfinity
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        if checkNaN {
            let sum = DL4S.sum(inputs[0]).item
            if sum.isNaN {
                fatalError("Found NaN")
            }
        }
        if checkInfinity {
            let max = DL4S.max(inputs[0]).item
            let min = -DL4S.max(-inputs[0]).item
            
            if !max.isFinite || !min.isFinite {
                fatalError("Found Infinity")
            }
        }
        
        return inputs[0]
    }
}
