//
//  Layers.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.02.19.
//

import Foundation


public protocol Layer {
    associatedtype Input: NumericType
    associatedtype Element: NumericType
    associatedtype Device: DeviceType
    
    var parameters: [Tensor<Element, Device>] { get }
    var trainable: Bool { get nonmutating set }
    
    func forward(_ inputs: [Tensor<Input, Device>]) -> Tensor<Element, Device>
}

public extension Layer {
    func forward(_ inputs: Tensor<Input, Device>...) -> Tensor<Element, Device> {
        return forward(inputs)
    }
    
    func asAny() -> AnyLayer<Input, Element, Device> {
        return AnyLayer(self)
    }
}

public struct AnyLayer<Input: NumericType, Element: NumericType, Device: DeviceType>: Layer {
    private let getParams: () -> [Tensor<Element, Device>]
    private let performForward: ([Tensor<Input, Device>]) -> (Tensor<Element, Device>)
    private let getTrainable: () -> Bool
    private let setTrainable: (Bool) -> ()
    
    public var parameters: [Tensor<Element, Device>] {
        return getParams()
    }
    
    public var trainable: Bool {
        get {
            return getTrainable()
        }
        nonmutating set {
            setTrainable(newValue)
        }
    }
    
    public init<L: Layer>(_ layer: L) where L.Element == Element, L.Input == Input, L.Device == Device {
        self.getParams = {layer.parameters}
        self.performForward = {layer.forward($0)}
        self.getTrainable = {layer.trainable}
        self.setTrainable = {layer.trainable = $0}
    }
    
    public func forward(_ inputs: [Tensor<Input, Device>]) -> Tensor<Element, Device> {
        return performForward(inputs)
    }
}


public class Sequential<Element: NumericType, Device: DeviceType>: Layer {
    public typealias Input = Element
    
    public let layers: [AnyLayer<Element, Element, Device>]
    
    public var trainable: Bool {
        get {
            return layers.contains(where: {$0.trainable})
        }
        set {
            layers.forEach {$0.trainable = newValue}
        }
    }
    
    public var parameters: [Tensor<Element, Device>] {
        return layers.flatMap {$0.parameters}
    }
    
    public init(_ layers: AnyLayer<Element, Element, Device>...) {
        self.layers = layers
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        return layers.reduce(inputs[0]) {$1.forward([$0])}
    }
    
    public func saveWeights(to url: URL) throws {
        let params = self.parameters
        let encoder = JSONEncoder()
        let encoded = try encoder.encode(params)
        try encoded.write(to: url, options: .atomic)
    }
    
    public func loadWeights(from url: URL) throws {
        let params = self.parameters
        let decoder = JSONDecoder()
        let encoded = try Data(contentsOf: url)
        let decodedParams = try decoder.decode([Tensor<Element, Device>].self, from: encoded)
        
        for (dst, src) in zip(params, decodedParams) {
            Device.Memory.assign(from: src.values, to: dst.values, count: dst.count)
            
            if let dstGrad = dst.gradient, let srcGrad = src.gradient {
                Device.Memory.assign(from: srcGrad, to: dstGrad, count: dst.count)
            }
        }
    }
}

public class Logging<Element: NumericType, Device: DeviceType>: Layer {
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var trainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public var rate: Int
    private var count: Int = 0
    
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

public class Dropout<Element: NumericType, Device: DeviceType>: Layer {
    public typealias Input = Element
    
    public var trainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var isTrainingMode: Bool = true
    
    public var dropoutRate: Float
    
    public init(rate: Float) {
        self.dropoutRate = rate
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        if isTrainingMode {
            let x = inputs[0]
            let mask: Tensor<Element, Device> = Random.bernoulli(p: (1 - dropoutRate), shape: Array(x.shape.dropFirst()))
            mask.tag = "DropoutMask"
            return x * mask
        } else {
            return inputs[0]
        }
    }
}

public class Lambda<Element: NumericType, Input: NumericType, Device: DeviceType>: Layer {
    public var trainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var transform: (Tensor<Input, Device>) -> Tensor<Element, Device>
    
    public init(_ transform: @escaping (Tensor<Input, Device>) -> Tensor<Element, Device>) {
        self.transform = transform
    }
    
    public func forward(_ inputs: [Tensor<Input, Device>]) -> Tensor<Element, Device> {
        return transform(inputs[0])
    }
}
