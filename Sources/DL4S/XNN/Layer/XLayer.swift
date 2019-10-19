//
//  XLayer.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
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

@dynamicCallable
public protocol XLayer {
    associatedtype Inputs
    associatedtype Outputs
    associatedtype Parameter: NumericType
    associatedtype Device: DeviceType
    
    var parameterPaths: [WritableKeyPath<Self, XTensor<Parameter, Device>>] { get }
    var parameters: [XTensor<Parameter, Device>] { get }
    
    func callAsFunction(_ inputs: Inputs) -> Outputs
}

public extension XLayer {
    func dynamicallyCall(withArguments arguments: [Inputs]) -> Outputs {
        return callAsFunction(arguments[0])
    }
}

public struct XAnyLayer<Inputs, Outputs, Parameter: NumericType, Device: DeviceType>: XLayer {
    public var parameterPaths: [WritableKeyPath<Self, XTensor<Parameter, Device>>] {
        parameters.indices.map {
            \Self.parameters[$0]
        }
    }
    
    public var parameters: [XTensor<Parameter, Device>] {
        get { getParameters() }
        set {
            
        }
    }
    
    private var getParameters: () -> [XTensor<Parameter, Device>]
    private var setParameters: ([XTensor<Parameter, Device>]) -> ()
    private var forward: (Inputs) -> Outputs
    
    public init<L: XLayer>(_ wrappedLayer: L) where L.Inputs == Inputs, L.Outputs == Outputs, L.Parameter == Parameter, L.Device == Device {
        var wrappedLayer = wrappedLayer
        let paths = wrappedLayer.parameterPaths
        
        getParameters = {
            wrappedLayer.parameters
        }
        setParameters = {
            zip(paths, $0).forEach {
                wrappedLayer[keyPath: $0] = $1
            }
        }
        forward = {
            wrappedLayer.callAsFunction($0)
        }
    }
    
    public func callAsFunction(_ inputs: Inputs) -> Outputs {
        forward(inputs)
    }
}