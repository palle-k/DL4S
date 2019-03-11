//
//  TensorOperation.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
//

import Foundation


protocol TensorOperation {
    associatedtype Element: NumericType
    associatedtype Device: DeviceType
    
    var sourceTensors: [Tensor<Element, Device>] { get }
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>)
    func zeroGradient()
    
    var symbol: String { get }
}

extension TensorOperation {
    func zeroGradient() {
        sourceTensors.forEach { v in
            v.zeroGradient()
        }
    }
    
    func asAny() -> AnyTensorOperation<Element, Device> {
        return AnyTensorOperation(operation: self)
    }
    
}

struct AnyTensorOperation<Element: NumericType, Device: DeviceType>: TensorOperation {
    var sourceTensors: [Tensor<Element, Device>] {
        return _getSource()
    }
    
    private let _getSource: () -> [Tensor<Element, Device>]
    private let _fillGradients: (Tensor<Element, Device>) -> ()
    private let _zeroGrad: () -> ()
    private let _getSymbol: () -> String
    
    init<Op: TensorOperation>(operation: Op) where Op.Element == Element, Op.Device == Device {
        self._getSource = {operation.sourceTensors}
        self._fillGradients = operation.fillSourceGradients
        self._zeroGrad = operation.zeroGradient
        self._getSymbol = {operation.symbol}
    }
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        _fillGradients(vector)
    }
    
    func zeroGradient() {
        _zeroGrad()
    }
    
    var symbol: String {
        return _getSymbol()
    }
}

protocol UnaryTensorOperation: TensorOperation {
    var source: Tensor<Element, Device> { get }
}

extension UnaryTensorOperation {
    var sourceTensors: [Tensor<Element, Device>] {
        return [source]
    }
}

protocol BinaryTensorOperation: TensorOperation {
    var lhs: Tensor<Element, Device> { get }
    var rhs: Tensor<Element, Device> { get }
}

extension BinaryTensorOperation {
    var sourceTensors: [Tensor<Element, Device>] {
        return [lhs, rhs]
    }
}
