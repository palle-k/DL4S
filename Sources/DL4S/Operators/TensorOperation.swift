//
//  TensorOperation.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
//

import Foundation


protocol TensorOperation {
    associatedtype Element: NumericType
    associatedtype DeviceType: Device
    
    var sourceTensors: [Tensor<Element, DeviceType>] { get }
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>)
    func zeroGradient()
    
    var symbol: String { get }
}

extension TensorOperation {
    func zeroGradient() {
        sourceTensors.forEach { v in
            v.zeroGradient()
        }
    }
    
    func asAny() -> AnyTensorOperation<Element, DeviceType> {
        return AnyTensorOperation(operation: self)
    }
    
}

struct AnyTensorOperation<Element: NumericType, DeviceType: Device>: TensorOperation {
    var sourceTensors: [Tensor<Element, DeviceType>] {
        return _getSource()
    }
    
    private let _getSource: () -> [Tensor<Element, DeviceType>]
    private let _fillGradients: (Tensor<Element, DeviceType>) -> ()
    private let _zeroGrad: () -> ()
    private let _getSymbol: () -> String
    
    init<Op: TensorOperation>(operation: Op) where Op.Element == Element, Op.DeviceType == DeviceType {
        self._getSource = {operation.sourceTensors}
        self._fillGradients = operation.fillSourceGradients
        self._zeroGrad = operation.zeroGradient
        self._getSymbol = {operation.symbol}
    }
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
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
    var source: Tensor<Element, DeviceType> { get }
}

extension UnaryTensorOperation {
    var sourceTensors: [Tensor<Element, DeviceType>] {
        return [source]
    }
}

protocol BinaryTensorOperation: TensorOperation {
    var lhs: Tensor<Element, DeviceType> { get }
    var rhs: Tensor<Element, DeviceType> { get }
}

extension BinaryTensorOperation {
    var sourceTensors: [Tensor<Element, DeviceType>] {
        return [lhs, rhs]
    }
}
