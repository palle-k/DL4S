//
//  TensorOperation.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
//

import Foundation


protocol TensorOperation {
    associatedtype Element: NumericType
    
    var sourceTensors: [Tensor<Element>] { get }
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>)
    func zeroGradient()
    
    var symbol: String { get }
}

extension TensorOperation {
    func zeroGradient() {
        sourceTensors.forEach { v in
            v.zeroGradient()
        }
    }
    
    func asAny() -> AnyTensorOperation<Element> {
        return AnyTensorOperation(operation: self)
    }
    
}

struct AnyTensorOperation<Element: NumericType>: TensorOperation {
    var sourceTensors: [Tensor<Element>] {
        return _getSource()
    }
    
    private let _getSource: () -> [Tensor<Element>]
    private let _fillGradients: (Tensor<Element>) -> ()
    private let _zeroGrad: () -> ()
    private let _getSymbol: () -> String
    
    init<Op: TensorOperation>(operation: Op) where Op.Element == Element {
        self._getSource = {operation.sourceTensors}
        self._fillGradients = operation.fillSourceGradients
        self._zeroGrad = operation.zeroGradient
        self._getSymbol = {operation.symbol}
    }
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
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
    var source: Tensor<Element> { get }
}

extension UnaryTensorOperation {
    var sourceTensors: [Tensor<Element>] {
        return [source]
    }
}

protocol BinaryTensorOperation: TensorOperation {
    var lhs: Tensor<Element> { get }
    var rhs: Tensor<Element> { get }
}

extension BinaryTensorOperation {
    var sourceTensors: [Tensor<Element>] {
        return [lhs, rhs]
    }
}
