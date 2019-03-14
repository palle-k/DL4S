//
//  TensorOperation.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
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
