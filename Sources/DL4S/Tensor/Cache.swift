//
//  Cache.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.12.19.
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


public enum TensorCache {
    static let one = Tensor<Float, GPU>(1)
    static let zero = Tensor<Float, GPU>(0)
    static let half = Tensor<Float, GPU>(0.5)
    static let two = Tensor<Float, GPU>(2)
    static let minusOne = Tensor<Float, GPU>(-1)
    
    private static var cache: [Float: Tensor<Float, GPU>] = [:]
    
    public static func value<Element: NumericType, Device: DeviceType>(_ value: Element) -> Tensor<Element, Device> {
        if Element.self == Float.self && Device.self == GPU.self {
            if let t = cache[value as! Float] {
                return t as! Tensor<Element, Device>
            }
            let t = Tensor<Element, Device>(value)
            cache[value as! Float] = (t as! Tensor<Float, GPU>)
            return t
        } else {
            return Tensor(value)
        }
    }
}
