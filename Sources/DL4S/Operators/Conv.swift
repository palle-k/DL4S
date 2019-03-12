//
//  Conv.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
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


fileprivate struct Conv2DOperation<Element: NumericType, Device: DeviceType>: TensorOperation {
    var sourceTensors: [Tensor<Element, Device>] {
        return [image, kernel]
    }
    
    var image: Tensor<Element, Device>
    var kernel: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        fatalError()
    }
    
    var symbol: String {
        return "conv"
    }
}


public func conv2d<Element, Device>(input: Tensor<Element, Device>, kernel: Tensor<Element, Device>) -> Tensor<Element, Device> {
    precondition(4 ~= input.dim)
    precondition(kernel.dim == 4)
    
    let result = Tensor<Element, Device>(
        shape: input.shape,
        parent: nil,
        context: Conv2DOperation(image: input, kernel: kernel).asAny()
    )
    
    fatalError("TODO: Batch processing")
    Device.Engine.conv2d(
        input: input.values,
        filter: kernel.values,
        result: result.values,
        width: input.shape[3],
        height: input.shape[2],
        kernelWidth: kernel.shape[3],
        kernelHeight: kernel.shape[2],
        kernelDepth: kernel.shape[1],
        kernelCount: kernel.shape[0]
    )
    
    return result
}
