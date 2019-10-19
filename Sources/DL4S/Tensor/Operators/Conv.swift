//
//  Conv.swift
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

//MARK: Img2col
public extension Tensor {
    func img2col(kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) -> Tensor<Element, Device> {
        let resultHeight = (shape[2] + 2 * padding - kernelHeight) / stride + 1
        let resultWidth = (shape[3] + 2 * padding - kernelWidth) / stride + 1
        
        let resultShape = [
            shape[1] * kernelWidth * kernelHeight,
            resultHeight * resultWidth * shape[0]
        ]
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.img2col(
            values: values,
            result: resultBuffer,
            kernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            padding: padding,
            stride: stride
        )
        
        return Tensor(
            using: resultBuffer,
            context: requiresGradient ? TensorContext(
                tag: "im2col",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.col2img(kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride, resultShape: self.shape)
                }]
            ) : nil
        )
    }
    
    func col2img(kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int, resultShape: [Int]) -> Tensor<Element, Device> {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.col2img(
            matrix: self.values,
            image: resultBuffer,
            kernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            padding: padding,
            stride: stride
        )
        
        return Tensor(
            using: resultBuffer,
            context: TensorContext(
                tag: "col2im",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.img2col(kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride)
                }]
            )
        )
    }
}

//MARK: Convolution
public extension Tensor {
    func convolved2d(filters: Tensor<Element, Device>, padding: Int? = nil, stride: Int = 1) -> Tensor<Element, Device> {
        let padding = padding ?? ((filters.shape[2] - 1) / 2)
        
        let outputShape = [
            shape[0],
            filters.shape[0],
            (shape[2] + 2 * padding - filters.shape[2]) / stride + 1,
            (shape[3] + 2 * padding - filters.shape[3]) / stride + 1
        ]
        
        let cols = self.img2col(kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: padding, stride: stride)
        let conv = filters
            .view(as: [filters.shape[0], filters.shape[1] * filters.shape[2] * filters.shape[3]])
            .matrixMultiplied(with: cols)
            .view(as: [outputShape[1], outputShape[0], outputShape[2], outputShape[3]])
        
        return conv.permuted(to: [1, 0, 2, 3])
    }
}

//MARK: Pooling
public extension Tensor {
    func maxPooled2d(windowSize: Int, padding: Int? = nil, stride: Int? = nil) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "MaxPool2D") {
            let padding = padding ?? ((windowSize - 1) / 2)
            let stride = stride ?? windowSize
            
            let outputShape = [
                shape[0],
                shape[1],
                (shape[2] + 2 * padding - windowSize) / stride + 1,
                (shape[3] + 2 * padding - windowSize) / stride + 1
            ]
            
            let cols = self
                .view(as: [shape[0] * shape[1], 1, shape[2], shape[3]])
                .img2col(
                    kernelWidth: windowSize,
                    kernelHeight: windowSize,
                    padding: padding,
                    stride: stride
                )
            let pooled = cols.reduceMax(along: [0])
            return pooled.view(as: outputShape)
        }
    }

    func averagePooled2d(windowSize: Int, padding: Int? = nil, stride: Int? = nil) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "AveragePool2D") {
            let padding = padding ?? ((windowSize - 1) / 2)
            let stride = stride ?? windowSize
            
            let outputShape = [
                shape[0],
                shape[1],
                (shape[2] + 2 * padding - windowSize) / stride + 1,
                (shape[3] + 2 * padding - windowSize) / stride + 1
            ]
            
            let cols = self
                .view(as: [shape[0] * shape[1], 1, shape[2], shape[3]])
                .img2col(
                    kernelWidth: windowSize,
                    kernelHeight: windowSize,
                    padding: padding,
                    stride: stride
                )
            let pooled = cols.reduceMean(along: [0])
            return pooled.view(as: outputShape)
        }
    }
}
