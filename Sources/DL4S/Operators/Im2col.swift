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


private struct Im2ColOperation<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    let kernelWidth: Int
    let kernelHeight: Int
    let padding: Int
    let stride: Int
    
    var symbol: String {
        return "im2col"
    }
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let sourceGradient = source.shapedGradient, let vectorGradient = vector.shapedGradient else {
            return
        }
        let tmp = Device.Memory.allocateBuffer(withShape: sourceGradient.shape, type: Element.self)
 
        Device.Engine.col2img(matrix: vectorGradient, image: tmp, kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride)
        Device.Engine.vAdd(lhs: tmp.values, rhs: sourceGradient.values, result: sourceGradient.values, count: sourceGradient.count)
        
        Device.Memory.free(tmp)
    }
}

private struct Col2ImOperatiion<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    let kernelWidth: Int
    let kernelHeight: Int
    let padding: Int
    let stride: Int
    
    var symbol: String {
        return "col2im"
    }
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let sourceGradient = source.shapedGradient, let vectorGradient = vector.shapedGradient else {
            return
        }
        let tmp = Device.Memory.allocateBuffer(withShape: sourceGradient.shape, type: Element.self)
        
        Device.Engine.img2col(values: vectorGradient, result: tmp, kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride)
        Device.Engine.vAdd(lhs: tmp.values, rhs: sourceGradient.values, result: sourceGradient.values, count: sourceGradient.count)
        
        Device.Memory.free(tmp)
    }
}

public func img2col<Element, Device>(images: Tensor<Element, Device>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) -> Tensor<Element, Device> {
    let resultHeight = (images.shape[2] + 2 * padding - kernelHeight) / stride + 1
    let resultWidth = (images.shape[3] + 2 * padding - kernelWidth) / stride + 1
    
    let resultShape = [
        images.shape[1] * kernelWidth * kernelHeight,
        resultHeight * resultWidth * images.shape[0]
    ]
    let result = Tensor<Element, Device>(
        shape: resultShape,
        parent: nil,
        context: Im2ColOperation(
            source: images,
            kernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            padding: padding,
            stride: stride
        ).asAny()
    )
    
    Device.Engine.img2col(
        values: images.shapedValues,
        result: result.shapedValues,
        kernelWidth: kernelWidth,
        kernelHeight: kernelHeight,
        padding: padding,
        stride: stride
    )
    
    return result
}

public func conv2d<Element, Device>(images: Tensor<Element, Device>, filters: Tensor<Element, Device>, padding: Int? = nil, stride: Int = 1) -> Tensor<Element, Device> {
    let padding = padding ?? ((filters.shape[2] - 1) / 2)
    
    let outputShape = [
        images.shape[0],
        filters.shape[0],
        (images.shape[2] + 2 * padding - filters.shape[2]) / stride + 1,
        (images.shape[3] + 2 * padding - filters.shape[3]) / stride + 1
    ]
    
    let cols = img2col(images: images, kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: padding, stride: stride)
    let conv = mmul(
        filters.view(as: [filters.shape[0], filters.shape[1] * filters.shape[2] * filters.shape[3]]),
        cols
    ).view(as: [outputShape[1], outputShape[0], outputShape[2], outputShape[3]])
    
    return conv.permuted(to: [1, 0, 2, 3])
}


public func maxPool2d<Element, Device>(images: Tensor<Element, Device>, windowSize: Int, padding: Int? = nil, stride: Int? = nil) -> Tensor<Element, Device> {
    let padding = padding ?? ((windowSize - 1) / 2)
    let stride = stride ?? windowSize
    
    let outputShape = [
        images.shape[0],
        images.shape[1],
        (images.shape[2] + 2 * padding - windowSize) / stride + 1,
        (images.shape[3] + 2 * padding - windowSize) / stride + 1
    ]
    
    let cols = img2col(
        images: images.view(as: images.shape[0] * images.shape[1], 1, images.shape[2], images.shape[3]),
        kernelWidth: windowSize,
        kernelHeight: windowSize,
        padding: padding,
        stride: stride
    )
    let pooled = max(cols, axis: 0)
    return pooled.view(as: outputShape)
}

public func avgPool2d<Element, Device>(images: Tensor<Element, Device>, windowSize: Int, padding: Int? = nil, stride: Int? = nil) -> Tensor<Element, Device> {
    let padding = padding ?? ((windowSize - 1) / 2)
    let stride = stride ?? windowSize
    
    let outputShape = [
        images.shape[0],
        images.shape[1],
        (images.shape[2] + 2 * padding - windowSize) / stride + 1,
        (images.shape[3] + 2 * padding - windowSize) / stride + 1
    ]
    
    let cols = img2col(
        images: images.view(as: images.shape[0] * images.shape[1], 1, images.shape[2], images.shape[3]),
        kernelWidth: windowSize,
        kernelHeight: windowSize,
        padding: padding,
        stride: stride
    )
    let pooled = mean(cols, axes: [0])
    return pooled.view(as: outputShape)
}


public extension Tensor {
    @inline(__always)
    func convolved2d(filters: Tensor<Element, Device>, padding: Int? = nil, stride: Int = 1) -> Tensor<Element, Device> {
        return DL4S.conv2d(images: self, filters: filters, padding: padding, stride: stride)
    }
    
    @inline(__always)
    func maxPooled2d(windowSize: Int, padding: Int? = nil, stride: Int? = nil) -> Tensor<Element, Device> {
        return DL4S.maxPool2d(images: self, windowSize: windowSize, padding: padding, stride: stride)
    }
    
    @inline(__always)
    func avgPooled2d(windowSize: Int, padding: Int? = nil, stride: Int? = nil) -> Tensor<Element, Device> {
        return DL4S.avgPool2d(images: self, windowSize: windowSize, padding: padding, stride: stride)
    }
}
