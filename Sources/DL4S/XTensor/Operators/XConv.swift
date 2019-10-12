//
//  XConv.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
//

import Foundation

//MARK: Img2col
public extension XTensor {
    func img2col(kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) -> XTensor<Element, Device> {
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
        
        return XTensor(
            using: resultBuffer,
            context: requiresGradient ? XTensorContext(
                tag: "Img2col",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.col2img(kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride, resultShape: self.shape)
                }]
            ) : nil
        )
    }
    
    func col2img(kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int, resultShape: [Int]) -> XTensor<Element, Device> {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.col2img(
            matrix: self.values,
            image: resultBuffer,
            kernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            padding: padding,
            stride: stride
        )
        
        return XTensor(
            using: resultBuffer,
            context: XTensorContext(
                tag: "Col2img",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.img2col(kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride)
                }]
            )
        )
    }
}

//MARK: Convolution
public extension XTensor {
    func convolved2d(filters: XTensor<Element, Device>, padding: Int? = nil, stride: Int = 1) -> XTensor<Element, Device> {
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
            .matMul(cols)
            .view(as: [outputShape[1], outputShape[0], outputShape[2], outputShape[3]])
        
        return conv.permuted(to: [1, 0, 2, 3])
    }
}

//MARK: Pooling
public extension XTensor {
    func maxPooled2d(windowSize: Int, padding: Int? = nil, stride: Int? = nil) -> XTensor<Element, Device> {
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

    func averagePooled2d(images: XTensor<Element, Device>, windowSize: Int, padding: Int? = nil, stride: Int? = nil) -> XTensor<Element, Device> {
        let padding = padding ?? ((windowSize - 1) / 2)
        let stride = stride ?? windowSize
        
        let outputShape = [
            images.shape[0],
            images.shape[1],
            (images.shape[2] + 2 * padding - windowSize) / stride + 1,
            (images.shape[3] + 2 * padding - windowSize) / stride + 1
        ]
        
        let cols = self
            .view(as: [images.shape[0] * images.shape[1], 1, images.shape[2], images.shape[3]])
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
