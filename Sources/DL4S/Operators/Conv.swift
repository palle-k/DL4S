//
//  Conv.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

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
