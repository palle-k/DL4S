//
//  Conv.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

import Foundation


fileprivate struct Conv2DOperation<Element: NumericType, DeviceType: Device>: TensorOperation {
    var sourceTensors: [Tensor<Element, DeviceType>] {
        return [image, kernel]
    }
    
    var image: Tensor<Element, DeviceType>
    var kernel: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        fatalError()
    }
    
    var symbol: String {
        return "conv"
    }
}


public func conv2d<Element, DeviceType>(input: Tensor<Element, DeviceType>, kernel: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    precondition(4 ~= input.dim)
    precondition(kernel.dim == 4)
    
    let result = Tensor<Element, DeviceType>(
        shape: input.shape,
        parent: nil,
        context: Conv2DOperation(image: input, kernel: kernel).asAny()
    )
    
    fatalError("TODO: Batch processing")
    DeviceType.EngineType.conv2d(
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
