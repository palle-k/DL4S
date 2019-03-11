//
//  Conv.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

import Foundation


fileprivate struct Conv2DOperation<Element: NumericType>: TensorOperation {
    var sourceTensors: [Tensor<Element>] {
        return [image, kernel]
    }
    
    var image: Tensor<Element>
    var kernel: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        fatalError()
    }
    
    var symbol: String {
        return "conv"
    }
}


public func conv2d<Element>(input: Tensor<Element>, kernel: Tensor<Element>) -> Tensor<Element> {
    precondition(4 ~= input.dim)
    precondition(kernel.dim == 4)
    
    let result = Tensor<Element>(
        shape: input.shape,
        parent: nil,
        context: Conv2DOperation(image: input, kernel: kernel).asAny()
    )
    
    fatalError("TODO: Batch processing")
    Element.conv2d(
        input: input.values.immutable,
        filter: kernel.values.immutable,
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
