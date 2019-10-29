//
//  Scatter.swift
//  DL4S
//
//  Created by Palle Klewitz on 29.10.19.
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

public extension Tensor {
    func gather(using context: Tensor<Int32, Device>, alongAxis axis: Int) -> Self {
        var resultShape = shape
        resultShape.remove(at: axis)
        let originalAxisSize = shape[axis]
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        
        Device.Engine.gather(expanded: self.values, context: context.values, result: resultBuffer, axis: axis)
        return Tensor(
            using: resultBuffer,
            context: requiresGradient ? TensorContext(
                tag: "gather",
                sources: [self],
                backpropagate: [{ resultGradient -> Self in
                    resultGradient.scatter(using: context, alongAxis: axis, withSize: originalAxisSize)
                }]
            ) : nil
        )
    }
    
    func scatter(using context: Tensor<Int32, Device>, alongAxis axis: Int, withSize axisSize: Int) -> Self {
        var resultShape = shape
        resultShape.insert(axisSize, at: axis)
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.scatter(reduced: values, context: context.values, result: resultBuffer, axis: axis)
        return Tensor(
            using: resultBuffer,
            context: requiresGradient ? TensorContext(
                tag: "scatter",
                sources: [self],
                backpropagate: [{ resultGradient -> Self in
                    resultGradient.gather(using: context, alongAxis: axis)
                }]
            ) : nil
        )
    }
}
