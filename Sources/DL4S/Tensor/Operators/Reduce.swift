//
//  Reduce.swift
//  DL4S
//
//  Created by Palle Klewitz on 03.10.19.
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


//MARK: Summation
public extension Tensor {
    /// Sums up elements along the given axes.
    ///
    /// - Parameter axes: Axes to sum
    /// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
    func reduceSum(along axes: [Int]) -> Tensor<Element, Device> {
        if axes.isEmpty {
            return self
        }
        
        var resultShape = shape
        for a in axes.reversed() {
            resultShape.remove(at: a)
        }
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.reduceSum(values: self.values, result: resultBuffer, axes: axes)
        
        if requiresGradient {
            return Tensor(
                using: resultBuffer,
                context: TensorContext(
                    tag: "sum\(axes)",
                    sources: [self],
                    backpropagateAccumulate: [{ resultGradient, acc in
                        var broadcastShape = self.shape
                        
                        for a in axes {
                            broadcastShape[a] = 1
                        }
                        
                        return (acc ?? Tensor(repeating: 0, shape: self.shape)) + resultGradient.view(as: broadcastShape)
                    }]
                )
            )
        } else {
            return Tensor(using: resultBuffer, context: nil)
        }
    }
    
    /// Sums up elements along the given axes
    /// - Parameter axes: Axes to sum
    /// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
    @inline(__always)
    func reduceSum(along axes: Int...) -> Self {
        reduceSum(along: axes)
    }
    
    /// Computes the sum of all elements of the tensor
    /// - Returns: Scalar, sum of all elements
    func reduceSum() -> Self {
        reduceSum(along: Array(0 ..< dim))
    }
    
    /// Computes the mean of the elements along the given axes
    /// - Parameter axes: Axes to compute the mean of
    /// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
    func reduceMean(along axes: [Int]) -> Self {
        reduceSum(along: axes) / Tensor(integerLiteral: axes.map {shape[$0]}.reduce(1, *))
    }
    
    /// Computes the mean of the elements along the given axes
    /// - Parameter axes: Tensor with shape equal to self.shape without the given reduction axes.
    func reduceMean(along axes: Int...) -> Self {
        reduceMean(along: axes)
    }
    
    /// Computes the mean of all elements of the tensor
    /// - Returns: Scalar, mean of all elements
    func reduceMean() -> Self {
        reduceMean(along: Array(0 ..< dim))
    }
    
    /// Computes the variance of the tensor along the given axes.
    ///
    /// - Parameter axes: Axes to compute the variance along.
    /// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
    func variance(along axes: [Int]) -> Self {
        let m = self.reduceMean(along: axes)
        return (self * self).reduceMean(along: axes) - m * m
    }
    
    /// Computes the variance of the tensor along the given axes.
    ///
    /// - Parameter axes: Axes to compute the variance along.
    /// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
    func variance(along axes: Int...) -> Self {
        variance(along: axes)
    }
    
    /// Computes the variance of all elements in the tensor.
    ///
    /// - Returns: Scalar, variance of all elements
    func variance() -> Self {
        variance(along: Array(0 ..< dim))
    }
    
    /// Returns the index of the largest element in the tensor.
    func argmax() -> Int {
        Device.Engine.argmax(values: values.values, count: count).0
    }
}

/// Computes the sum of all elements in the given tensor
/// - Parameter tensor: Tensor to sum
/// - Returns: Scalar, sum of all elements
public func sum<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.reduceSum()
}

/// Computes the sum along the given axes on the given tensor.
/// - Parameters:
///   - tensor: Tensor to reduce
///   - axes: Axes to reduce along
/// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
public func sum<Element, Device>(_ tensor: Tensor<Element, Device>, axes: [Int]) -> Tensor<Element, Device> {
    tensor.reduceSum(along: axes)
}

/// Computes the sum along the given axes on the given tensor.
/// - Parameters:
///   - tensor: Tensor to reduce
///   - axes: Axes to reduce along
/// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
public func sum<Element, Device>(_ tensor: Tensor<Element, Device>, axes: Int...) -> Tensor<Element, Device> {
    tensor.reduceSum(along: axes)
}

/// Computes the mean of all elements in the given tensor
/// - Parameter tensor: Tensor to average
/// - Returns: Scalar, mean of all elements
/// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
public func mean<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.reduceMean()
}

/// Computes the mean along the given axes on the given tensor.
/// - Parameters:
///   - tensor: Tensor to reduce
///   - axes: Axes to reduce along
/// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
public func mean<Element, Device>(_ tensor: Tensor<Element, Device>, axes: [Int]) -> Tensor<Element, Device> {
    tensor.reduceMean(along: axes)
}

/// Computes the mean along the given axes on the given tensor.
/// - Parameters:
///   - tensor: Tensor to reduce
///   - axes: Axes to reduce along
/// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
public func mean<Element, Device>(_ tensor: Tensor<Element, Device>, axes: Int...) -> Tensor<Element, Device> {
    tensor.reduceMean(along: axes)
}

/// Computes the variance of all elements in the given tensor
/// - Parameter tensor: Tensor to average
/// - Returns: Scalar, variance of all elements
/// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
public func variance<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.variance()
}

/// Computes the variance along the given axes on the given tensor.
/// - Parameters:
///   - tensor: Tensor to reduce
///   - axes: Axes to reduce along
/// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
public func variance<Element, Device>(_ tensor: Tensor<Element, Device>, axes: [Int]) -> Tensor<Element, Device> {
    tensor.variance(along: axes)
}

/// Computes the variance along the given axes on the given tensor.
/// - Parameters:
///   - tensor: Tensor to reduce
///   - axes: Axes to reduce along
/// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
public func variance<Element, Device>(_ tensor: Tensor<Element, Device>, axes: Int...) -> Tensor<Element, Device> {
    tensor.variance(along: axes)
}

//MARK: Min/Max
public extension Tensor {
    
    /// Computes the maximum values along the given axes of the tensor.
    /// - Parameter axes: Axes to reduce along
    /// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
    func reduceMax(along axes: [Int]) -> Self {
        var resultShape: [Int] = shape
        for a in axes.reversed() {
            resultShape.remove(at: a)
        }
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        if requiresGradient {
            precondition(axes.count == 1, "Scattering (reduceMax backpropagation) is only available along a single axis.")
            
            let contextBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Int32.self)
            if let axis = axes.first, axes.count == 1 {
                Device.Engine.reduceMax(values: values, result: resultBuffer, context: contextBuffer, axis: axis)
            } else {
                Device.Engine.reduceMax(values: values, result: resultBuffer, context: contextBuffer, axes: axes)
            }
            
            let context = Tensor<Int32, Device>(using: contextBuffer, context: nil)
            
            let axisShape = self.shape[axes[0]]
            
            return Tensor(
                using: resultBuffer,
                context: TensorContext(
                    tag: "max\(axes)",
                    sources: [self],
                    backpropagate: [{ resultGradient in
                        return resultGradient.scatter(using: context, alongAxis: axes[0], withSize: axisShape)
                    }]
                )
            )
        } else {
            if let axis = axes.first, axes.count == 1 {
                Device.Engine.reduceMax(values: values, result: resultBuffer, context: nil, axis: axis)
            } else {
                Device.Engine.reduceMax(values: values, result: resultBuffer, context: nil, axes: axes)
            }
            
            return Tensor(using: resultBuffer, context: nil)
        }
    }
    
    
    /// Computes the maximum values along the given axes of the tensor.
    /// - Parameter axes: Axes to reduce along
    /// - Returns: Tensor with shape equal to self.shape without the given reduction axes.
    func reduceMax(along axes: Int...) -> Self {
        reduceMax(along: axes)
    }
    
    
    /// Computes the maximum of all values in the tensor
    /// - Returns: Scalar, maximum of all elements.
    func reduceMax() -> Self {
        reduceMax(along: Array(0 ..< dim))
    }
}
