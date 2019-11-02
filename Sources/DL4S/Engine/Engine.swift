//
//  Engine.swift
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

//MARK: Device
/// A device, on which tensor operations can be executed.
public protocol DeviceType {
    /// Memory manager for a device
    associatedtype Memory: MemoryOperatorsType where Memory.Device == Self
    /// Tensor operation engine for a device
    associatedtype Engine: EngineType where Engine.Device == Self
}

//MARK: Memory
/// Memory manager for a device
public protocol MemoryOperatorsType {
    associatedtype Device: DeviceType where Device.Memory == Self
    associatedtype RawBuffer: Hashable
    
    
    /// Allocates a buffer with capacity for the given amount of elements of the given type.
    ///
    /// - Parameters:
    ///   - withCapacity: Capacity to reserve
    ///   - type: Type of elements in the buffer
    static func allocateBuffer<Element>(withCapacity: Int, type: Element.Type) -> Buffer<Element, Device>
    
    /// Allocates a buffer with capacity for the amount of elements in the given shape
    /// - Parameters:
    ///   - shape: Shape of the buffer to allocate
    ///   - type: Type of elements in the buffer
    static func allocateBuffer<Element>(withShape shape: [Int], type: Element.Type) -> ShapedBuffer<Element, Device>
    
    /// Releases all resources associated with the given buffer
    /// - Parameter buffer: Buffer to release
    static func free<Element>(_ buffer: Buffer<Element, Device>)
    
    /// Releases all resources associated with the given buffer
    /// - Parameter buffer: Buffer to release
    static func free<Element>(_ buffer: ShapedBuffer<Element, Device>)
    
    /// Copies values from the given host buffer to the memory of the device
    /// - Parameters:
    ///   - source: Source buffer
    ///   - destination: Device buffer
    ///   - count: Number of elements to copy
    static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, Device>, count: Int)
    
    /// Copies values between two device buffers
    /// - Parameters:
    ///   - source: Source device buffer
    ///   - destination: Target device buffer
    ///   - count: Number of elements to copy
    static func assign<Element>(from source: Buffer<Element, Device>, to destination: Buffer<Element, Device>, count: Int)
    
    /// Copies values from the given device buffer to the given host buffer
    /// - Parameters:
    ///   - source: Device buffer
    ///   - destination: Host buffer
    ///   - count: Number of elements to copy
    static func assign<Element>(from source: Buffer<Element, Device>, to destination: UnsafeMutableBufferPointer<Element>, count: Int)
    
    /// Returns the first value in the buffer
    /// - Parameter source: Buffer to return the first value of
    static func getValue<Element>(from source: Buffer<Element, Device>) -> Element
    
    /// Returns the number of elements in the buffer
    /// - Parameter buffer: Number of elements in the buffer
    static func getSize<Element>(of buffer: Buffer<Element, Device>) -> Int
    
    /// Retrieves a slice of values from the given buffer
    /// - Parameters:
    ///   - slice: Index
    ///   - buffer: Buffer to read from
    ///   - shape: Shape of the buffer
    /// - Returns: The result buffer, a boolean indicating whether the result buffer is a copy (true) or a pointer in the same memory region (false) and the shape of the result.
    static func get<Element>(slice: [Int?], of buffer: Buffer<Element, Device>, with shape: [Int]) -> (Buffer<Element, Device>, Bool, [Int])
    
    /// Retrieves a slice of values from the given buffer
    /// - Parameters:
    ///   - slice: Index
    ///   - buffer: Buffer to read from
    ///   - shape: Shape of the buffer
    /// - Returns: The result buffer, a boolean indicating whether the result buffer is a copy (true) or a pointer in the same memory region (false) and the shape of the result.
    static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, Device>, with shape: [Int]) -> (Buffer<Element, Device>, Bool, [Int])
    
    /// Writes a slice into a target buffer
    /// - Parameters:
    ///   - slice: Slice index
    ///   - buffer: Buffer to write to
    ///   - dstShape: Shape of the destination buffer
    ///   - source: Buffer to read from
    ///   - sourceShape: Shape of the source buffer
    static func set<Element>(slice: [Int?], of buffer: Buffer<Element, Device>, with dstShape: [Int], from source: Buffer<Element, Device>, with sourceShape: [Int])
    
    /// Writes a slice into a target buffer
    /// - Parameters:
    ///   - slice: Slice index
    ///   - buffer: Buffer to write to
    ///   - dstShape: Shape of the destination buffer
    ///   - source: Buffer to read from
    ///   - sourceShape: Shape of the source buffer
    static func set<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, Device>, with dstShape: [Int], from source: Buffer<Element, Device>, with sourceShape: [Int])
    
    /// Sets the first element of the device buffer
    /// - Parameters:
    ///   - buffer: Target buffer
    ///   - newValue: Value to set the first slot of the target to
    static func setPointee<Element>(of buffer: Buffer<Element, Device>, to newValue: Element)
    
    /// Returns a buffer that uses the same region of memory but is advanced by the given number of elements
    /// - Parameters:
    ///   - buffer: Parent buffer
    ///   - advancement: Number of elements to advance the start index by
    static func advance<Element>(buffer: Buffer<Element, Device>, by advancement: Int) -> Buffer<Element, Device>
}

//MARK: Engine
/// Tensor operation engine for a device
public protocol EngineType {
    associatedtype Device: DeviceType where Device.Engine == Self
    
    //MARK: Simple operations
    
    /// Fills a buffer with a given value
    /// - Parameters:
    ///   - value: Value to fill the buffer with
    ///   - result: Buffer to fill
    ///   - count: Number of elements to write
    static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, count: Int)
    
    /// Element-wise Vector-vector add
    /// - Parameters:
    ///   - lhs: First summand
    ///   - rhs: Second summand
    ///   - result: Result buffer
    ///   - count: Number of elements to add
    static func vAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    
    /// Element-wise Vector negate
    /// - Parameters:
    ///   - val: Vector to negate
    ///   - result: Result buffer
    ///   - count: Number of elements to negate
    static func vNeg<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    
    /// Element-wise Vector-vector subtract
    /// - Parameters:
    ///   - lhs: Left-hand side vector
    ///   - rhs: Right-hand side vector
    ///   - result: Result bufffer
    ///   - count: Number of elements to subtract
    static func vSub<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    
    /// Element-wise Vector-vector multiply
    /// - Parameters:
    ///   - lhs: First factor
    ///   - rhs: Second factor
    ///   - result: Result buffer
    ///   - count: Number of elements to multiply
    static func vMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    
    /// Element-wise Vector-vector divide
    /// - Parameters:
    ///   - lhs: Dividend
    ///   - rhs: Divisor
    ///   - result: Result buffer
    ///   - count: Number of elements to divide
    static func vDiv<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)

    //MARK: Matrix operations
    
    /// Matrix multiply add in-place
    /// - Parameters:
    ///   - lhs: Left-hand side matrix
    ///   - rhs: Right-hand side matrix
    ///   - result: Summand and result buffer
    ///   - alpha: Matrix multiplication scale
    ///   - beta: Add scale
    ///   - transposeFirst: Whether to transpose the first matrix
    ///   - transposeSecond: Whether to transpose the second matrix
    static func gemm<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool)
    
    /// Broadcast matrix multiply add in-place
    /// - Parameters:
    ///   - lhs: Left-hand side matrix
    ///   - rhs: Right-hand side matrix
    ///   - result: Summand and result buffer
    ///   - alpha: Matrix multiplication scale
    ///   - beta: Add scale
    ///   - transposeFirst: Whether to transpose the first matrix
    ///   - transposeSecond: Whether to transpose the second matrix
    static func broadcastGemm<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool)
    
    /// Band matrix extraction
    /// - Parameters:
    ///   - buffer: Source matrix
    ///   - result: Result buffer
    ///   - belowDiagonal: Number of elements below diagonal to keep, nil for all elements
    ///   - aboveDiagonal: Number of elements above diagonal to keep, nil for all elements
    static func band<N: NumericType>(buffer: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, belowDiagonal: Int?, aboveDiagonal: Int?)
    
    //MARK: Broadcasting
    
    /// Broadcast adds two buffers
    /// - Parameters:
    ///   - lhs: First summand
    ///   - rhs: Second summand
    ///   - result: Sum vector
    static func broadcastAdd<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Broadcast subtracts one vector from another
    /// - Parameters:
    ///   - lhs: Left-hand side vector
    ///   - rhs: Vector to subtact from lhs
    ///   - result: Result buffer
    static func broadcastSub<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Broadcast multiplies two buffers
    /// - Parameters:
    ///   - lhs: First factor
    ///   - rhs: Second factor
    ///   - result: Product vector
    static func broadcastMul<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Broadcast divides one vector by another
    /// - Parameters:
    ///   - lhs: Left-hand side vector
    ///   - rhs: Vector to divide lhs with
    ///   - result: Result buffer
    static func broadcastDiv<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    //MARK: Reduction
    
    /// Reduces one buffer into another along one axis by computing the sum.
    /// - Parameters:
    ///   - values: Buffer to reduce
    ///   - result: Result buffer
    ///   - axis: Axis to reduce along
    static func reduceSum<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, axis: Int)
    
    /// Reduces one buffer into another along one axis by computing the maximum.
    /// - Parameters:
    ///   - values: Buffer to reduce
    ///   - result: Result buffer
    ///   - context: Context vector that stores the argmax
    ///   - axis: Axis to reduce along
    static func reduceMax<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, axis: Int)
    
    /// Reduces one buffer into another along one axis by computing the minimum.
    /// - Parameters:
    ///   - values: Buffer to reduce
    ///   - result: Result buffer
    ///   - context: Context vector that stores the argmin
    ///   - axis: Axis to reduce along
    static func reduceMin<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, axis: Int)
    
    /// Reduces one buffer into another along one axis by computing the mean.
    /// - Parameters:
    ///   - values: Buffer to reduce
    ///   - result: Result buffer
    ///   - axis: Axis to reduce along
    static func reduceMean<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, axis: Int)
    
    /// Reduces one buffer into another along multiple axes by computing the sum.
    /// - Parameters:
    ///   - values: Buffer to reduce
    ///   - result: Result buffer
    ///   - axes: Axes to reduce along
    static func reduceSum<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, axes: [Int])
    
    /// Reduces one buffer into another along multiple axes by computing the maximum.
    /// - Parameters:
    ///   - values: Buffer to reduce
    ///   - result: Result buffer
    ///   - context: Context buffer that stores the argmax
    ///   - axes: Axes to reduce along
    static func reduceMax<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, axes: [Int])
    
    /// Reduces one buffer into another along multiple axes by computing the minimum.
    /// - Parameters:
    ///   - values: Buffer to reduce
    ///   - result: Result buffer
    ///   - context: Context buffer that stores the argmin
    ///   - axes: Axes to reduce along
    static func reduceMin<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, axes: [Int])
    
    /// Reduces one buffer into another along multiple axes by computing the mean.
    /// - Parameters:
    ///   - values: Buffer to reduce
    ///   - result: Result buffer
    ///   - axes: Axes to reduce along
    static func reduceMean<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, axes: [Int])
    
    /// Computes the sum of a buffer
    /// - Parameters:
    ///   - values: Buffer to sum up
    ///   - result: Result buffer
    static func sum<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Computes the mean of a buffer
    /// - Parameters:
    ///   - values: Buffer to compute the mean of
    ///   - result: Result buffer
    static func mean<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Stores the maximum value in result and returns the argmax
    /// - Parameters:
    ///   - values: Value buffer
    ///   - result: Result buffer
    @discardableResult static func max<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>) -> Int
    
    /// Stores the minimum value in result and returns the argmin
    /// - Parameters:
    ///   - values: Value buffer
    ///   - result: Result buffer
    @discardableResult static func min<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>) -> Int
    
    /// Computes the argmax of a buffer
    /// - Parameters:
    ///   - values: Buffer to compute the argmax of
    ///   - count: Number of elements in the buffer
    static func argmax<N: NumericType>(values: Buffer<N, Device>, count: Int) -> (Int, N)
    
    //MARK: Element-wise functions
    
    /// Element-wise exponentiate
    /// - Parameters:
    ///   - values: Buffer of values to exponentiate
    ///   - result: Result buffer
    static func exp<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise log
    /// - Parameters:
    ///   - values: Buffer of values to compute the log of
    ///   - result: Result buffer
    static func log<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise square root
    /// - Parameters:
    ///   - values: Buffer of values to compute the square root of
    ///   - result: Result buffer
    static func sqrt<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise recitified linear unit
    /// - Parameters:
    ///   - values: Buffer of values to compute the relu of
    ///   - result: Result buffer
    static func relu<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise heaviside step function
    /// - Parameters:
    ///   - values: Buffer of values to compute the heaviside step function of
    ///   - result: Result buffer
    static func heaviside<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise sine
    /// - Parameters:
    ///   - values: Buffer of values to compute the sine of
    ///   - result: Result buffer
    static func sin<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise cosine
    /// - Parameters:
    ///   - values: Buffer of values to compute the cosine of
    ///   - result: Result buffer
    static func cos<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise tangent
    /// - Parameters:
    ///   - values: Buffer of values to compute the tangent of
    ///   - result: Result buffer
    static func tan<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise hyperbolic sine
    /// - Parameters:
    ///   - values: Buffer of values to compute the hyperbolic sine of
    ///   - result: Result buffer
    static func sinh<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise hyperbolic cosine
    /// - Parameters:
    ///   - values: Buffer of values to compute the hyperbolic cosine of
    ///   - result: Result buffer
    static func cosh<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise hyperbolic tangent
    /// - Parameters:
    ///   - values: Buffer of values to compute the hyperbolic tangent of
    ///   - result: Result buffer
    static func tanh<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    //MARK: Shuffling
    static func scatter<N: NumericType>(reduced: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>, result: ShapedBuffer<N, Device>, axis: Int)
    static func gather<N: NumericType>(expanded: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>, result: ShapedBuffer<N, Device>, axis: Int)
    
    static func permuteAxes<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, arangement: [Int])
    static func permuteAxesAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, arangement: [Int])
    
    static func subscriptRead<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    static func subscriptWrite<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    static func subscriptReadAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    static func subscriptWriteAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    
    static func reverse<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func reverseAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func stack<N>(buffers: [ShapedBuffer<N, Device>], result: ShapedBuffer<N, Device>, axis: Int)
    static func unstackAdd<N: NumericType>(stacked: ShapedBuffer<N, Device>, add: [ShapedBuffer<N, Device>], result: [ShapedBuffer<N, Device>], axis: Int)
    static func unstack<N: NumericType>(stacked: ShapedBuffer<N, Device>, result: [ShapedBuffer<N, Device>], axis: Int)
    
    static func arange<N: NumericType>(lowerBound: N, upperBound: N, result: ShapedBuffer<N, Device>)
    
    //MARK: Convolution Helpers
    static func img2col<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int)
    static func col2img<N: NumericType>(matrix: ShapedBuffer<N, Device>, image: ShapedBuffer<N, Device>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int)
}
