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
    
    /// Element-wise maximum
    /// - Parameters:
    ///   - lhs: First buffer
    ///   - rhs: Second buffer
    ///   - result: Result buffer
    static func max<N: NumericType>(_ lhs: ShapedBuffer<N, Device>, _ rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise maximum
    /// - Parameters:
    ///   - lhs: First buffer
    ///   - rhs: Second buffer
    ///   - result: Result buffer
    ///   - context: Context buffer
    static func max<N: NumericType>(_ lhs: ShapedBuffer<N, Device>, _ rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<N, Device>)
    
    /// Element-wise minimum
    /// - Parameters:
    ///   - lhs: First buffer
    ///   - rhs: Second buffer
    ///   - result: Result buffer
    static func min<N: NumericType>(_ lhs: ShapedBuffer<N, Device>, _ rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Element-wise minimum
    /// - Parameters:
    ///   - lhs: First buffer
    ///   - rhs: Second buffer
    ///   - result: Result buffer
    ///   - context: Context buffer
    static func min<N: NumericType>(_ lhs: ShapedBuffer<N, Device>, _ rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<N, Device>)
    
    //MARK: Shuffling
    
    /// Scatters elements to indices determined by the context along the specified axis.
    /// - Parameters:
    ///   - reduced: Values to scatter
    ///   - context: Index vector
    ///   - result: Buffer to scatter to
    ///   - axis: Axis to scatter along
    static func scatter<N: NumericType>(reduced: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>, result: ShapedBuffer<N, Device>, axis: Int, ignoreIndex: Int32)
    
    /// Gathers elements from indices determined by the context along the specified axis
    /// - Parameters:
    ///   - expanded: Buffer to gather elements from
    ///   - context: Index vector
    ///   - result: Result buffer
    ///   - axis: Axis to gather along
    static func gather<N: NumericType>(expanded: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>, result: ShapedBuffer<N, Device>, axis: Int, ignoreIndex: Int32)
    
    /// Performs an axis permutation / transpose oepration
    /// - Parameters:
    ///   - values: Buffer of values to permute
    ///   - result: Result buffer
    ///   - arangement: Arangement of axes.
    static func permuteAxes<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, arangement: [Int])
    
    /// Performs an axis permutation / transpose oepration and adds another value vector
    /// - Parameters:
    ///   - values: Buffer of values to permute
    ///   - add: Buffer of values to add to the result
    ///   - result: Result buffer
    ///   - arangement: Arangement of axes.
    static func permuteAxesAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, arangement: [Int])
    
    /// Reads elements from the given index
    /// - Parameters:
    ///   - values: Buffer of values to read
    ///   - result: Buffer to write the values to
    ///   - index: Index to read from, nil values indicate that all values along the corresponding axis should be read
    static func subscriptRead<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    
    /// Writes elements to the result tensor at the given index
    /// - Parameters:
    ///   - values: Values to write
    ///   - result: Buffer to write to
    ///   - index: Index to write to, nil values indicate that all values along the corresponding axis should be written
    static func subscriptWrite<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    
    /// Reads elements from the given index and adds a second vector
    /// - Parameters:
    ///   - values: Buffer of values to read
    ///   - add: Buffer to add
    ///   - result: Buffer to write the values to
    ///   - index: Index to read from, nil values indicate that all values along the corresponding axis should be read
    static func subscriptReadAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    
    /// Writes elements to the result tensor at the given index and adds a second vector
    /// - Parameters:
    ///   - values: Values to write
    ///   - add: Buffer to add
    ///   - result: Buffer to write to
    ///   - index: Index to write to, nil values indicate that all values along the corresponding axis should be written
    static func subscriptWriteAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    
    /// Reverses the order of elements along the first dimension of the buffer
    /// - Parameters:
    ///   - values: Buffer to reverse
    ///   - result: Result buffer
    static func reverse<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Reverses the order of elements along the first dimension of the buffer and adds a second buffer
    /// - Parameters:
    ///   - values: Buffer to reverse
    ///   - add: Buffer to add
    ///   - result: Result buffer
    static func reverseAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    /// Stacks the given array of buffers along the stacking axis into the result buffer.
    /// - Parameters:
    ///   - buffers: Buffers to stack
    ///   - result: Result buffer
    ///   - axis: Axis to stack the buffers along
    static func stack<N>(buffers: [ShapedBuffer<N, Device>], result: ShapedBuffer<N, Device>, axis: Int)
    
    /// Reverses the stacking operation and adds a list of buffers to the unstacked results
    /// - Parameters:
    ///   - stacked: Buffer that stores elements of stacked buffers
    ///   - add: Buffers to add to the corresponding result buffers
    ///   - result: Buffers to write the result to
    ///   - axis: Axis to unstack along
    static func unstackAdd<N: NumericType>(stacked: ShapedBuffer<N, Device>, add: [ShapedBuffer<N, Device>], result: [ShapedBuffer<N, Device>], axis: Int)
    
    /// Reverses the stacking operation
    /// - Parameters:
    ///   - stacked: Buffer that stores elements of stacked buffers
    ///   - result: Buffers to write the result to
    ///   - axis: Axis to unstack along
    static func unstack<N: NumericType>(stacked: ShapedBuffer<N, Device>, result: [ShapedBuffer<N, Device>], axis: Int)
    
    /// Writes linear interpolation values from lowerBound to upperBound into the result buffer.
    /// - Parameters:
    ///   - lowerBound: Start value
    ///   - upperBound: End value
    ///   - result: Result buffer
    static func arange<N: NumericType>(lowerBound: N, upperBound: N, result: ShapedBuffer<N, Device>)
    
    //MARK: Convolution Helpers
    
    /// Performs an img2col transformation that extracts all windows for a convolution into a matrix.
    /// - Parameters:
    ///   - values: Image buffer, shape [batchSize, channels, height, width]
    ///   - result: Result buffer, shape [channels * kernelWidth * kernelHeight, number of windows]
    ///   - kernelWidth: Width of the convolution kernel
    ///   - kernelHeight: Height of the convolution kernel
    ///   - padding: Zero padding applied around the input image
    ///   - stride: Stride, with which the window is moved over the input image
    static func img2col<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int)
    
    /// Performs an col2img transformation that aggregates all windows from a convolution matrix into an image tensor.
    /// - Parameters:
    ///   - values: Buffer, shape [channels * kernelWidth * kernelHeight, number of windows]
    ///   - result: Image buffer, shape [batchSize, channels, height, width]
    ///   - kernelWidth: Width of the convolution kernel
    ///   - kernelHeight: Height of the convolution kernel
    ///   - padding: Zero padding applied around the input image
    ///   - stride: Stride, with which the window is moved over the input image
    static func col2img<N: NumericType>(matrix: ShapedBuffer<N, Device>, image: ShapedBuffer<N, Device>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int)
}
