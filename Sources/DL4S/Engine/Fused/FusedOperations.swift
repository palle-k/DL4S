//
//  FusedOperations.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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

/// High-level operations of a device, such as convolutions, normalizations, activations, losses, and attention, and their first derivatives.
///
/// A fused operation computes a high-level operation, or its first derivative, in one requirement.
/// Every fused operation has a default implementation, which composes the result from tensor operations and so from the basic operations of the ``EngineType``.
/// A device can implement a fused operation with a fused kernel to replace the default.
///
/// Fused operations take and return tensors. They read the values of their arguments and never record a compute graph: their results have no context.
/// A backward requirement adds the gradient of every source that requires a gradient to the accumulated gradient of the source,
/// or stores it when the source has no accumulated gradient yet. It does not change the accumulated gradients of the other sources.
/// Without a gradient graph, the accumulated gradients are uniquely referenced, so an implementation can add to them in place,
/// for example with a GEMM with beta 1.
/// It never takes the result of the forward operation, except for operations whose derivative is cheapest to compute from the result,
/// such as ``tanhBackward(output:outputGradient:accumulating:)``.
///
/// The fused operations of a device are ``DeviceType/FusedOperations``. Other types can conform too, such as
/// ``DefaultFusedOperations``, which uses the default implementation of every requirement.
public protocol FusedOperationsType<Device> {
    associatedtype Device: DeviceType

    // MARK: Convolution and pooling

    /// Computes a 2D convolution and adds a bias.
    /// - Parameters:
    ///   - input: Images, shape [batchSize, inputChannels, height, width]
    ///   - filters: Filters, shape [outputChannels, inputChannels, kernelHeight, kernelWidth]
    ///   - bias: Bias with one value per output channel, shape [outputChannels], or nil for no bias
    ///   - padding: Zero padding applied around the input images
    ///   - stride: Stride, with which the kernel moves over the input images
    /// - Returns: Result without context, shape [batchSize, outputChannels, (height + 2 \* padding - kernelHeight) / stride + 1, (width + 2 \* padding - kernelWidth) / stride + 1]
    static func convolution2d<N: NumericType>(input: Tensor<N, Device>, filters: Tensor<N, Device>, bias: Tensor<N, Device>?, padding: Int, stride: Int) -> Tensor<N, Device>

    /// Computes the gradients of ``convolution2d(input:filters:bias:padding:stride:)``.
    /// - Parameters:
    ///   - input: Images of the forward operation
    ///   - filters: Filters of the forward operation
    ///   - bias: Bias of the forward operation, or nil for no bias
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - padding: Padding of the forward operation
    ///   - stride: Stride of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func convolution2dBackward<N: NumericType>(input: Tensor<N, Device>, filters: Tensor<N, Device>, bias: Tensor<N, Device>?, outputGradient: Tensor<N, Device>, padding: Int, stride: Int, accumulating gradients: inout (input: Tensor<N, Device>?, filters: Tensor<N, Device>?, bias: Tensor<N, Device>?))

    /// Computes a transposed 2D convolution (fractionally strided convolution) and adds a bias.
    ///
    /// The filters with the shape [outputChannels, inputChannels, kernelHeight, kernelWidth] are read in memory order
    /// as a matrix with the shape [inputChannels, outputChannels \* kernelHeight \* kernelWidth].
    /// - Parameters:
    ///   - input: Images, shape [batchSize, inputChannels, height, width]
    ///   - filters: Filters, shape [outputChannels, inputChannels, kernelHeight, kernelWidth]
    ///   - bias: Bias with one value per output channel, shape [outputChannels], or nil for no bias
    ///   - inset: Number of elements that are removed from the edges of the result
    ///   - stride: Stride, with which the kernel moves over the result. Larger strides give larger results.
    /// - Returns: Result without context, shape [batchSize, outputChannels, (height - 1) \* stride - 2 \* inset + kernelHeight, (width - 1) \* stride - 2 \* inset + kernelWidth]
    static func transposedConvolution2d<N: NumericType>(input: Tensor<N, Device>, filters: Tensor<N, Device>, bias: Tensor<N, Device>?, inset: Int, stride: Int) -> Tensor<N, Device>

    /// Computes the gradients of ``transposedConvolution2d(input:filters:bias:inset:stride:)``.
    /// - Parameters:
    ///   - input: Images of the forward operation
    ///   - filters: Filters of the forward operation
    ///   - bias: Bias of the forward operation, or nil for no bias
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - inset: Inset of the forward operation
    ///   - stride: Stride of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func transposedConvolution2dBackward<N: NumericType>(input: Tensor<N, Device>, filters: Tensor<N, Device>, bias: Tensor<N, Device>?, outputGradient: Tensor<N, Device>, inset: Int, stride: Int, accumulating gradients: inout (input: Tensor<N, Device>?, filters: Tensor<N, Device>?, bias: Tensor<N, Device>?))

    /// Selects the largest value of every window of every channel. Padding adds zeros.
    /// - Parameters:
    ///   - input: Images, shape [batchSize, channels, height, width]
    ///   - windowSize: Width and height of a window
    ///   - padding: Zero padding applied around the input images
    ///   - stride: Stride, with which the window moves over the input images
    /// - Returns: Result without context, shape [batchSize, channels, (height + 2 \* padding - windowSize) / stride + 1, (width + 2 \* padding - windowSize) / stride + 1]
    static func maxPooling2d<N: NumericType>(input: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, Device>

    /// Computes the gradient of ``maxPooling2d(input:windowSize:padding:stride:)``.
    ///
    /// The gradient of a window goes to the position of its largest value.
    /// - Parameters:
    ///   - input: Images of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - windowSize: Window size of the forward operation
    ///   - padding: Padding of the forward operation
    ///   - stride: Stride of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func maxPooling2dBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the mean of every window of every channel. Padding adds zeros, which count for the mean.
    /// - Parameters:
    ///   - input: Images, shape [batchSize, channels, height, width]
    ///   - windowSize: Width and height of a window
    ///   - padding: Zero padding applied around the input images
    ///   - stride: Stride, with which the window moves over the input images
    /// - Returns: Result without context, shape [batchSize, channels, (height + 2 \* padding - windowSize) / stride + 1, (width + 2 \* padding - windowSize) / stride + 1]
    static func averagePooling2d<N: NumericType>(input: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, Device>

    /// Computes the gradient of ``averagePooling2d(input:windowSize:padding:stride:)``.
    /// - Parameters:
    ///   - input: Images of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - windowSize: Window size of the forward operation
    ///   - padding: Padding of the forward operation
    ///   - stride: Stride of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func averagePooling2dBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int, accumulating gradient: inout Tensor<N, Device>?)

    // MARK: Activations

    /// Computes the gradient of the element-wise hyperbolic tangent, `outputGradient * (1 - output * output)`.
    /// - Parameters:
    ///   - output: Result of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func tanhBackward<N: NumericType>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the gradient of the element-wise rectified linear unit, `outputGradient` where `input > 0` and 0 elsewhere.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func reluBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the element-wise sigmoid function, `1 / (1 + exp(-input))`.
    /// - Parameter input: Input values
    /// - Returns: Result without context
    static func sigmoid<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradient of ``sigmoid(input:)``, `outputGradient * output * (1 - output)`.
    /// - Parameters:
    ///   - output: Result of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func sigmoidBackward<N: NumericType>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the softmax function along an axis.
    /// - Parameters:
    ///   - input: Input values
    ///   - axis: Axis to normalize along
    /// - Returns: Result without context
    static func softmax<N: NumericType>(input: Tensor<N, Device>, axis: Int) -> Tensor<N, Device>

    /// Computes the gradient of ``softmax(input:axis:)``, `output * (outputGradient - sum(outputGradient * output, axis))`.
    /// - Parameters:
    ///   - output: Result of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - axis: Axis of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func softmaxBackward<N: NumericType>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axis: Int, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the logarithm of the softmax function along an axis.
    /// - Parameters:
    ///   - input: Input values
    ///   - axis: Axis to normalize along
    /// - Returns: Result without context
    static func logSoftmax<N: NumericType>(input: Tensor<N, Device>, axis: Int) -> Tensor<N, Device>

    /// Computes the gradient of ``logSoftmax(input:axis:)``, `outputGradient - exp(output) * sum(outputGradient, axis)`.
    /// - Parameters:
    ///   - output: Result of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - axis: Axis of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func logSoftmaxBackward<N: NumericType>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axis: Int, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the element-wise leaky rectified linear unit, `input` where `input > 0` and `leakage * input` elsewhere.
    /// - Parameters:
    ///   - input: Input values
    ///   - leakage: Slope for negative inputs, broadcastable to the shape of the input
    /// - Returns: Result without context
    static func leakyRelu<N: NumericType>(input: Tensor<N, Device>, leakage: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradients of ``leakyRelu(input:leakage:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - leakage: Leakage of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func leakyReluBackward<N: NumericType>(input: Tensor<N, Device>, leakage: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradients: inout (input: Tensor<N, Device>?, leakage: Tensor<N, Device>?))

    /// Computes the element-wise Gaussian error linear unit, `input * sigmoid(1.702 * input)`.
    /// - Parameter input: Input values
    /// - Returns: Result without context
    static func gelu<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradient of ``gelu(input:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func geluBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the element-wise Swish activation, `input * sigmoid(beta * input)`.
    /// - Parameters:
    ///   - input: Input values
    ///   - beta: Slope of the sigmoid, broadcastable to the shape of the input
    /// - Returns: Result without context
    static func swish<N: NumericType>(input: Tensor<N, Device>, beta: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradients of ``swish(input:beta:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - beta: Beta of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func swishBackward<N: NumericType>(input: Tensor<N, Device>, beta: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradients: inout (input: Tensor<N, Device>?, beta: Tensor<N, Device>?))

    /// Computes the element-wise Mish activation, `input * tanh(log(1 + exp(input)))`.
    /// - Parameter input: Input values
    /// - Returns: Result without context
    static func mish<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradient of ``mish(input:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func mishBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the element-wise LiSHT activation, `input * tanh(input)`.
    /// - Parameter input: Input values
    /// - Returns: Result without context
    static func lisht<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradient of ``lisht(input:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func lishtBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the element-wise exponential linear unit, `input` where `input > 0` and `alpha * (exp(input) - 1)` elsewhere.
    /// - Parameters:
    ///   - input: Input values
    ///   - alpha: Scale of the exponential part, broadcastable to the shape of the input
    /// - Returns: Result without context
    static func elu<N: NumericType>(input: Tensor<N, Device>, alpha: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradients of ``elu(input:alpha:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - alpha: Alpha of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func eluBackward<N: NumericType>(input: Tensor<N, Device>, alpha: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradients: inout (input: Tensor<N, Device>?, alpha: Tensor<N, Device>?))

    /// Computes the element-wise softplus activation, `log(1 + exp(input))`.
    /// - Parameter input: Input values
    /// - Returns: Result without context
    static func softplus<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradient of ``softplus(input:)``, `outputGradient * sigmoid(input)`.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func softplusBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the element-wise squareplus activation, `(input + sqrt(input * input + 4)) / 2`.
    /// - Parameter input: Input values
    /// - Returns: Result without context
    static func squareplus<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradient of ``squareplus(input:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func squareplusBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradient: inout Tensor<N, Device>?)

    // MARK: Reductions

    /// Computes the mean along axes.
    /// - Parameters:
    ///   - input: Values to reduce
    ///   - axes: Axes to reduce along, in ascending order
    /// - Returns: Result without context, with the shape of the input without the reduced axes
    static func reduceMean<N: NumericType>(input: Tensor<N, Device>, axes: [Int]) -> Tensor<N, Device>

    /// Computes the gradient of ``reduceMean(input:axes:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - axes: Axes of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func reduceMeanBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axes: [Int], accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the biased variance along axes, `mean(input * input) - mean(input) * mean(input)`.
    /// - Parameters:
    ///   - input: Values to reduce
    ///   - axes: Axes to reduce along, in ascending order
    /// - Returns: Result without context, with the shape of the input without the reduced axes
    static func variance<N: NumericType>(input: Tensor<N, Device>, axes: [Int]) -> Tensor<N, Device>

    /// Computes the gradient of ``variance(input:axes:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - axes: Axes of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func varianceBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axes: [Int], accumulating gradient: inout Tensor<N, Device>?)

    // MARK: Normalization

    /// Normalizes every sample to a mean of 0 and a standard deviation of 1, then scales and shifts it.
    ///
    /// The result is `(input - mean) / (sqrt(variance) + epsilon) * scale + shift`.
    /// The statistics are computed along the trailing axes that `scale` has. The leading axes are samples.
    /// - Parameters:
    ///   - input: Values to normalize
    ///   - scale: Scale, with the shape of the trailing axes of the input
    ///   - shift: Shift, with the shape of the trailing axes of the input
    ///   - epsilon: Value added to the standard deviation
    /// - Returns: Result without context
    static func layerNormalization<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, epsilon: N) -> Tensor<N, Device>

    /// Computes the gradients of ``layerNormalization(input:scale:shift:epsilon:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - scale: Scale of the forward operation
    ///   - shift: Shift of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - epsilon: Epsilon of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func layerNormalizationBackward<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, outputGradient: Tensor<N, Device>, epsilon: N, accumulating gradients: inout (input: Tensor<N, Device>?, scale: Tensor<N, Device>?, shift: Tensor<N, Device>?))

    /// Normalizes the input along the batch axis with the statistics of the batch, then scales and shifts it.
    ///
    /// The result is `(input - mean) / (sqrt(variance) + epsilon) * scale + shift`,
    /// where mean and variance are the statistics along axis 0.
    /// - Parameters:
    ///   - input: Values to normalize, shape [batchSize, ...]
    ///   - scale: Scale, broadcastable to the shape of the input without the batch axis
    ///   - shift: Shift, broadcastable to the shape of the input without the batch axis
    ///   - epsilon: Value added to the standard deviation
    /// - Returns: Results without context: the normalized values, and the mean and the biased variance of the batch,
    ///   which have the shape of the input without the batch axis.
    static func batchNormalization<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, epsilon: N) -> (output: Tensor<N, Device>, mean: Tensor<N, Device>, variance: Tensor<N, Device>)

    /// Computes the gradients of ``batchNormalization(input:scale:shift:epsilon:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - scale: Scale of the forward operation
    ///   - shift: Shift of the forward operation
    ///   - outputGradient: Gradient of the normalized values of the forward operation
    ///   - epsilon: Epsilon of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func batchNormalizationBackward<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, outputGradient: Tensor<N, Device>, epsilon: N, accumulating gradients: inout (input: Tensor<N, Device>?, scale: Tensor<N, Device>?, shift: Tensor<N, Device>?))

    /// Normalizes the input with given statistics, then scales and shifts it.
    ///
    /// The result is `(input - mean) / (sqrt(variance) + epsilon) * scale + shift`.
    /// - Parameters:
    ///   - input: Values to normalize, shape [batchSize, ...]
    ///   - scale: Scale, broadcastable to the shape of the input without the batch axis
    ///   - shift: Shift, broadcastable to the shape of the input without the batch axis
    ///   - mean: Mean, broadcastable to the shape of the input without the batch axis
    ///   - variance: Variance, broadcastable to the shape of the input without the batch axis
    ///   - epsilon: Value added to the standard deviation
    /// - Returns: Result without context
    static func batchNormalization<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, mean: Tensor<N, Device>, variance: Tensor<N, Device>, epsilon: N) -> Tensor<N, Device>

    /// Computes the gradients of ``batchNormalization(input:scale:shift:mean:variance:epsilon:)``.
    ///
    /// The mean and the variance get no gradient.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - scale: Scale of the forward operation
    ///   - shift: Shift of the forward operation
    ///   - mean: Mean of the forward operation
    ///   - variance: Variance of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - epsilon: Epsilon of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func batchNormalizationBackward<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, mean: Tensor<N, Device>, variance: Tensor<N, Device>, outputGradient: Tensor<N, Device>, epsilon: N, accumulating gradients: inout (input: Tensor<N, Device>?, scale: Tensor<N, Device>?, shift: Tensor<N, Device>?))

    // MARK: Layers

    /// Multiplies a matrix with weights and adds a bias.
    /// - Parameters:
    ///   - input: Input matrix, shape [batchSize, inputSize]
    ///   - weights: Weights, shape [inputSize, outputSize]
    ///   - bias: Bias, shape [outputSize], or nil for no bias
    /// - Returns: Result without context, shape [batchSize, outputSize]
    static func linear<N: NumericType>(input: Tensor<N, Device>, weights: Tensor<N, Device>, bias: Tensor<N, Device>?) -> Tensor<N, Device>

    /// Computes the gradients of ``linear(input:weights:bias:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - weights: Weights of the forward operation
    ///   - bias: Bias of the forward operation, or nil for no bias
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func linearBackward<N: NumericType>(input: Tensor<N, Device>, weights: Tensor<N, Device>, bias: Tensor<N, Device>?, outputGradient: Tensor<N, Device>, accumulating gradients: inout (input: Tensor<N, Device>?, weights: Tensor<N, Device>?, bias: Tensor<N, Device>?))

    /// Sets random elements to zero.
    /// - Parameters:
    ///   - input: Input values
    ///   - rate: Probability, with which an element is set to zero
    /// - Returns: Results without context: the values, and the mask with 1 for every element that is kept and 0 for every element that is set to zero.
    static func dropout<N: NumericType>(input: Tensor<N, Device>, rate: Float) -> (output: Tensor<N, Device>, mask: Tensor<N, Device>)

    /// Computes the gradient of ``dropout(input:rate:)``, `outputGradient * mask`.
    /// - Parameters:
    ///   - mask: Mask of the forward operation
    ///   - outputGradient: Gradient of the values of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func dropoutBackward<N: NumericType>(mask: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradient: inout Tensor<N, Device>?)

    // MARK: Losses

    /// Computes the mean binary cross entropy, `mean(-expected * log(actual) - (1 - expected) * log(1 - actual))`.
    /// - Parameters:
    ///   - expected: Expected probabilities
    ///   - actual: Predicted probabilities in the range (0, 1), with the shape of `expected`
    /// - Returns: Scalar loss without context
    static func binaryCrossEntropy<N: NumericType>(expected: Tensor<N, Device>, actual: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradients of ``binaryCrossEntropy(expected:actual:)``.
    /// - Parameters:
    ///   - expected: Expected probabilities of the forward operation
    ///   - actual: Predicted probabilities of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func binaryCrossEntropyBackward<N: NumericType>(expected: Tensor<N, Device>, actual: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradients: inout (expected: Tensor<N, Device>?, actual: Tensor<N, Device>?))

    /// Computes the mean categorical cross entropy, `mean(-log(actual[expected]))`.
    ///
    /// Rows with the label `ignoreIndex` add 0 to the sum, but count for the mean.
    /// - Parameters:
    ///   - expected: Expected labels, shape [count]
    ///   - actual: Predicted probabilities in the range (0, 1), shape [count, classes]
    ///   - ignoreIndex: Label that is ignored
    /// - Returns: Scalar loss without context
    static func categoricalCrossEntropy<N: NumericType>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, ignoreIndex: Int32) -> Tensor<N, Device>

    /// Computes the gradient of ``categoricalCrossEntropy(expected:actual:ignoreIndex:)``.
    /// - Parameters:
    ///   - expected: Expected labels of the forward operation
    ///   - actual: Predicted probabilities of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - ignoreIndex: Ignored label of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func categoricalCrossEntropyBackward<N: NumericType>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, outputGradient: Tensor<N, Device>, ignoreIndex: Int32, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the mean categorical negative log likelihood, `mean(-actual[expected])`.
    ///
    /// Rows with the label `ignoreIndex` add 0 to the sum, but count for the mean.
    /// - Parameters:
    ///   - expected: Expected labels, shape [count]
    ///   - actual: Predicted log probabilities, shape [count, classes]
    ///   - ignoreIndex: Label that is ignored
    /// - Returns: Scalar loss without context
    static func categoricalNegativeLogLikelihood<N: NumericType>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, ignoreIndex: Int32) -> Tensor<N, Device>

    /// Computes the gradient of ``categoricalNegativeLogLikelihood(expected:actual:ignoreIndex:)``.
    /// - Parameters:
    ///   - expected: Expected labels of the forward operation
    ///   - actual: Predicted log probabilities of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - ignoreIndex: Ignored label of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func categoricalNegativeLogLikelihoodBackward<N: NumericType>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, outputGradient: Tensor<N, Device>, ignoreIndex: Int32, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the sum of squared differences, divided by the number of rows of `expected`.
    ///
    /// The divisor is `expected.shape[0]` when `expected` has two or more axes, and 1 otherwise.
    /// - Parameters:
    ///   - expected: Expected values
    ///   - actual: Predicted values, broadcastable with `expected`
    /// - Returns: Scalar loss without context
    static func meanSquaredError<N: NumericType>(expected: Tensor<N, Device>, actual: Tensor<N, Device>) -> Tensor<N, Device>

    /// Computes the gradients of ``meanSquaredError(expected:actual:)``.
    /// - Parameters:
    ///   - expected: Expected values of the forward operation
    ///   - actual: Predicted values of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func meanSquaredErrorBackward<N: NumericType>(expected: Tensor<N, Device>, actual: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradients: inout (expected: Tensor<N, Device>?, actual: Tensor<N, Device>?))

    /// Computes the mean of the absolute values, multiplied with a factor, `mean(abs(input)) * scale`.
    /// - Parameters:
    ///   - input: Input values
    ///   - scale: Factor of the loss
    /// - Returns: Scalar loss without context
    static func l1Loss<N: NumericType>(input: Tensor<N, Device>, scale: N) -> Tensor<N, Device>

    /// Computes the gradient of ``l1Loss(input:scale:)``. The gradient at 0 is 0.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - scale: Factor of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func l1LossBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, scale: N, accumulating gradient: inout Tensor<N, Device>?)

    /// Computes the mean of the squared values, multiplied with a factor, `mean(input * input) * scale`.
    /// - Parameters:
    ///   - input: Input values
    ///   - scale: Factor of the loss
    /// - Returns: Scalar loss without context
    static func l2Loss<N: NumericType>(input: Tensor<N, Device>, scale: N) -> Tensor<N, Device>

    /// Computes the gradient of ``l2Loss(input:scale:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - scale: Factor of the forward operation
    ///   - gradient: Accumulated gradient of the source. The gradient of the operation is added to it, or stored in it when it is nil.
    static func l2LossBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, scale: N, accumulating gradient: inout Tensor<N, Device>?)

    // MARK: Attention

    /// Computes scaled dot product attention, `softmax(queries × keysᵀ / temperature - 10⁹ \* mask) × values`.
    /// - Parameters:
    ///   - queries: Queries, shape [batchSize, heads, queryCount, keyDim]
    ///   - keys: Keys, shape [batchSize, heads, keyCount, keyDim]
    ///   - values: Values, shape [batchSize, heads, keyCount, valueDim]
    ///   - mask: Mask with 1 for every key that a query must not attend to and 0 elsewhere,
    ///     broadcastable to [batchSize, heads, queryCount, keyCount], or nil for no mask
    ///   - temperature: Divisor of the dot products
    /// - Returns: Result without context, shape [batchSize, heads, queryCount, valueDim]
    static func scaledDotProductAttention<N: NumericType>(queries: Tensor<N, Device>, keys: Tensor<N, Device>, values: Tensor<N, Device>, mask: Tensor<N, Device>?, temperature: N) -> Tensor<N, Device>

    /// Computes the gradients of ``scaledDotProductAttention(queries:keys:values:mask:temperature:)``.
    ///
    /// The mask gets no gradient.
    /// - Parameters:
    ///   - queries: Queries of the forward operation
    ///   - keys: Keys of the forward operation
    ///   - values: Values of the forward operation
    ///   - mask: Mask of the forward operation, or nil for no mask
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - temperature: Temperature of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func scaledDotProductAttentionBackward<N: NumericType>(queries: Tensor<N, Device>, keys: Tensor<N, Device>, values: Tensor<N, Device>, mask: Tensor<N, Device>?, outputGradient: Tensor<N, Device>, temperature: N, accumulating gradients: inout (queries: Tensor<N, Device>?, keys: Tensor<N, Device>?, values: Tensor<N, Device>?))

    /// Computes multi-head attention with input and output projections.
    ///
    /// The operation projects the queries, keys, and values with their weights, splits the projections into heads,
    /// computes ``scaledDotProductAttention(queries:keys:values:mask:temperature:)`` for every head,
    /// joins the heads, and multiplies the result with the output weights.
    /// - Parameters:
    ///   - queries: Queries, shape [batchSize, queryCount, hiddenDim]
    ///   - keys: Keys, shape [batchSize, keyCount, hiddenDim]
    ///   - values: Values, shape [batchSize, keyCount, hiddenDim]
    ///   - mask: Mask with 1 for every key that a query must not attend to and 0 elsewhere,
    ///     broadcastable to [batchSize, heads, queryCount, keyCount], or nil for no mask
    ///   - queryWeights: Query projection, shape [hiddenDim, heads \* keyDim]
    ///   - keyWeights: Key projection, shape [hiddenDim, heads \* keyDim]
    ///   - valueWeights: Value projection, shape [hiddenDim, heads \* valueDim]
    ///   - outputWeights: Output projection, shape [heads \* valueDim, outputDim]
    ///   - heads: Number of attention heads
    ///   - temperature: Divisor of the dot products
    /// - Returns: Result without context, shape [batchSize, queryCount, outputDim]
    static func multiHeadAttention<N: NumericType>(queries: Tensor<N, Device>, keys: Tensor<N, Device>, values: Tensor<N, Device>, mask: Tensor<N, Device>?, queryWeights: Tensor<N, Device>, keyWeights: Tensor<N, Device>, valueWeights: Tensor<N, Device>, outputWeights: Tensor<N, Device>, heads: Int, temperature: N) -> Tensor<N, Device>

    /// Computes the gradients of ``multiHeadAttention(queries:keys:values:mask:queryWeights:keyWeights:valueWeights:outputWeights:heads:temperature:)``.
    ///
    /// The mask gets no gradient.
    /// - Parameters:
    ///   - queries: Queries of the forward operation
    ///   - keys: Keys of the forward operation
    ///   - values: Values of the forward operation
    ///   - mask: Mask of the forward operation, or nil for no mask
    ///   - queryWeights: Query projection of the forward operation
    ///   - keyWeights: Key projection of the forward operation
    ///   - valueWeights: Value projection of the forward operation
    ///   - outputWeights: Output projection of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation
    ///   - heads: Number of heads of the forward operation
    ///   - temperature: Temperature of the forward operation
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func multiHeadAttentionBackward<N: NumericType>(queries: Tensor<N, Device>, keys: Tensor<N, Device>, values: Tensor<N, Device>, mask: Tensor<N, Device>?, queryWeights: Tensor<N, Device>, keyWeights: Tensor<N, Device>, valueWeights: Tensor<N, Device>, outputWeights: Tensor<N, Device>, outputGradient: Tensor<N, Device>, heads: Int, temperature: N, accumulating gradients: inout MultiHeadAttentionGradients<N, Device>)

    // MARK: Recurrent cells

    /// Computes one step of a gated recurrent unit.
    ///
    /// With `z = sigmoid(updateInput + state × updateWeights)`, `r = sigmoid(resetInput + state × resetWeights)`, and
    /// `c = tanh(candidateInput + (r * state) × candidateWeights)`, the new state is `(1 - z) * state + z * c`.
    /// - Parameters:
    ///   - updateInput: Projection of the input for the update gate, including its bias, shape [batchSize, hiddenSize]
    ///   - resetInput: Projection of the input for the reset gate, including its bias, shape [batchSize, hiddenSize]
    ///   - candidateInput: Projection of the input for the candidate state, including its bias, shape [batchSize, hiddenSize]
    ///   - state: Previous state, shape [batchSize, hiddenSize]
    ///   - updateWeights: Weights of the state for the update gate, shape [hiddenSize, hiddenSize]
    ///   - resetWeights: Weights of the state for the reset gate, shape [hiddenSize, hiddenSize]
    ///   - candidateWeights: Weights of the reset state for the candidate state, shape [hiddenSize, hiddenSize]
    /// - Returns: New state without context, shape [batchSize, hiddenSize]
    static func gatedRecurrentUnitStep<N: NumericType>(
        updateInput: Tensor<N, Device>,
        resetInput: Tensor<N, Device>,
        candidateInput: Tensor<N, Device>,
        state: Tensor<N, Device>,
        updateWeights: Tensor<N, Device>,
        resetWeights: Tensor<N, Device>,
        candidateWeights: Tensor<N, Device>,
    ) -> Tensor<N, Device>

    /// Computes the gradients of ``gatedRecurrentUnitStep(updateInput:resetInput:candidateInput:state:updateWeights:resetWeights:candidateWeights:)``.
    /// - Parameters:
    ///   - updateInput: Update gate input of the forward operation
    ///   - resetInput: Reset gate input of the forward operation
    ///   - candidateInput: Candidate input of the forward operation
    ///   - state: Previous state of the forward operation
    ///   - updateWeights: Update gate weights of the forward operation
    ///   - resetWeights: Reset gate weights of the forward operation
    ///   - candidateWeights: Candidate weights of the forward operation
    ///   - outputGradient: Gradient of the new state
    ///   - gradients: Accumulated gradients of the sources. The gradient of every source that requires a gradient is added to its value,
    ///     or stored in it when the value is nil. The values of the other sources do not change.
    static func gatedRecurrentUnitStepBackward<N: NumericType>(
        updateInput: Tensor<N, Device>,
        resetInput: Tensor<N, Device>,
        candidateInput: Tensor<N, Device>,
        state: Tensor<N, Device>,
        updateWeights: Tensor<N, Device>,
        resetWeights: Tensor<N, Device>,
        candidateWeights: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        accumulating gradients: inout GatedRecurrentUnitGradients<N, Device>,
    )

    // MARK: Optimizers

    /// Performs one step of the Adam optimizer for one parameter.
    ///
    /// The moments are updated in place: `firstMoment = beta1 * firstMoment + (1 - beta1) * gradient` and
    /// `secondMoment = beta2 * secondMoment + (1 - beta2) * gradient * gradient`. With AMSGrad, `secondMomentMax`
    /// becomes the element-wise maximum of itself and the second moment, and normalizes the step instead of the second moment.
    /// - Parameters:
    ///   - parameter: Parameter before the step
    ///   - gradient: Gradient of the parameter
    ///   - firstMoment: First moment, updated in place
    ///   - secondMoment: Second moment, updated in place
    ///   - secondMomentMax: Maximum of the second moments, updated in place, or nil without AMSGrad
    ///   - learningRate: Learning rate
    ///   - beta1: Decay rate of the first moment
    ///   - beta2: Decay rate of the second moment
    ///   - epsilon: Value added to the square root of the corrected second moment
    ///   - beta1Power: `beta1` to the power of the step number, for the bias correction of the first moment
    ///   - beta2Power: `beta2` to the power of the step number, for the bias correction of the second moment
    /// - Returns: Parameter after the step without context:
    ///   `parameter - learningRate * m / (sqrt(v) + epsilon)` with `m = firstMoment / (1 - beta1Power)` and `v = secondMoment / (1 - beta2Power)`
    static func adamUpdate<N: NumericType>(
        parameter: Tensor<N, Device>,
        gradient: Tensor<N, Device>,
        firstMoment: inout Tensor<N, Device>,
        secondMoment: inout Tensor<N, Device>,
        secondMomentMax: inout Tensor<N, Device>?,
        learningRate: N,
        beta1: N,
        beta2: N,
        epsilon: N,
        beta1Power: N,
        beta2Power: N,
    ) -> Tensor<N, Device>

    /// Creates the sinusoidal positional encoding of [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    ///
    /// The element at position `p` and index `2i` is `sin(p / 10000^(i / (hiddenSize / 2)))`,
    /// and the element at index `2i + 1` is the cosine of the same value.
    /// - Parameters:
    ///   - length: Number of positions
    ///   - hiddenSize: Number of elements per position, a multiple of 2
    /// - Returns: Encoding without context, shape [length, hiddenSize]
    static func positionalEncoding<N: NumericType>(length: Int, hiddenSize: Int) -> Tensor<N, Device>
}

/// Fused operations that use the default implementation of every requirement.
///
/// A device that implements a fused operation only for some inputs, for example only along the last axis,
/// calls the default implementation for the other inputs.
public struct DefaultFusedOperations<Device: DeviceType>: FusedOperationsType {}

/// Accumulated gradients of the sources of one step of a gated recurrent unit, nil for the sources without a gradient.
public struct GatedRecurrentUnitGradients<Element: NumericType, Device: DeviceType>: Sendable {
    /// Gradient of the update gate input
    public var updateInput: Tensor<Element, Device>?
    /// Gradient of the reset gate input
    public var resetInput: Tensor<Element, Device>?
    /// Gradient of the candidate input
    public var candidateInput: Tensor<Element, Device>?
    /// Gradient of the previous state
    public var state: Tensor<Element, Device>?
    /// Gradient of the update gate weights
    public var updateWeights: Tensor<Element, Device>?
    /// Gradient of the reset gate weights
    public var resetWeights: Tensor<Element, Device>?
    /// Gradient of the candidate weights
    public var candidateWeights: Tensor<Element, Device>?

    /// Creates a set of gradients, all of them nil.
    public init() {}

    /// Creates the gradients from values in the order of the sources of the step: update input, reset input,
    /// candidate input, state, update weights, reset weights, and candidate weights.
    /// - Parameter gradients: Seven gradients, nil for the sources without a gradient
    public init(inSourceOrder gradients: [Tensor<Element, Device>?]) {
        precondition(gradients.count == 7, "A step of a gated recurrent unit has seven sources.")
        (updateInput, resetInput, candidateInput, state) = (gradients[0], gradients[1], gradients[2], gradients[3])
        (updateWeights, resetWeights, candidateWeights) = (gradients[4], gradients[5], gradients[6])
    }

    /// The gradients in the order of the sources of the step: update input, reset input, candidate input, state,
    /// update weights, reset weights, and candidate weights.
    public var inSourceOrder: [Tensor<Element, Device>?] {
        [updateInput, resetInput, candidateInput, state, updateWeights, resetWeights, candidateWeights]
    }

    /// Adds the gradients to the accumulated gradients, or stores them where there is no accumulated gradient yet.
    /// - Parameter gradients: Gradients to add, nil for the sources without a gradient
    public mutating func accumulate(_ gradients: Self) {
        Tensor.accumulate(gradients.updateInput, into: &updateInput)
        Tensor.accumulate(gradients.resetInput, into: &resetInput)
        Tensor.accumulate(gradients.candidateInput, into: &candidateInput)
        Tensor.accumulate(gradients.state, into: &state)
        Tensor.accumulate(gradients.updateWeights, into: &updateWeights)
        Tensor.accumulate(gradients.resetWeights, into: &resetWeights)
        Tensor.accumulate(gradients.candidateWeights, into: &candidateWeights)
    }
}

/// Accumulated gradients of the sources of multi-head attention, nil for the sources without a gradient.
public struct MultiHeadAttentionGradients<Element: NumericType, Device: DeviceType>: Sendable {
    /// Gradient of the queries
    public var queries: Tensor<Element, Device>?
    /// Gradient of the keys
    public var keys: Tensor<Element, Device>?
    /// Gradient of the values
    public var values: Tensor<Element, Device>?
    /// Gradient of the query projection
    public var queryWeights: Tensor<Element, Device>?
    /// Gradient of the key projection
    public var keyWeights: Tensor<Element, Device>?
    /// Gradient of the value projection
    public var valueWeights: Tensor<Element, Device>?
    /// Gradient of the output projection
    public var outputWeights: Tensor<Element, Device>?

    /// Creates a set of gradients.
    /// - Parameters:
    ///   - queries: Gradient of the queries
    ///   - keys: Gradient of the keys
    ///   - values: Gradient of the values
    ///   - queryWeights: Gradient of the query projection
    ///   - keyWeights: Gradient of the key projection
    ///   - valueWeights: Gradient of the value projection
    ///   - outputWeights: Gradient of the output projection
    public init(queries: Tensor<Element, Device>?, keys: Tensor<Element, Device>?, values: Tensor<Element, Device>?, queryWeights: Tensor<Element, Device>?, keyWeights: Tensor<Element, Device>?, valueWeights: Tensor<Element, Device>?, outputWeights: Tensor<Element, Device>?) {
        self.queries = queries
        self.keys = keys
        self.values = values
        self.queryWeights = queryWeights
        self.keyWeights = keyWeights
        self.valueWeights = valueWeights
        self.outputWeights = outputWeights
    }

    /// The gradients in the order of the sources: queries, keys, values, query weights, key weights, value weights, and output weights.
    public var inSourceOrder: [Tensor<Element, Device>?] {
        [queries, keys, values, queryWeights, keyWeights, valueWeights, outputWeights]
    }

    /// Adds the gradients to the accumulated gradients, or stores them where there is no accumulated gradient yet.
    /// - Parameter gradients: Gradients to add, nil for the sources without a gradient
    public mutating func accumulate(_ gradients: Self) {
        Tensor.accumulate(gradients.queries, into: &queries)
        Tensor.accumulate(gradients.keys, into: &keys)
        Tensor.accumulate(gradients.values, into: &values)
        Tensor.accumulate(gradients.queryWeights, into: &queryWeights)
        Tensor.accumulate(gradients.keyWeights, into: &keyWeights)
        Tensor.accumulate(gradients.valueWeights, into: &valueWeights)
        Tensor.accumulate(gradients.outputWeights, into: &outputWeights)
    }
}
