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

/// Constants that the implementations of the fused operations share. The GPU kernels repeat them in `prelude.metal`.
enum FusedConstants {
    /// Factor of the input in the sigmoid of the approximation of GELU, `input * sigmoid(1.702 * input)`.
    static let geluSlope = 1.702
    /// Factor of the mask that attention subtracts from the scores, so that the softmax sets the weights of the masked
    /// entries to 0.
    static let maskScale = 1e9
}

/// High-level operations of a device, such as convolutions, normalizations, activations, losses, and attention, and their first derivatives.
///
/// A fused operation computes a high-level operation, or its first derivative, in one requirement.
/// Every fused operation has a default implementation, which computes it with the basic operations of the ``EngineType``.
/// A device can implement a fused operation with a fused kernel to replace the default.
///
/// Fused operations work on buffers, like the ``EngineType``: they read their arguments and write their results into the
/// buffers that the caller allocates. The result buffers do not share memory with the arguments.
/// A backward requirement receives one ``GradientBuffer`` per source, or nil for a source whose gradient is not requested.
/// It writes the gradient of the source into the buffer, or adds it when ``GradientBuffer/adds`` is true, for example with a GEMM with beta 1.
/// A backward requirement never takes the result of the forward operation, except for operations whose derivative is cheapest
/// to compute from the result, such as ``tanhBackward(output:outputGradient:inputGradient:)``.
///
/// Every argument has the shape that its requirement states, and no buffer is empty. The gradient of a result has the shape
/// of that result, and a ``GradientBuffer`` has the shape of its source. Other arguments are a programming error, and an
/// implementation can stop the program with a precondition failure. An implementation that does not support all valid arguments, for example
/// a softmax only along the last axis, calls ``DefaultFusedOperations`` for the other arguments.
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
    ///   - result: Result, shape [batchSize, outputChannels, (height + 2 \* padding - kernelHeight) / stride + 1, (width + 2 \* padding - kernelWidth) / stride + 1]
    static func convolution2d<N: NumericType>(input: ShapedBuffer<N, Device>, filters: ShapedBuffer<N, Device>, bias: ShapedBuffer<N, Device>?, padding: Int, stride: Int, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``convolution2d(input:filters:bias:padding:stride:result:)``.
    /// - Parameters:
    ///   - input: Images of the forward operation
    ///   - filters: Filters of the forward operation
    ///   - bias: Bias of the forward operation, or nil for no bias
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - padding: Padding of the forward operation
    ///   - stride: Stride of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    ///   - filterGradient: Gradient of the filters, or nil when it is not requested
    ///   - biasGradient: Gradient of the bias, or nil when it is not requested
    static func convolution2dBackward<N: NumericType>(input: ShapedBuffer<N, Device>, filters: ShapedBuffer<N, Device>, bias: ShapedBuffer<N, Device>?, outputGradient: ShapedBuffer<N, Device>, padding: Int, stride: Int, inputGradient: GradientBuffer<N, Device>?, filterGradient: GradientBuffer<N, Device>?, biasGradient: GradientBuffer<N, Device>?)

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
    ///   - result: Result, shape [batchSize, outputChannels, (height - 1) \* stride - 2 \* inset + kernelHeight, (width - 1) \* stride - 2 \* inset + kernelWidth]
    static func transposedConvolution2d<N: NumericType>(input: ShapedBuffer<N, Device>, filters: ShapedBuffer<N, Device>, bias: ShapedBuffer<N, Device>?, inset: Int, stride: Int, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``transposedConvolution2d(input:filters:bias:inset:stride:result:)``.
    /// - Parameters:
    ///   - input: Images of the forward operation
    ///   - filters: Filters of the forward operation
    ///   - bias: Bias of the forward operation, or nil for no bias
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inset: Inset of the forward operation
    ///   - stride: Stride of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    ///   - filterGradient: Gradient of the filters, or nil when it is not requested
    ///   - biasGradient: Gradient of the bias, or nil when it is not requested
    static func transposedConvolution2dBackward<N: NumericType>(input: ShapedBuffer<N, Device>, filters: ShapedBuffer<N, Device>, bias: ShapedBuffer<N, Device>?, outputGradient: ShapedBuffer<N, Device>, inset: Int, stride: Int, inputGradient: GradientBuffer<N, Device>?, filterGradient: GradientBuffer<N, Device>?, biasGradient: GradientBuffer<N, Device>?)

    /// Selects the largest value of every window of every channel. Padding adds zeros.
    /// - Parameters:
    ///   - input: Images, shape [batchSize, channels, height, width]
    ///   - windowSize: Width and height of a window
    ///   - padding: Zero padding applied around the input images
    ///   - stride: Stride, with which the window moves over the input images
    ///   - result: Result, shape [batchSize, channels, (height + 2 \* padding - windowSize) / stride + 1, (width + 2 \* padding - windowSize) / stride + 1]
    static func maxPooling2d<N: NumericType>(input: ShapedBuffer<N, Device>, windowSize: Int, padding: Int, stride: Int, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``maxPooling2d(input:windowSize:padding:stride:result:)``.
    ///
    /// The gradient of a window goes to the position of its largest value.
    /// - Parameters:
    ///   - input: Images of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - windowSize: Window size of the forward operation
    ///   - padding: Padding of the forward operation
    ///   - stride: Stride of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func maxPooling2dBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, windowSize: Int, padding: Int, stride: Int, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the mean of every window of every channel. Padding adds zeros, which count for the mean.
    /// - Parameters:
    ///   - input: Images, shape [batchSize, channels, height, width]
    ///   - windowSize: Width and height of a window
    ///   - padding: Zero padding applied around the input images
    ///   - stride: Stride, with which the window moves over the input images
    ///   - result: Result, shape [batchSize, channels, (height + 2 \* padding - windowSize) / stride + 1, (width + 2 \* padding - windowSize) / stride + 1]
    static func averagePooling2d<N: NumericType>(input: ShapedBuffer<N, Device>, windowSize: Int, padding: Int, stride: Int, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``averagePooling2d(input:windowSize:padding:stride:result:)``.
    /// - Parameters:
    ///   - input: Images of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - windowSize: Window size of the forward operation
    ///   - padding: Padding of the forward operation
    ///   - stride: Stride of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func averagePooling2dBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, windowSize: Int, padding: Int, stride: Int, inputGradient: GradientBuffer<N, Device>?)

    // MARK: Activations

    /// Computes the gradient of the element-wise hyperbolic tangent, `outputGradient * (1 - output * output)`.
    /// - Parameters:
    ///   - output: Result of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func tanhBackward<N: NumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the gradient of the element-wise rectified linear unit, `outputGradient` where `input > 0` and 0 elsewhere.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func reluBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the element-wise sigmoid function, `1 / (1 + exp(-input))`.
    /// - Parameters:
    ///   - input: Input values
    ///   - result: Result, with the shape of the input
    static func sigmoid<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``sigmoid(input:result:)``, `outputGradient * output * (1 - output)`.
    /// - Parameters:
    ///   - output: Result of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func sigmoidBackward<N: NumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the softmax function along an axis.
    /// - Parameters:
    ///   - input: Input values
    ///   - axis: Axis to normalize along
    ///   - result: Result, with the shape of the input
    static func softmax<N: NumericType>(input: ShapedBuffer<N, Device>, axis: Int, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``softmax(input:axis:result:)``, `output * (outputGradient - sum(outputGradient * output, axis))`.
    /// - Parameters:
    ///   - output: Result of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - axis: Axis of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func softmaxBackward<N: NumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, axis: Int, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the logarithm of the softmax function along an axis.
    /// - Parameters:
    ///   - input: Input values
    ///   - axis: Axis to normalize along
    ///   - result: Result, with the shape of the input
    static func logSoftmax<N: NumericType>(input: ShapedBuffer<N, Device>, axis: Int, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``logSoftmax(input:axis:result:)``, `outputGradient - exp(output) * sum(outputGradient, axis)`.
    /// - Parameters:
    ///   - output: Result of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - axis: Axis of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func logSoftmaxBackward<N: NumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, axis: Int, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the element-wise leaky rectified linear unit, `input` where `input > 0` and `leakage * input` elsewhere.
    /// - Parameters:
    ///   - input: Input values
    ///   - leakage: Slope for negative inputs, broadcastable to the shape of the input
    ///   - result: Result, with the shape of the input
    static func leakyRelu<N: NumericType>(input: ShapedBuffer<N, Device>, leakage: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``leakyRelu(input:leakage:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - leakage: Leakage of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    ///   - leakageGradient: Gradient of the leakage, or nil when it is not requested
    static func leakyReluBackward<N: NumericType>(input: ShapedBuffer<N, Device>, leakage: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?, leakageGradient: GradientBuffer<N, Device>?)

    /// Computes the element-wise Gaussian error linear unit, `input * sigmoid(1.702 * input)`.
    /// - Parameters:
    ///   - input: Input values
    ///   - result: Result, with the shape of the input
    static func gelu<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``gelu(input:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func geluBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the element-wise Swish activation, `input * sigmoid(beta * input)`.
    /// - Parameters:
    ///   - input: Input values
    ///   - beta: Slope of the sigmoid, broadcastable to the shape of the input
    ///   - result: Result, with the shape of the input
    static func swish<N: NumericType>(input: ShapedBuffer<N, Device>, beta: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``swish(input:beta:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - beta: Beta of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    ///   - betaGradient: Gradient of the beta, or nil when it is not requested
    static func swishBackward<N: NumericType>(input: ShapedBuffer<N, Device>, beta: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?, betaGradient: GradientBuffer<N, Device>?)

    /// Computes the element-wise Mish activation, `input * tanh(log(1 + exp(input)))`.
    /// - Parameters:
    ///   - input: Input values
    ///   - result: Result, with the shape of the input
    static func mish<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``mish(input:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func mishBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the element-wise LiSHT activation, `input * tanh(input)`.
    /// - Parameters:
    ///   - input: Input values
    ///   - result: Result, with the shape of the input
    static func lisht<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``lisht(input:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func lishtBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the element-wise exponential linear unit, `input` where `input > 0` and `alpha * (exp(input) - 1)` elsewhere.
    /// - Parameters:
    ///   - input: Input values
    ///   - alpha: Scale of the exponential part, broadcastable to the shape of the input
    ///   - result: Result, with the shape of the input
    static func elu<N: NumericType>(input: ShapedBuffer<N, Device>, alpha: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``elu(input:alpha:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - alpha: Alpha of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    ///   - alphaGradient: Gradient of the alpha, or nil when it is not requested
    static func eluBackward<N: NumericType>(input: ShapedBuffer<N, Device>, alpha: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?, alphaGradient: GradientBuffer<N, Device>?)

    /// Computes the element-wise softplus activation, `log(1 + exp(input))`.
    /// - Parameters:
    ///   - input: Input values
    ///   - result: Result, with the shape of the input
    static func softplus<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``softplus(input:result:)``, `outputGradient * sigmoid(input)`.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func softplusBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the element-wise squareplus activation, `(input + sqrt(input * input + 4)) / 2`.
    /// - Parameters:
    ///   - input: Input values
    ///   - result: Result, with the shape of the input
    static func squareplus<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``squareplus(input:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func squareplusBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?)

    // MARK: Reductions

    /// Computes the biased variance along axes, `mean(input * input) - mean(input) * mean(input)`.
    /// - Parameters:
    ///   - input: Values to reduce
    ///   - axes: Axes to reduce along, in ascending order
    ///   - result: Result, with the shape of the input without the reduced axes
    static func variance<N: NumericType>(input: ShapedBuffer<N, Device>, axes: [Int], result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``variance(input:axes:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - axes: Axes of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func varianceBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, axes: [Int], inputGradient: GradientBuffer<N, Device>?)

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
    ///   - result: Result, with the shape of the input
    static func layerNormalization<N: NumericType>(input: ShapedBuffer<N, Device>, scale: ShapedBuffer<N, Device>, shift: ShapedBuffer<N, Device>, epsilon: N, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``layerNormalization(input:scale:shift:epsilon:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - scale: Scale of the forward operation
    ///   - shift: Shift of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - epsilon: Epsilon of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    ///   - scaleGradient: Gradient of the scale, or nil when it is not requested
    ///   - shiftGradient: Gradient of the shift, or nil when it is not requested
    static func layerNormalizationBackward<N: NumericType>(input: ShapedBuffer<N, Device>, scale: ShapedBuffer<N, Device>, shift: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, epsilon: N, inputGradient: GradientBuffer<N, Device>?, scaleGradient: GradientBuffer<N, Device>?, shiftGradient: GradientBuffer<N, Device>?)

    /// Normalizes the input along the batch axis with the statistics of the batch, then scales and shifts it.
    ///
    /// The result is `(input - mean) / (sqrt(variance) + epsilon) * scale + shift`,
    /// where mean and variance are the statistics along axis 0.
    /// - Parameters:
    ///   - input: Values to normalize, shape [batchSize, ...]
    ///   - scale: Scale, broadcastable to the shape of the input without the batch axis
    ///   - shift: Shift, broadcastable to the shape of the input without the batch axis
    ///   - epsilon: Value added to the standard deviation
    ///   - result: Normalized values, with the shape of the input
    ///   - mean: Mean of the batch, with the shape of the input without the batch axis
    ///   - variance: Biased variance of the batch, with the shape of the input without the batch axis
    static func batchNormalization<N: NumericType>(input: ShapedBuffer<N, Device>, scale: ShapedBuffer<N, Device>, shift: ShapedBuffer<N, Device>, epsilon: N, result: MutableShapedBuffer<N, Device>, mean: MutableShapedBuffer<N, Device>, variance: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``batchNormalization(input:scale:shift:epsilon:result:mean:variance:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - scale: Scale of the forward operation
    ///   - shift: Shift of the forward operation
    ///   - outputGradient: Gradient of the normalized values of the forward operation, with the shape of the input
    ///   - epsilon: Epsilon of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    ///   - scaleGradient: Gradient of the scale, or nil when it is not requested
    ///   - shiftGradient: Gradient of the shift, or nil when it is not requested
    static func batchNormalizationBackward<N: NumericType>(input: ShapedBuffer<N, Device>, scale: ShapedBuffer<N, Device>, shift: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, epsilon: N, inputGradient: GradientBuffer<N, Device>?, scaleGradient: GradientBuffer<N, Device>?, shiftGradient: GradientBuffer<N, Device>?)

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
    ///   - result: Result, with the shape of the input
    static func batchNormalization<N: NumericType>(input: ShapedBuffer<N, Device>, scale: ShapedBuffer<N, Device>, shift: ShapedBuffer<N, Device>, mean: ShapedBuffer<N, Device>, variance: ShapedBuffer<N, Device>, epsilon: N, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``batchNormalization(input:scale:shift:mean:variance:epsilon:result:)``.
    ///
    /// The mean and the variance get no gradient.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - scale: Scale of the forward operation
    ///   - shift: Shift of the forward operation
    ///   - mean: Mean of the forward operation
    ///   - variance: Variance of the forward operation
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - epsilon: Epsilon of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    ///   - scaleGradient: Gradient of the scale, or nil when it is not requested
    ///   - shiftGradient: Gradient of the shift, or nil when it is not requested
    static func batchNormalizationBackward<N: NumericType>(input: ShapedBuffer<N, Device>, scale: ShapedBuffer<N, Device>, shift: ShapedBuffer<N, Device>, mean: ShapedBuffer<N, Device>, variance: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, epsilon: N, inputGradient: GradientBuffer<N, Device>?, scaleGradient: GradientBuffer<N, Device>?, shiftGradient: GradientBuffer<N, Device>?)

    // MARK: Layers

    /// Multiplies a matrix with weights and adds a bias.
    /// - Parameters:
    ///   - input: Input matrix, shape [batchSize, inputSize]
    ///   - weights: Weights, shape [inputSize, outputSize]
    ///   - bias: Bias, shape [outputSize], or nil for no bias
    ///   - result: Result, shape [batchSize, outputSize]
    static func linear<N: NumericType>(input: ShapedBuffer<N, Device>, weights: ShapedBuffer<N, Device>, bias: ShapedBuffer<N, Device>?, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``linear(input:weights:bias:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - weights: Weights of the forward operation
    ///   - bias: Bias of the forward operation, or nil for no bias
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    ///   - weightGradient: Gradient of the weights, or nil when it is not requested
    ///   - biasGradient: Gradient of the bias, or nil when it is not requested
    static func linearBackward<N: NumericType>(input: ShapedBuffer<N, Device>, weights: ShapedBuffer<N, Device>, bias: ShapedBuffer<N, Device>?, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?, weightGradient: GradientBuffer<N, Device>?, biasGradient: GradientBuffer<N, Device>?)

    /// Sets random elements to zero.
    /// - Parameters:
    ///   - input: Input values
    ///   - rate: Probability, with which an element is set to zero
    ///   - result: Values, with the shape of the input
    ///   - mask: Mask with 1 for every element that is kept and 0 for every element that is set to zero, with the shape of the input
    static func dropout<N: NumericType>(input: ShapedBuffer<N, Device>, rate: Float, result: MutableShapedBuffer<N, Device>, mask: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``dropout(input:rate:result:mask:)``, `outputGradient * mask`.
    /// - Parameters:
    ///   - mask: Mask of the forward operation
    ///   - outputGradient: Gradient of the values of the forward operation, with the shape of the input
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func dropoutBackward<N: NumericType>(mask: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?)

    // MARK: Losses

    /// Computes the mean binary cross entropy, `mean(-expected * log(actual) - (1 - expected) * log(1 - actual))`.
    ///
    /// The expected and the predicted values are read in memory order, so their shapes can differ.
    /// - Parameters:
    ///   - expected: Expected probabilities
    ///   - actual: Predicted probabilities in the range (0, 1), with the number of elements of `expected`
    ///   - result: Loss, a scalar
    static func binaryCrossEntropy<N: NumericType>(expected: ShapedBuffer<N, Device>, actual: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``binaryCrossEntropy(expected:actual:result:)``.
    /// - Parameters:
    ///   - expected: Expected probabilities of the forward operation
    ///   - actual: Predicted probabilities of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - expectedGradient: Gradient of the expected values, or nil when it is not requested
    ///   - actualGradient: Gradient of the predicted values, or nil when it is not requested
    static func binaryCrossEntropyBackward<N: NumericType>(expected: ShapedBuffer<N, Device>, actual: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, expectedGradient: GradientBuffer<N, Device>?, actualGradient: GradientBuffer<N, Device>?)

    /// Computes the mean categorical cross entropy, `mean(-log(actual[expected]))`.
    ///
    /// Rows with the label `ignoreIndex` add 0 to the sum, but count for the mean.
    /// - Parameters:
    ///   - expected: Expected labels, with the shape of `actual` without the last axis
    ///   - actual: Predicted probabilities in the range (0, 1), with the classes along the last axis
    ///   - ignoreIndex: Label that is ignored
    ///   - result: Loss, a scalar
    static func categoricalCrossEntropy<N: NumericType>(expected: ShapedBuffer<Int32, Device>, actual: ShapedBuffer<N, Device>, ignoreIndex: Int32, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``categoricalCrossEntropy(expected:actual:ignoreIndex:result:)``.
    /// - Parameters:
    ///   - expected: Expected labels of the forward operation
    ///   - actual: Predicted probabilities of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - ignoreIndex: Ignored label of the forward operation
    ///   - actualGradient: Gradient of the predicted values, or nil when it is not requested
    static func categoricalCrossEntropyBackward<N: NumericType>(expected: ShapedBuffer<Int32, Device>, actual: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, ignoreIndex: Int32, actualGradient: GradientBuffer<N, Device>?)

    /// Computes the mean categorical negative log likelihood, `mean(-actual[expected])`.
    ///
    /// Rows with the label `ignoreIndex` add 0 to the sum, but count for the mean.
    /// - Parameters:
    ///   - expected: Expected labels, with the shape of `actual` without the last axis
    ///   - actual: Predicted log probabilities, with the classes along the last axis
    ///   - ignoreIndex: Label that is ignored
    ///   - result: Loss, a scalar
    static func categoricalNegativeLogLikelihood<N: NumericType>(expected: ShapedBuffer<Int32, Device>, actual: ShapedBuffer<N, Device>, ignoreIndex: Int32, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``categoricalNegativeLogLikelihood(expected:actual:ignoreIndex:result:)``.
    /// - Parameters:
    ///   - expected: Expected labels of the forward operation
    ///   - actual: Predicted log probabilities of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - ignoreIndex: Ignored label of the forward operation
    ///   - actualGradient: Gradient of the predicted values, or nil when it is not requested
    static func categoricalNegativeLogLikelihoodBackward<N: NumericType>(expected: ShapedBuffer<Int32, Device>, actual: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, ignoreIndex: Int32, actualGradient: GradientBuffer<N, Device>?)

    /// Computes the sum of squared differences, divided by the number of rows of `expected`.
    ///
    /// The divisor is `expected.shape[0]` when `expected` has two or more axes, and 1 otherwise.
    /// - Parameters:
    ///   - expected: Expected values
    ///   - actual: Predicted values, broadcastable with `expected`
    ///   - result: Loss, a scalar
    static func meanSquaredError<N: NumericType>(expected: ShapedBuffer<N, Device>, actual: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``meanSquaredError(expected:actual:result:)``.
    /// - Parameters:
    ///   - expected: Expected values of the forward operation
    ///   - actual: Predicted values of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - expectedGradient: Gradient of the expected values, or nil when it is not requested
    ///   - actualGradient: Gradient of the predicted values, or nil when it is not requested
    static func meanSquaredErrorBackward<N: NumericType>(expected: ShapedBuffer<N, Device>, actual: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, expectedGradient: GradientBuffer<N, Device>?, actualGradient: GradientBuffer<N, Device>?)

    /// Computes the mean of the absolute values, multiplied with a factor, `mean(abs(input)) * scale`.
    /// - Parameters:
    ///   - input: Input values
    ///   - scale: Factor of the loss
    ///   - result: Loss, a scalar
    static func l1Loss<N: NumericType>(input: ShapedBuffer<N, Device>, scale: N, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``l1Loss(input:scale:result:)``. The gradient at 0 is 0.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - scale: Factor of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func l1LossBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, scale: N, inputGradient: GradientBuffer<N, Device>?)

    /// Computes the mean of the squared values, multiplied with a factor, `mean(input * input) * scale`.
    /// - Parameters:
    ///   - input: Input values
    ///   - scale: Factor of the loss
    ///   - result: Loss, a scalar
    static func l2Loss<N: NumericType>(input: ShapedBuffer<N, Device>, scale: N, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradient of ``l2Loss(input:scale:result:)``.
    /// - Parameters:
    ///   - input: Input of the forward operation
    ///   - outputGradient: Gradient of the loss, a scalar
    ///   - scale: Factor of the forward operation
    ///   - inputGradient: Gradient of the input, or nil when it is not requested
    static func l2LossBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, scale: N, inputGradient: GradientBuffer<N, Device>?)

    // MARK: Attention

    /// Computes scaled dot product attention, `softmax(queries × keysᵀ / temperature - 10⁹ * mask) × values`.
    ///
    /// The batch axes of the queries, keys, and values are broadcastable, and the result has the broadcast batch axis.
    /// The numbers of heads of the keys and of the values divide the number of heads of the queries. Query head `h` uses
    /// key head `h / (heads / keyHeads)` and value head `h / (heads / valueHeads)`, so a group of query heads shares
    /// one key head and one value head (grouped-query attention). With one key head and one value head, all query heads
    /// share them (multi-query attention).
    /// - Parameters:
    ///   - queries: Queries, shape [batchSize, heads, queryCount, keyDim]
    ///   - keys: Keys, shape [batchSize, keyHeads, keyCount, keyDim]
    ///   - values: Values, shape [batchSize, valueHeads, keyCount, valueDim]
    ///   - mask: Mask with 1 for every key that a query must not attend to and 0 elsewhere,
    ///     broadcastable to [batchSize, heads, queryCount, keyCount], or nil for no mask
    ///   - temperature: Divisor of the dot products
    ///   - result: Result, shape [batchSize, heads, queryCount, valueDim]
    static func scaledDotProductAttention<N: NumericType>(queries: ShapedBuffer<N, Device>, keys: ShapedBuffer<N, Device>, values: ShapedBuffer<N, Device>, mask: ShapedBuffer<N, Device>?, temperature: N, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``scaledDotProductAttention(queries:keys:values:mask:temperature:result:)``.
    ///
    /// The mask gets no gradient.
    /// - Parameters:
    ///   - queries: Queries of the forward operation
    ///   - keys: Keys of the forward operation
    ///   - values: Values of the forward operation
    ///   - mask: Mask of the forward operation, or nil for no mask
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - temperature: Temperature of the forward operation
    ///   - queryGradient: Gradient of the queries, or nil when it is not requested
    ///   - keyGradient: Gradient of the keys, or nil when it is not requested
    ///   - valueGradient: Gradient of the values, or nil when it is not requested
    static func scaledDotProductAttentionBackward<N: NumericType>(queries: ShapedBuffer<N, Device>, keys: ShapedBuffer<N, Device>, values: ShapedBuffer<N, Device>, mask: ShapedBuffer<N, Device>?, outputGradient: ShapedBuffer<N, Device>, temperature: N, queryGradient: GradientBuffer<N, Device>?, keyGradient: GradientBuffer<N, Device>?, valueGradient: GradientBuffer<N, Device>?)

    /// Computes multi-head attention with input and output projections.
    ///
    /// The operation projects the queries, keys, and values with their weights, splits the projections into heads,
    /// computes ``scaledDotProductAttention(queries:keys:values:mask:temperature:result:)`` for every head,
    /// joins the heads, and multiplies the result with the output weights.
    ///
    /// The projections of the keys and the values have `keyHeads` heads, which divides `heads`. The number follows from
    /// the shapes of the weights. With fewer key heads than query heads, a group of query heads shares one key head and
    /// one value head (grouped-query attention).
    /// - Parameters:
    ///   - queries: Queries, shape [batchSize, queryCount, hiddenDim]
    ///   - keys: Keys, shape [batchSize, keyCount, hiddenDim]
    ///   - values: Values, shape [batchSize, keyCount, hiddenDim]
    ///   - mask: Mask with 1 for every key that a query must not attend to and 0 elsewhere,
    ///     broadcastable to [batchSize, heads, queryCount, keyCount], or nil for no mask
    ///   - queryWeights: Query projection, shape [hiddenDim, heads \* keyDim]
    ///   - keyWeights: Key projection, shape [hiddenDim, keyHeads \* keyDim]
    ///   - valueWeights: Value projection, shape [hiddenDim, keyHeads \* valueDim]
    ///   - outputWeights: Output projection, shape [heads \* valueDim, outputDim]
    ///   - heads: Number of query heads
    ///   - temperature: Divisor of the dot products
    ///   - result: Result, shape [batchSize, queryCount, outputDim]
    static func multiHeadAttention<N: NumericType>(queries: ShapedBuffer<N, Device>, keys: ShapedBuffer<N, Device>, values: ShapedBuffer<N, Device>, mask: ShapedBuffer<N, Device>?, queryWeights: ShapedBuffer<N, Device>, keyWeights: ShapedBuffer<N, Device>, valueWeights: ShapedBuffer<N, Device>, outputWeights: ShapedBuffer<N, Device>, heads: Int, temperature: N, result: MutableShapedBuffer<N, Device>)

    /// Computes the gradients of ``multiHeadAttention(queries:keys:values:mask:queryWeights:keyWeights:valueWeights:outputWeights:heads:temperature:result:)``.
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
    ///   - outputGradient: Gradient of the result of the forward operation, with the shape of the result
    ///   - heads: Number of query heads of the forward operation
    ///   - temperature: Temperature of the forward operation
    ///   - gradients: Gradients of the sources, nil for the sources whose gradient is not requested
    static func multiHeadAttentionBackward<N: NumericType>(queries: ShapedBuffer<N, Device>, keys: ShapedBuffer<N, Device>, values: ShapedBuffer<N, Device>, mask: ShapedBuffer<N, Device>?, queryWeights: ShapedBuffer<N, Device>, keyWeights: ShapedBuffer<N, Device>, valueWeights: ShapedBuffer<N, Device>, outputWeights: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, heads: Int, temperature: N, gradients: MultiHeadAttentionGradients<GradientBuffer<N, Device>?>)

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
    ///   - result: New state, shape [batchSize, hiddenSize]
    static func gatedRecurrentUnitStep<N: NumericType>(
        updateInput: ShapedBuffer<N, Device>,
        resetInput: ShapedBuffer<N, Device>,
        candidateInput: ShapedBuffer<N, Device>,
        state: ShapedBuffer<N, Device>,
        updateWeights: ShapedBuffer<N, Device>,
        resetWeights: ShapedBuffer<N, Device>,
        candidateWeights: ShapedBuffer<N, Device>,
        result: MutableShapedBuffer<N, Device>,
    )

    /// Computes the gradients of ``gatedRecurrentUnitStep(updateInput:resetInput:candidateInput:state:updateWeights:resetWeights:candidateWeights:result:)``.
    /// - Parameters:
    ///   - updateInput: Update gate input of the forward operation
    ///   - resetInput: Reset gate input of the forward operation
    ///   - candidateInput: Candidate input of the forward operation
    ///   - state: Previous state of the forward operation
    ///   - updateWeights: Update gate weights of the forward operation
    ///   - resetWeights: Reset gate weights of the forward operation
    ///   - candidateWeights: Candidate weights of the forward operation
    ///   - outputGradient: Gradient of the new state, shape [batchSize, hiddenSize]
    ///   - gradients: Gradients of the sources, nil for the sources whose gradient is not requested
    static func gatedRecurrentUnitStepBackward<N: NumericType>(
        updateInput: ShapedBuffer<N, Device>,
        resetInput: ShapedBuffer<N, Device>,
        candidateInput: ShapedBuffer<N, Device>,
        state: ShapedBuffer<N, Device>,
        updateWeights: ShapedBuffer<N, Device>,
        resetWeights: ShapedBuffer<N, Device>,
        candidateWeights: ShapedBuffer<N, Device>,
        outputGradient: ShapedBuffer<N, Device>,
        gradients: GatedRecurrentUnitGradients<GradientBuffer<N, Device>?>,
    )

    // MARK: Optimizers

    /// Performs one step of the Adam optimizer for one parameter.
    ///
    /// The moments are updated in place: `firstMoment = beta1 * firstMoment + (1 - beta1) * gradient` and
    /// `secondMoment = beta2 * secondMoment + (1 - beta2) * gradient * gradient`. With AMSGrad, `secondMomentMax`
    /// becomes the element-wise maximum of itself and the second moment, and normalizes the step instead of the second moment.
    /// - Parameters:
    ///   - parameter: Parameter before the step
    ///   - gradient: Gradient of the parameter, with the shape of the parameter
    ///   - firstMoment: First moment, with the shape of the parameter, updated in place
    ///   - secondMoment: Second moment, with the shape of the parameter, updated in place
    ///   - secondMomentMax: Maximum of the second moments, with the shape of the parameter, updated in place, or nil without AMSGrad
    ///   - learningRate: Learning rate
    ///   - beta1: Decay rate of the first moment
    ///   - beta2: Decay rate of the second moment
    ///   - epsilon: Value added to the square root of the corrected second moment
    ///   - beta1Power: `beta1` to the power of the step number, for the bias correction of the first moment
    ///   - beta2Power: `beta2` to the power of the step number, for the bias correction of the second moment
    ///   - result: Parameter after the step, with the shape of the parameter, `parameter - learningRate * m / (sqrt(v) + epsilon)` with `m = firstMoment / (1 - beta1Power)` and `v = secondMoment / (1 - beta2Power)`
    static func adamUpdate<N: NumericType>(
        parameter: ShapedBuffer<N, Device>,
        gradient: ShapedBuffer<N, Device>,
        firstMoment: MutableShapedBuffer<N, Device>,
        secondMoment: MutableShapedBuffer<N, Device>,
        secondMomentMax: MutableShapedBuffer<N, Device>?,
        learningRate: N,
        beta1: N,
        beta2: N,
        epsilon: N,
        beta1Power: N,
        beta2Power: N,
        result: MutableShapedBuffer<N, Device>,
    )

    /// Creates the sinusoidal positional encoding of [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    ///
    /// The element at position `p` and index `2i` is `sin(p / 10000^(i / (hiddenSize / 2)))`,
    /// and the element at index `2i + 1` is the cosine of the same value.
    /// - Parameters:
    ///   - length: Number of positions
    ///   - hiddenSize: Number of elements per position, a multiple of 2
    ///   - result: Encoding, shape [length, hiddenSize]
    static func positionalEncoding<N: NumericType>(length: Int, hiddenSize: Int, result: MutableShapedBuffer<N, Device>)
}

/// Fused operations that use the default implementation of every requirement.
///
/// A device that implements a fused operation only for some inputs, for example only along the last axis,
/// calls the default implementation for the other inputs.
public struct DefaultFusedOperations<Device: DeviceType>: FusedOperationsType {}

/// One value for each of the seven sources of a step of a gated recurrent unit, such as its gradient.
///
/// The sources are, in order: the update input, the reset input, the candidate input, the state, the update weights,
/// the reset weights, and the candidate weights.
public struct GatedRecurrentUnitGradients<Gradient> {
    /// Gradient of the update gate input
    public var updateInput: Gradient
    /// Gradient of the reset gate input
    public var resetInput: Gradient
    /// Gradient of the candidate input
    public var candidateInput: Gradient
    /// Gradient of the previous state
    public var state: Gradient
    /// Gradient of the update gate weights
    public var updateWeights: Gradient
    /// Gradient of the reset gate weights
    public var resetWeights: Gradient
    /// Gradient of the candidate weights
    public var candidateWeights: Gradient

    /// Creates the gradients from values in the order of the sources.
    /// - Parameter gradients: Seven values
    public init(inSourceOrder gradients: [Gradient]) {
        precondition(gradients.count == 7, "A step of a gated recurrent unit has seven sources.")
        (updateInput, resetInput, candidateInput, state) = (gradients[0], gradients[1], gradients[2], gradients[3])
        (updateWeights, resetWeights, candidateWeights) = (gradients[4], gradients[5], gradients[6])
    }

    /// The values in the order of the sources.
    public var inSourceOrder: [Gradient] {
        [updateInput, resetInput, candidateInput, state, updateWeights, resetWeights, candidateWeights]
    }
}

/// One value for each of the seven sources of multi-head attention, such as its gradient.
///
/// The sources are, in order: the queries, the keys, the values, the query weights, the key weights, the value weights,
/// and the output weights.
public struct MultiHeadAttentionGradients<Gradient> {
    /// Gradient of the queries
    public var queries: Gradient
    /// Gradient of the keys
    public var keys: Gradient
    /// Gradient of the values
    public var values: Gradient
    /// Gradient of the query projection
    public var queryWeights: Gradient
    /// Gradient of the key projection
    public var keyWeights: Gradient
    /// Gradient of the value projection
    public var valueWeights: Gradient
    /// Gradient of the output projection
    public var outputWeights: Gradient

    /// Creates the gradients from values in the order of the sources.
    /// - Parameter gradients: Seven values
    public init(inSourceOrder gradients: [Gradient]) {
        precondition(gradients.count == 7, "Multi-head attention has seven sources.")
        (queries, keys, values) = (gradients[0], gradients[1], gradients[2])
        (queryWeights, keyWeights, valueWeights, outputWeights) = (gradients[3], gradients[4], gradients[5], gradients[6])
    }

    /// The values in the order of the sources.
    public var inSourceOrder: [Gradient] {
        [queries, keys, values, queryWeights, keyWeights, valueWeights, outputWeights]
    }
}
