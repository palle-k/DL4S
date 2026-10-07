//
//  BatchNorm.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.10.19.
//  Copyright (c) 2019 - 2020 - Palle Klewitz
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
import Synchronization

/// A layer that normalizes its inputs along the batch dimension
///
/// In training mode, the layer normalizes every element of the input with the mean and the variance of the batch,
/// and updates the running statistics with the batch statistics:
/// `running = momentum * running + (1 - momentum) * batch`. The variance is the biased variance.
/// In inference mode, the layer normalizes with the running statistics. Use
/// `model.modifyLayers(of: BatchNorm<Element, Device>.self) { $0.isTraining = false }` to switch a model to inference mode.
///
/// The running statistics have the shape `inputSize`. On an axis on which `inputSize` has the size 1 and the batch
/// statistics do not, the running statistics combine the statistics of all positions of the axis.
///
/// A call in training mode updates the running statistics of the layer and of the copies of the layer that were made
/// before the call: the copies share the running statistics until one of them sets them or loads them from a checkpoint.
public struct BatchNorm<Element: RandomizableType, Device: DeviceType>: LayerType, Codable, Sendable {
    // The forward pass does not mutate the layer, so the statistics that it updates are in a reference type.

    /// Whether the layer normalizes with the statistics of the batch and updates the running statistics (true),
    /// or normalizes with the running statistics (false).
    public var isTraining = true

    /// Learned shift vector
    public var shift: Tensor<Element, Device>

    /// Learned scale vector
    public var scale: Tensor<Element, Device>

    /// Weight of the running statistics when the layer updates them with the statistics of a batch.
    public var momentum: Element

    private var statistics: BatchStatistics<Element, Device>

    /// Running mean of the inputs, which the layer uses in inference mode. It has the shape `inputSize`.
    public var runningMean: Tensor<Element, Device> {
        get {
            statistics.values.mean
        }
        set {
            statistics = BatchStatistics(mean: newValue, variance: runningVariance)
        }
    }

    /// Running biased variance of the inputs, which the layer uses in inference mode. It has the shape `inputSize`.
    public var runningVariance: Tensor<Element, Device> {
        get {
            statistics.values.variance
        }
        set {
            statistics = BatchStatistics(mean: runningMean, variance: newValue)
        }
    }

    /// A layer that normalizes its inputs along the batch dimension.
    /// - Parameters:
    ///   - inputSize: Shape of the scale, the shift, and the running statistics. It must broadcast to the shape of the
    ///     inputs without the batch axis.
    ///   - momentum: Weight of the running statistics when the layer updates them with the statistics of a batch.
    public init(inputSize: [Int], momentum: Element = Element(0.9)) {
        shift = Tensor(repeating: 0, shape: inputSize, requiresGradient: true)
        scale = Tensor(repeating: 1, shape: inputSize, requiresGradient: true)
        statistics = BatchStatistics(mean: Tensor(repeating: 0, shape: inputSize), variance: Tensor(repeating: 1, shape: inputSize))
        self.momentum = momentum

        #if DEBUG
        shift.tag = "shift"
        scale.tag = "scale"
        #endif
    }

    private enum CodingKeys: String, CodingKey {
        case isTraining
        case shift
        case scale
        case momentum
        case runningMean
        case runningVariance
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        isTraining = try container.decode(Bool.self, forKey: .isTraining)
        shift = try container.decode(Tensor<Element, Device>.self, forKey: .shift)
        scale = try container.decode(Tensor<Element, Device>.self, forKey: .scale)
        momentum = try container.decode(Element.self, forKey: .momentum)
        statistics = try BatchStatistics(
            mean: container.decode(Tensor<Element, Device>.self, forKey: .runningMean),
            variance: container.decode(Tensor<Element, Device>.self, forKey: .runningVariance),
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(isTraining, forKey: .isTraining)
        try container.encode(shift, forKey: .shift)
        try container.encode(scale, forKey: .scale)
        try container.encode(momentum, forKey: .momentum)
        let values = statistics.values
        try container.encode(values.mean, forKey: .runningMean)
        try container.encode(values.variance, forKey: .runningVariance)
    }

    public mutating func visitTensors(_ visitor: inout TensorVisitor<Element, Device>) {
        visitor.weight(&shift, named: "shift")
        visitor.weight(&scale, named: "scale")
        // A visitor that replaces the statistics, such as a checkpoint load, gives the layer its own statistics, so that
        // the copies of the layer keep theirs.
        let previous = statistics.values
        var values = previous
        visitor.frozen(&values.mean, named: "runningMean")
        visitor.frozen(&values.variance, named: "runningVariance")
        if values.mean.backpropID != previous.mean.backpropID || values.variance.backpropID != previous.variance.backpropID {
            statistics = BatchStatistics(mean: values.mean, variance: values.variance)
        }
    }

    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        guard isTraining else {
            let values = statistics.values
            return inputs.batchNormalized(scale: scale, shift: shift, mean: values.mean, variance: values.variance)
        }
        let normalized = inputs.batchNormalized(scale: scale, shift: shift)
        updateRunningStatistics(mean: normalized.mean, variance: normalized.variance)
        return normalized.output
    }

    /// Updates the running statistics with the mean and the biased variance of a batch, which have the shape of the
    /// inputs without the batch axis.
    private func updateRunningStatistics(mean batchMean: Tensor<Element, Device>, variance batchVariance: Tensor<Element, Device>) {
        let runningShape = statistics.values.mean.shape
        precondition(batchMean.dim >= runningShape.count, "The shape \(runningShape) of the layer must broadcast to the shape \(batchMean.shape) of the inputs without the batch axis.")
        let paddedShape = Array(repeating: 1, count: batchMean.dim - runningShape.count) + runningShape
        let axes = batchMean.shape.indices.filter { paddedShape[$0] == 1 && batchMean.shape[$0] != 1 }

        var mean = batchMean
        var variance = batchVariance
        if !axes.isEmpty {
            mean = batchMean.reduceMean(along: axes)
            // The variance of all positions is the mean of the variances plus the variance of the means.
            variance = (batchVariance + batchMean * batchMean).reduceMean(along: axes) - mean * mean
        }
        let observed = BatchStatistics<Element, Device>.Values(mean: mean.view(as: runningShape), variance: variance.view(as: runningShape))
        let momentum = Tensor<Element, Device>(momentum)
        statistics.update { values in
            values.mean = observed.mean + (values.mean - observed.mean) * momentum
            values.variance = observed.variance + (values.variance - observed.variance) * momentum
        }
    }
}

// The copies of a layer share the statistics, and their forward passes can run on several threads, so the updates take a lock.
/// The running mean and biased variance of a ``BatchNorm`` layer.
final class BatchStatistics<Element: NumericType, Device: DeviceType>: Sendable {
    /// The running mean and biased variance.
    struct Values: Sendable {
        var mean: Tensor<Element, Device>
        var variance: Tensor<Element, Device>
    }

    private let state: Mutex<Values>

    init(mean: Tensor<Element, Device>, variance: Tensor<Element, Device>) {
        // The statistics never record a gradient graph.
        state = Mutex(Values(mean: mean.detached(), variance: variance.detached()))
    }

    /// The current statistics.
    var values: Values {
        state.withLock { $0 }
    }

    /// Changes the statistics while no other thread reads or changes them.
    func update(_ body: (inout Values) -> Void) {
        state.withLock { values in
            body(&values)
        }
    }
}
