//
//  MNISTTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 15.10.19.
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

import Testing
import DL4S

/// Training runs on MNIST.
///
/// The sample tests train briefly on 5,000 images and run in CI.
/// The full tests reproduce the original runs on all 60,000 images and need `DL4S_LONG_TESTS=1`.
///
/// Debug builds without an accelerated backend skip the suite, see `trainsModel`.
@Suite(.serialized, .trainsModel)
struct MNISTTests {
    private typealias Classifier = any LayerType<Tensor<Float, CPU>, Tensor<Float, CPU>, Float, CPU>

    /// Data set of a training run and the accuracy that the model must reach on its test images.
    enum Scale: Sendable {
        case sample
        case full

        var data: MNISTData {
            switch self {
            case .sample: MNIST.sample
            case .full: MNIST.full
            }
        }

        var minimumAccuracy: Float {
            switch self {
            case .sample: 0.6
            case .full: 0.7
            }
        }
    }

    /// Number of optimizer steps, batch size, and learning rate of a training run.
    struct TrainingRun: Sendable {
        let steps: Int
        let batchSize: Int
        let learningRate: Float

        /// About 3 epochs on the 5,000-image sample.
        static let sample = TrainingRun(steps: 60, batchSize: 256, learningRate: 0.001)

        /// A shorter run for the convolutional model, which is slow in debug builds. It reaches about 85 % accuracy on the sample.
        static let convolutionalSample = TrainingRun(steps: 30, batchSize: 64, learningRate: 0.003)

        /// The original run of 100 steps on the full data set.
        static let full = TrainingRun(steps: 100, batchSize: 256, learningRate: 0.001)
    }

    /// Activation functions of the dense classifier. The `logSoftmax` case trains with the negative log likelihood loss.
    enum DenseActivation: CaseIterable, Sendable {
        case relu, swish, mish, gelu, lisht, logSoftmax
    }

    private func makeDenseClassifier(activation: DenseActivation, using generator: inout WyHash) -> Classifier {
        let first = Dense<Float, CPU>(inputSize: 28 * 28, outputSize: 500, using: &generator)
        let second = Dense<Float, CPU>(inputSize: 500, outputSize: 300, using: &generator)
        let output = Dense<Float, CPU>(inputSize: 300, outputSize: 10, using: &generator)

        switch activation {
        case .relu:
            return Sequential {
                first
                Relu<Float, CPU>()
                second
                Relu<Float, CPU>()
                output
                Softmax<Float, CPU>()
            }
        case .swish:
            return Sequential {
                first
                Swish<Float, CPU>(trainableWithChannels: 500)
                second
                Swish<Float, CPU>(trainableWithChannels: 300)
                output
                Softmax<Float, CPU>()
            }
        case .mish:
            return Sequential {
                first
                Mish<Float, CPU>()
                second
                Mish<Float, CPU>()
                output
                Softmax<Float, CPU>()
            }
        case .gelu:
            return Sequential {
                first
                Gelu<Float, CPU>()
                second
                Gelu<Float, CPU>()
                output
                Softmax<Float, CPU>()
            }
        case .lisht:
            return Sequential {
                first
                LiSHT<Float, CPU>()
                second
                LiSHT<Float, CPU>()
                output
                Softmax<Float, CPU>()
            }
        case .logSoftmax:
            return Sequential {
                first
                LiSHT<Float, CPU>()
                second
                LiSHT<Float, CPU>()
                output
                LogSoftmax<Float, CPU>()
            }
        }
    }

    private func makeConvClassifier(using generator: inout WyHash) -> Classifier {
        let firstConvolution = Convolution2D<Float, CPU>(inputChannels: 1, outputChannels: 6, kernelSize: (5, 5), padding: 0, using: &generator)
        let secondConvolution = Convolution2D<Float, CPU>(inputChannels: 6, outputChannels: 16, kernelSize: (5, 5), padding: 0, using: &generator)
        let hidden = Dense<Float, CPU>(inputSize: 16 * 4 * 4, outputSize: 120, using: &generator)
        let output = Dense<Float, CPU>(inputSize: 120, outputSize: 10, using: &generator)

        return Sequential {
            firstConvolution
            LayerNorm<Float, CPU>(inputSize: [6, 24, 24])
            Relu<Float, CPU>()
            MaxPool2D<Float, CPU>(windowSize: 2, stride: 2)
            secondConvolution
            LayerNorm<Float, CPU>(inputSize: [16, 8, 8])
            Relu<Float, CPU>()
            MaxPool2D<Float, CPU>(windowSize: 2, stride: 2)
            Flatten<Float, CPU>()
            hidden
            LayerNorm<Float, CPU>(inputSize: [120])
            Relu<Float, CPU>()
            output
            Softmax<Float, CPU>()
        }
    }

    private func makeGRUClassifier(using generator: inout WyHash) -> Classifier {
        let gru = GRU<Float, CPU>(inputSize: 28, hiddenSize: 128, direction: .forward, using: &generator)
        let output = Dense<Float, CPU>(inputSize: 128, outputSize: 10, using: &generator)

        return Sequential {
            gru
            Lambda<GRU<Float, CPU>.Outputs, Tensor<Float, CPU>, Float, CPU> { outputs in
                outputs.0
            }
            output
            Softmax<Float, CPU>()
        }
    }

    /// Trains the model with Adam and returns its accuracy on the test images.
    ///
    /// - Parameters:
    ///   - model: Model to train.
    ///   - scale: Data set to train and test on.
    ///   - run: Number of steps, batch size, and learning rate.
    ///   - loss: Loss of a prediction, given the labels and the model output.
    ///   - input: Maps a batch of images with the shape `[batch, 1, 28, 28]` to the input shape of the model.
    private func trainAndEvaluate<Layer: LayerType>(
        _ model: Layer,
        scale: Scale,
        run: TrainingRun,
        loss: (Tensor<Int32, CPU>, Tensor<Float, CPU>) -> Tensor<Float, CPU>,
        input: (Tensor<Float, CPU>) -> Tensor<Float, CPU>
    ) -> Float where Layer.Inputs == Tensor<Float, CPU>, Layer.Outputs == Tensor<Float, CPU>, Layer.Parameter == Float, Layer.Device == CPU {
        let data = scale.data
        var generator = WyHash(seed: 1)
        var optimizer = Adam(model: model, learningRate: Tensor(run.learningRate))

        for _ in 0 ..< run.steps {
            let (images, labels) = MNIST.minibatch(from: data.trainingImages, labels: data.trainingLabels, count: run.batchSize, using: &generator)
            let predicted = optimizer.model(input(images))
            optimizer.update(along: loss(labels, predicted).gradients(of: optimizer.model.parameters))
        }

        let testCount = data.testImages.shape[0]
        var correct: Float = 0
        for start in stride(from: 0, to: testCount, by: 1000) {
            let range = start ..< min(start + 1000, testCount)
            let scores = optimizer.model(input(data.testImages[range]))
            correct += MNIST.accuracy(of: scores, labels: data.testLabels[range]) * Float(range.count)
        }
        return correct / Float(testCount)
    }

    private func runDenseClassifier(activation: DenseActivation, scale: Scale) -> Float {
        var generator = WyHash(seed: 42)
        let model = makeDenseClassifier(activation: activation, using: &generator)
        let loss: (Tensor<Int32, CPU>, Tensor<Float, CPU>) -> Tensor<Float, CPU> = switch activation {
        case .logSoftmax: { categoricalNegativeLogLikelihood(expected: $0, actual: $1) }
        default: { categoricalCrossEntropy(expected: $0, actual: $1) }
        }
        return trainAndEvaluate(model, scale: scale, run: scale == .sample ? .sample : .full, loss: loss) { $0.view(as: [-1, 28 * 28]) }
    }

    private func runConvClassifier(scale: Scale) -> Float {
        var generator = WyHash(seed: 42)
        let model = makeConvClassifier(using: &generator)
        return trainAndEvaluate(model, scale: scale, run: scale == .sample ? .convolutionalSample : .full, loss: { categoricalCrossEntropy(expected: $0, actual: $1) }) { $0 }
    }

    private func runGRUClassifier(scale: Scale) -> Float {
        var generator = WyHash(seed: 42)
        let model = makeGRUClassifier(using: &generator)
        return trainAndEvaluate(model, scale: scale, run: scale == .sample ? .sample : .full, loss: { categoricalCrossEntropy(expected: $0, actual: $1) }) {
            // The GRU reads one image row per time step: [sequence length, batch, features].
            $0.view(as: [-1, 28, 28]).permuted(to: [1, 0, 2])
        }
    }

    @Test(arguments: DenseActivation.allCases)
    func denseClassifierLearnsSample(activation: DenseActivation) {
        let accuracy = runDenseClassifier(activation: activation, scale: .sample)
        #expect(accuracy > Scale.sample.minimumAccuracy, "accuracy \(accuracy)")
    }

    @Test(.longRunning, arguments: DenseActivation.allCases)
    func denseClassifierLearnsFullSet(activation: DenseActivation) {
        let accuracy = runDenseClassifier(activation: activation, scale: .full)
        #expect(accuracy > Scale.full.minimumAccuracy, "accuracy \(accuracy)")
    }

    @Test func convClassifierLearnsSample() {
        let accuracy = runConvClassifier(scale: .sample)
        #expect(accuracy > Scale.sample.minimumAccuracy, "accuracy \(accuracy)")
    }

    @Test(.longRunning)
    func convClassifierLearnsFullSet() {
        let accuracy = runConvClassifier(scale: .full)
        #expect(accuracy > Scale.full.minimumAccuracy, "accuracy \(accuracy)")
    }

    @Test func gruClassifierLearnsSample() {
        let accuracy = runGRUClassifier(scale: .sample)
        #expect(accuracy > Scale.sample.minimumAccuracy, "accuracy \(accuracy)")
    }

    @Test(.longRunning)
    func gruClassifierLearnsFullSet() {
        let accuracy = runGRUClassifier(scale: .full)
        #expect(accuracy > Scale.full.minimumAccuracy, "accuracy \(accuracy)")
    }
}
