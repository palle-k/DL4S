//
//  MNIST.swift
//  DL4STests
//
//  Created by Palle Klewitz on 05.09.26.
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
import DL4S

/// MNIST images and labels for training and testing.
///
/// Images have the shape `[count, 1, 28, 28]` with values in `0 ..< 1`. Labels have the shape `[count]`.
struct MNISTData: Sendable {
    let trainingImages: Tensor<Float, CPU>
    let trainingLabels: Tensor<Int32, CPU>
    let testImages: Tensor<Float, CPU>
    let testLabels: Tensor<Int32, CPU>
}

/// Loads the MNIST files that are bundled with the test target.
///
/// Each data set is parsed once per process, on first use, and shared between tests.
enum MNIST {
    /// 5,000 training and 1,000 test images. Enough for a quick convergence check.
    static let sample = load(trainingCount: 5_000, testCount: 1_000)

    /// The full data set for the long-running training tests.
    static let full = load(trainingCount: 60_000, testCount: 10_000)

    private static func load(trainingCount: Int, testCount: Int) -> MNISTData {
        MNISTData(
            trainingImages: loadImages(named: "train-images", count: trainingCount),
            trainingLabels: loadLabels(named: "train-labels", count: trainingCount),
            testImages: loadImages(named: "t10k-images", count: testCount),
            testLabels: loadLabels(named: "t10k-labels", count: testCount)
        )
    }

    private static func loadImages(named name: String, count: Int) -> Tensor<Float, CPU> {
        // The idx3 header is 16 bytes: magic number, image count, row count, column count.
        let pixels = loadBytes(named: name, extension: "idx3-ubyte", headerSize: 16, count: count * 28 * 28)
        return Tensor(pixels.map { Float($0) / 256 }, shape: [count, 1, 28, 28])
    }

    private static func loadLabels(named name: String, count: Int) -> Tensor<Int32, CPU> {
        // The idx1 header is 8 bytes: magic number and label count.
        let labels = loadBytes(named: name, extension: "idx1-ubyte", headerSize: 8, count: count)
        return Tensor(labels.map(Int32.init), shape: [count])
    }

    private static func loadBytes(named name: String, extension fileExtension: String, headerSize: Int, count: Int) -> [UInt8] {
        guard let url = Bundle.module.url(forResource: name, withExtension: fileExtension) else {
            fatalError("MNIST resource \(name).\(fileExtension) is missing from the test bundle.")
        }
        do {
            let data = try Data(contentsOf: url)
            precondition(data.count >= headerSize + count, "MNIST resource \(name) holds fewer than \(count) records.")
            return [UInt8](data[headerSize ..< headerSize + count])
        } catch {
            fatalError("MNIST resource \(name).\(fileExtension) cannot be read: \(error)")
        }
    }

    /// Draws `count` images with their labels at random. The generator makes the draw reproducible.
    static func minibatch<Generator: RandomNumberGenerator>(from images: Tensor<Float, CPU>, labels: Tensor<Int32, CPU>, count: Int, using generator: inout Generator) -> (images: Tensor<Float, CPU>, labels: Tensor<Int32, CPU>) {
        let indices = (0 ..< count).map { _ in Int(generator.next(upperBound: UInt(images.shape[0]))) }
        return (
            images: Tensor(stacking: indices.map { images[$0].unsqueezed(at: 0) }),
            labels: Tensor(indices.map { labels[$0].item })
        )
    }

    /// Fraction of rows in `scores` whose largest value is at the position that `labels` names.
    static func accuracy(of scores: Tensor<Float, CPU>, labels: Tensor<Int32, CPU>) -> Float {
        let classCount = scores.shape[1]
        let values = scores.elements
        let expected = labels.elements
        var correct = 0
        for row in 0 ..< scores.shape[0] {
            let rowScores = values[(row * classCount) ..< ((row + 1) * classCount)]
            let predicted = rowScores.indices.max { rowScores[$0] < rowScores[$1] }! - row * classCount
            if Int32(predicted) == expected[row] {
                correct += 1
            }
        }
        return Float(correct) / Float(scores.shape[0])
    }
}
