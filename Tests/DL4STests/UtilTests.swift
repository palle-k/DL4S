//
//  UtilTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 02.05.19.
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

@testable import DL4S
import Foundation
import Testing

struct UtilTests {
    @Test func emptyTensorsPassThroughOperations() {
        var batch = Tensor<Float, CPU>(uninitializedShape: [0, 4])
        batch.requiresGradient = true
        let weights = Tensor<Float, CPU>(repeating: 1, shape: [4, 3], requiresGradient: true)
        let scale = Tensor<Float, CPU>(repeating: 1, shape: [4], requiresGradient: true)
        let shift = Tensor<Float, CPU>(repeating: 0, shape: [4], requiresGradient: true)
        let outputs = [
            batch.softmax(axis: 1),
            batch.layerNormalized(scale: scale, shift: shift),
            batch.matrixMultiplied(with: weights),
            batch + Tensor<Float, CPU>(repeating: 1, shape: [1, 4]),
        ]
        #expect(outputs.map(\.shape) == [[0, 4], [0, 4], [0, 3], [0, 4]])
        let gradients = outputs[2].reduceSum().gradients(of: [batch, weights])
        #expect(gradients[0].shape == [0, 4])
        #expect(gradients[1].elements == [Float](repeating: 0, count: 12))

        let pieces = Tensor<Float, CPU>([1, 2, 3]).unstacked(along: 0, withLengths: [3, 0])
        #expect(pieces.map(\.shape) == [[3], [0]])
        #expect(Tensor<Float, CPU>(uninitializedShape: [3, 0])[1].shape == [0])
    }
}
