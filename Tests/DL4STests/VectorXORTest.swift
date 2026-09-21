//
//  VectorXORTest.swift
//  DL4STests
//
//  Created by Palle Klewitz on 27.02.19.
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

import DL4S
import Foundation
import Testing

struct VectorXORTest {
    @Test func testXNN() {
        let xor_src = Tensor<Float, CPU>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ])
        let xor_dst = Tensor<Float, CPU>([
            [0],
            [1],
            [1],
            [0],
        ])

        var generator = WyHash(seed: 42)
        var net = Sequential {
            Dense<Float, CPU>(inputSize: 2, outputSize: 6, using: &generator)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 6, outputSize: 1, using: &generator)
            Sigmoid<Float, CPU>()
        }
        var optim = Adam<Float, CPU>(learningRate: 0.05)

        var firstLoss: Float = 0
        var lastLoss: Float = 0
        for epoch in 1 ... 100 {
            let pred = net(xor_src)
            let loss = binaryCrossEntropy(expected: xor_dst, actual: pred)
            net.update { parameters in
                optim.update(&parameters, along: loss.gradients(of: parameters))
            }

            if epoch == 1 {
                firstLoss = loss.item
            }
            lastLoss = loss.item
        }
        #expect(lastLoss < firstLoss)

        let predictions = net(xor_src).view(as: -1)

        var correctCount = 0
        for i in 0 ..< 4 where round(predictions[i].item) == xor_dst[i, 0].item {
            correctCount += 1
        }

        let accuracy = Float(correctCount) / 4

        #expect(accuracy == 1)
    }
}
