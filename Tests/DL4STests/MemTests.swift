//
//  MemTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 03.03.19.
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
import Testing

struct MemTests {
    @Test func testSliceRead() {
        let a: Tensor<Float, CPU> = Tensor((0 ..< 16).map(Float.init), shape: 4, 4)

        for row in 0 ..< 4 {
            let expectedRow: [Float] = (0 ..< 4).map { Float(row * 4 + $0) }
            #expect(a[row, nil] == Tensor(expectedRow))
        }
        for column in 0 ..< 4 {
            let expectedColumn: [Float] = (0 ..< 4).map { Float($0 * 4 + column) }
            #expect(a[nil, column] == Tensor(expectedColumn))
        }
    }

    @Test func rangeSliceWithFullAxisBetweenRangesReadsAndWritesRegion() {
        let a: Tensor<Float, CPU> = Tensor((0 ..< 120).map(Float.init), shape: 6, 5, 4)
        let expected = (1 ..< 4).flatMap { i in (0 ..< 5).flatMap { j in (1 ..< 3).map { k in Float(i * 20 + j * 4 + k) } } }

        let slice = a[1 ..< 4, nil, 1 ..< 3]
        #expect(slice.shape == [3, 5, 2])
        #expect(slice.elements == expected)

        var b = Tensor<Float, CPU>(repeating: 0, shape: 6, 5, 4)
        b[1 ..< 4, nil, 1 ..< 3] = slice
        #expect(b[1 ..< 4, nil, 1 ..< 3].elements == expected)
        #expect(b.reduceSum().item == expected.reduce(0, +))
    }
}
