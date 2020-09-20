//
//  File.swift
//  
//
//  Created by Palle Klewitz on 20.09.20.
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

func makeEncoderMasks<Element, Device>(sequenceLengths: [Int]) -> Tensor<Element, Device> {
    let maxInLen = sequenceLengths.reduce(0, max)
    let batchSize = sequenceLengths.count
    
    return Tensor<Element, Device>(sequenceLengths.map {
        Array(repeating: 0, count: $0) + Array(repeating: 1, count: maxInLen - $0)
    }).view(as: batchSize, 1, 1, maxInLen) // TODO: Check if maxLen in 3rd or 4th position
}

func makeDecoderMasks<Element, Device>(sequenceLengths: [Int]) -> Tensor<Element, Device> {
    let batchSize = sequenceLengths.count
    let maxLen = sequenceLengths.reduce(0, max)
    
    let decoderSeqMask = Tensor<Element, Device>(sequenceLengths.map {
        Array(repeating: 0, count: $0) + Array(repeating: 1, count: maxLen - $0)
    })
    
    let decoderCausalMask = Tensor<Element, Device>(repeating: 1, shape: maxLen, maxLen)
        .bandMatrix(belowDiagonal: -1, aboveDiagonal: nil) // [maxLen, maxLen]
    
    return 1 - relu(1 - decoderSeqMask.view(as: batchSize, 1, 1, maxLen) - decoderCausalMask)
}
