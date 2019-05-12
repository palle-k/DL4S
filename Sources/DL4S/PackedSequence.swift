//
//  PackedSequence.swift
//  DL4S
//
//  Created by Palle Klewitz on 03.05.19.
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


public class PackedSequence<Element: NumericType, Device: DeviceType> {
    let tensor: Tensor<Element, Device>
    let lengths: [Int]
    let offsets: [Int]
    
    public var tensors: [Tensor<Element, Device>] {
        return zip(offsets, lengths).map { offset, length in
            tensor[offset ..< (offset + length)]
        }
    }
    
    public init(from tensor: Tensor<Element, Device>, withLengths lengths: [Int]) {
        precondition(tensor.dim >= 1, "Tensor must be at least 1-dimensional")
        precondition(lengths.reduce(0, +) == tensor.shape[0], "Length of packed sequences must sum up to tensor shape[0]")
        
        self.tensor = tensor
        self.lengths = lengths
        self.offsets = lengths
            .reduce(into: [0], {$0.append($0.last! + $1)})
            .dropLast()
    }
    
    public init(of tensors: [Tensor<Element, Device>]) {
        self.tensor = stack(tensors)
        self.lengths = tensors.map {$0.shape[0]}
        self.offsets = lengths
            .reduce(into: [0], {$0.append($0.last! + $1)})
            .dropLast()
    }
    
    public subscript(index: Int) -> Tensor<Element, Device> {
        let availableTensorOffsets = zip(offsets, lengths)
            .filter {index < $1}
            .map {$0.0}
        
        return stack(availableTensorOffsets.map { offset in
            self.tensor[offset + index].unsqueeze(at: 0)
        })
    }
}
