//
//  PackedSequence.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.10.19.
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


public struct PackedSequence<Element: NumericType, Device: DeviceType> {
    private(set) var packed: Tensor<Element, Device>
    let lengths: [Int]
    let offsets: [Int]
    
    public init(packing sequences: [Tensor<Element, Device>]) {
        // sequences: array of [seqlen, x]
        let steps = sequences.map { seq in
            (0 ..< seq.shape[0]).map {seq[$0]}
        }
        var all: [Tensor<Element, Device>] = []
        for t in 0 ..< (steps.map({$0.count}).max() ?? 0) {
            for s in steps where s.count > t {
                all.append(s[t].unsqueezed(at: 0))
            }
        }
        // packed: tensor of [totalLength, x]
        packed = Tensor(stacking: all, along: 0)
        lengths = steps.map {$0.count}
        offsets = lengths.reduce(into: [0], {$0.append($0.last! + $1)}).dropLast()
    }
    
    public func unpacked() -> [Tensor<Element, Device>] {
        let all = (0 ..< packed.shape[0]).map {
            packed[$0]
        }
        return zip(offsets, lengths).map { o, l in
            Tensor(stacking: all[o ..< (o + l)].map {
                $0.unsqueezed(at: 0)
            })
        }
    }
    
    public func batch(at step: Int) -> Tensor<Element, Device> {
        fatalError()
    }
}
