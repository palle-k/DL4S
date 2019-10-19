//
//  ConvTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 13.03.19.
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

import XCTest
@testable import DL4S
import AppKit

extension NSImage {
    func save(to path: String, type: NSBitmapImageRep.FileType = .png) throws {
        guard let imgData = self.tiffRepresentation, let rep = NSBitmapImageRep(data: imgData) else {
            throw NSError()
        }
        guard let png = rep.representation(using: type, properties: [:]) else {
            throw NSError()
        }
        try png.write(to: URL(fileURLWithPath: path))
    }
}

class ConvTests: XCTestCase {
    func testIm2col() {
        let a = XTensor<Float, CPU>((0 ..< 16).map(Float.init), shape: 1, 1, 4, 4)
        let c = a.repeated(4)
        let d = c * XTensor<Float, CPU>([1,0.5,0.25,0.125]).view(as: 4, 1, 1, 1)
        
        print(d)
        
        let result = XTensor<Float, CPU>(repeating: 0, shape: 9, 64)
        CPU.Engine.img2col(values: d.values, result: result.values, kernelWidth: 3, kernelHeight: 3, padding: 1, stride: 1)
        print(result.permuted(to: 1, 0))
    }
    
    func testConv1() {
        let a = XTensor<Float, CPU>((0 ..< 16).map(Float.init), shape: 1, 1, 4, 4)
        let c = a.repeated(4)
        let d = c * XTensor<Float, CPU>([1,0.5,0.25,0.125]).view(as: 4, 1, 1, 1)
        
        let filters = XTensor<Float, CPU>([
            [
                [[1]]
            ],
            [
                [[-1]]
            ]
        ])
        
        print(d.convolved2d(filters: filters))
    }
    
    func testConv() {
        let filters = XTensor<Float, CPU>([
            [
                [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]]
            ],
            [
                [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
            ]
        ]) / XTensor<Float, CPU>([16, 4]).view(as: -1, 1, 1, 1)
        
        print(filters.shape)
        
        let ((images, _), _) = XMNIST.loadMNIST(from: "/Users/Palle/Developer/DL4S/", type: Float.self, device: CPU.self)
        
        let batch = Random.minibatch(from: images, count: 64)
        
        let filtered = batch.convolved2d(filters: filters)
        
        for i in 0 ..< batch.shape[0] {
            let src = batch[i]
            let dst = filtered[i]
            
            let srcImg = NSImage(src)
            try? srcImg?.save(to: "/Users/Palle/Desktop/conv/src_\(i).png")
            
            for j in 0 ..< dst.shape[0] {
                let dstImg = NSImage(dst[j].permuted(to: 1, 0).unsqueezed(at: 0))
                try? dstImg?.save(to: "/Users/Palle/Desktop/conv/dst_\(i)_\(j).png")
            }
        }
    }
}
