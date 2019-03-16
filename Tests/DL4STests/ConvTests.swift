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
import DL4S
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
    func testConv() {
        
        let filters = Tensor<Float, CPU>([
            [
                [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
            ],
            [
                [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]
            ]
        ]) / 8
        
        print(filters.shape)
        
        let ((images, _), _) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        let batch = Random.minibatch(from: images, count: 16).unsqueeze(at: 1) // add depth dimension
        
        let filtered = conv2d(input: batch, kernel: filters)
        
        for i in 0 ..< batch.shape[0] {
            let src = batch[i]
            let dst = filtered[i]
            
            let srcImg = NSImage(src)
            try? srcImg?.save(to: "/Users/Palle/Desktop/conv/src_\(i).png")
            
            for j in 0 ..< dst.shape[0] {
                let dstImg = NSImage(dst[i].unsqueeze(at: 0))
                try? dstImg?.save(to: "/Users/Palle/Desktop/conv/dst_\(i)_\(j).png")
                
            }
        }
    }
}
