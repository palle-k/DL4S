//
//  ConvTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 13.03.19.
//

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
