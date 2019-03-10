//
//  Primitives.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

import Foundation


func conv2d_same(source: UnsafeBufferPointer<Float>, destination: UnsafeBufferPointer<Float>, kernels: UnsafeMutableBufferPointer<Float>, width: Int, height: Int, depth: Int, dstDepth: Int, kWidth: Int, kHeight: Int) {
    for k in 0 ..< dstDepth {
        let kernel = kernels.advanced(by: kWidth * kHeight * depth * k)
        
        for z in 0 ..< depth {
            for y in 0 ..< height {
                for x in 0 ..< width {
                    var v: Float = 0
                    fatalError()
                }
            }
        }
    }
}
