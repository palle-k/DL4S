//
//  File.swift
//  
//
//  Created by Palle Klewitz on 21.05.20.
//

import Foundation
import XCTest
import DL4S
import AF

typealias GPU = ArrayFire

class ArrayFireTests: XCTestCase {
    func testAF() {
        GPU.setOpenCL()
        GPU.printInfo()
        GPU.printMemInfo()
        
        let a = Tensor<Float, ArrayFire>([0, 1, 2, 3, 4])
        let b = Tensor<Float, ArrayFire>([[10], [20], [30]])
        print((a + b).copied(to: CPU.self))
        
        GPU.printMemInfo()
    }
    
}
