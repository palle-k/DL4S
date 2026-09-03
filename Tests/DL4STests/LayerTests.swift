//
//  LayerTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 03.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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
import Synchronization
import DL4S

final class LayerTests: XCTestCase {
    private typealias TensorLayer = any LayerType<Tensor<Float, CPU>, Tensor<Float, CPU>, Float, CPU>
    
    /// Returns a copy of the layer with all parameters set to zero.
    ///
    /// The generic parameter opens the existential, so the writable key paths of the concrete layer type can be used.
    private func zeroingParameters<Layer: LayerType>(of layer: Layer) -> Layer where Layer.Parameter == Float, Layer.Device == CPU {
        var copy = layer
        for path in layer.parameterPaths {
            copy[keyPath: path] = Tensor(repeating: 0, shape: layer[keyPath: path].shape)
        }
        return copy
    }
    
    func testExistentialLayerRunsInferenceAndCopiesIndependently() {
        let dense = Dense<Float, CPU>(inputSize: 4, outputSize: 3)
        let layer: TensorLayer = dense
        let input = Tensor<Float, CPU>(uniformlyDistributedWithShape: [2, 4])
        let expected = dense(input)
        
        XCTAssertEqual(layer(input), expected)
        XCTAssertEqual(layer.parameters.count, dense.parameters.count)
        
        let zeroed: TensorLayer = zeroingParameters(of: layer)
        
        XCTAssertEqual(zeroed(input), Tensor(repeating: 0, shape: [2, 3]))
        XCTAssertEqual(layer(input), expected)
        XCTAssertEqual(dense(input), expected)
    }
    
    /// Parameter key paths of a composite layer are appended key paths. They must cross a `@Sendable` boundary
    /// and must still write to the right parameter.
    func testAppendedParameterPathsAreSendable() {
        let model = Sequential {
            Dense<Float, CPU>(inputSize: 3, outputSize: 2)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 2, outputSize: 1)
        }
        let paths = model.parameterPaths
        XCTAssertEqual(paths.count, 4)
        
        let zeroed: @Sendable () -> [Tensor<Float, CPU>] = {
            var copy = model
            for path in paths {
                copy[keyPath: path] = Tensor(repeating: 0, shape: copy[keyPath: path].shape)
            }
            return copy.parameters
        }
        
        let group = DispatchGroup()
        group.enter()
        let result = Mutex<[Tensor<Float, CPU>]>([])
        Thread {
            result.withLock { $0 = zeroed() }
            group.leave()
        }.start()
        group.wait()
        
        let parameters = result.withLock { $0 }
        XCTAssertEqual(parameters.map(\.shape), model.parameters.map(\.shape))
        XCTAssertTrue(parameters.allSatisfy { $0.elements.allSatisfy { $0 == 0 } })
        XCTAssertFalse(model.parameters[0].elements.allSatisfy { $0 == 0 })
    }
}
