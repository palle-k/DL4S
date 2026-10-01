//
//  GPUDecodingTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 01.10.26.
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

#if canImport(Metal) && canImport(MetalPerformanceShaders)
@testable import DL4S
import Testing

extension GPUTests {
    @Suite(.serialized)
    struct GPUDecodingTests {
        /// Decodes padded prompts, two steps, and a step after a selection of sequences.
        private func decodeSteps<D: DeviceType>(_ decoder: TransformerDecoder<Float, D>, input: Tensor<Float, D>, next: Tensor<Float, D>, encoder: Tensor<Float, D>?) -> [Tensor<Float, D>] {
            let encoded = encoder.map { EncodedSequence(states: $0, lengths: [5, 3]) }
            var state = decoder.makeState(batchSize: 2, encoded: encoded)
            let prompts = decoder.decode(input[nil, 0 ..< 3], lengths: [3, 2], state: &state)
            let steps = [decoder.decode(input[nil, 3 ..< 4], state: &state), decoder.decode(input[nil, 4 ..< 5], state: &state)]
            state = state.selecting(sequences: [1, 0, 1])
            // The positions after the length of a prompt are padding, whose results are undefined.
            return [prompts[0 ..< 1], prompts[1 ..< 2, 0 ..< 2]] + steps + [decoder.decode(next, state: &state)]
        }

        @Test(arguments: [false, true])
        func decodingStepsMatchCPU(attendsToEncoder: Bool) {
            let cpuDecoder = DecodingFixture.decoder(Float.self, attendsToEncoder: attendsToEncoder)
            let input = GPUTest.random([2, 5, DecodingFixture.hiddenDim], seed: 80)
            let next = GPUTest.random([3, 1, DecodingFixture.hiddenDim], seed: 81)
            let encoder = attendsToEncoder ? GPUTest.random([2, 5, DecodingFixture.hiddenDim], seed: 82) : nil
            GPUTest.compare("decoding, encoder attention: \(attendsToEncoder)") { gpu in
                guard gpu else {
                    return decodeSteps(cpuDecoder, input: input, next: next, encoder: encoder)
                }
                var decoder = TransformerDecoder<Float, GPU>(layerCount: 2, heads: 4, keyDim: 4, valueDim: 4, modelDim: DecodingFixture.hiddenDim, forwardDim: 32, dropout: 0.1, attendsToEncoder: attendsToEncoder)
                decoder.modifyLayers(of: Dropout<Float, GPU>.self) { $0.isActive = false }
                let weights = cpuDecoder.parameters
                decoder.update { parameters in
                    parameters = weights.map { Tensor<Float, GPU>($0, requiresGradient: true) }
                }
                return GPUTest.host(decodeSteps(decoder, input: Tensor(input), next: Tensor(next), encoder: encoder.map { Tensor<Float, GPU>($0) }))
            }
        }
    }
}
#endif
