//
//  TransformerDecodingTests.swift
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

@testable import DL4S
import Testing

/// Layers and inputs of the decoding tests, with dropout disabled, so that the results are deterministic.
enum DecodingFixture {
    static let hiddenDim = 16

    static func decoder<Element: RandomizableType>(_: Element.Type, attendsToEncoder: Bool, seed: UInt64 = 1) -> TransformerDecoder<Element, CPU> {
        var generator = WyHash(seed: seed)
        var decoder = TransformerDecoder<Element, CPU>(layerCount: 2, heads: 4, keyDim: 4, valueDim: 4, modelDim: hiddenDim, forwardDim: 32, dropout: 0.1, attendsToEncoder: attendsToEncoder, using: &generator)
        decoder.modifyLayers(of: Dropout<Element, CPU>.self) { $0.isActive = false }
        return decoder
    }

    static func random(_ shape: [Int], seed: UInt64) -> Tensor<Double, CPU> {
        var generator = WyHash(seed: seed)
        return Tensor(uniformlyDistributedWithShape: shape, min: -1, max: 1, using: &generator)
    }

    /// Pads a tensor with the shape [1, length, hiddenDim] with zeros to the given length.
    static func padded(_ tensor: Tensor<Double, CPU>, to length: Int) -> Tensor<Double, CPU> {
        guard tensor.shape[1] < length else {
            return tensor
        }
        return Tensor(stacking: [tensor, Tensor(repeating: 0, shape: [1, length - tensor.shape[1], hiddenDim])], along: 1)
    }
}

struct TransformerDecodingTests {
    @Test(arguments: [false, true])
    func decodingStepsMatchTheWholeSequence(attendsToEncoder: Bool) {
        let decoder = DecodingFixture.decoder(Double.self, attendsToEncoder: attendsToEncoder)
        let input = DecodingFixture.random([2, 6, DecodingFixture.hiddenDim], seed: 2)
        let encoded = attendsToEncoder ? EncodedSequence(states: DecodingFixture.random([2, 5, DecodingFixture.hiddenDim], seed: 3), lengths: [5, 3]) : nil
        let expected = decoder(TransformerDecoderInputs(input: input, lengths: [6, 6], encoded: encoded))

        var state = decoder.makeState(batchSize: 2, encoded: encoded)
        // A step of two positions, then one position per step.
        var results = [decoder.decode(input[nil, 0 ..< 2], state: &state)]
        for position in 2 ..< 6 {
            results.append(decoder.decode(input[nil, position ..< position + 1], state: &state))
        }

        expectClose(Tensor(stacking: results, along: 1), expected, tolerance: 1e-20)
        #expect(state.lengths == [6, 6])
        #expect(state.count == 6)
    }

    @Test func paddedPromptsMatchEverySequenceAlone() {
        let decoder = DecodingFixture.decoder(Double.self, attendsToEncoder: false)
        let promptLengths = [4, 1, 3]
        let steps = 3
        let sequences = promptLengths.enumerated().map { DecodingFixture.random([1, $0.element + steps, DecodingFixture.hiddenDim], seed: 10 + UInt64($0.offset)) }

        var state = decoder.makeState(batchSize: sequences.count)
        let prompts = Tensor(stacking: zip(sequences, promptLengths).map { DecodingFixture.padded($0[nil, 0 ..< $1], to: 4) }, along: 0)
        let promptResults = decoder.decode(prompts, lengths: promptLengths, state: &state)
        let stepResults = (0 ..< steps).map { step in
            let next = Tensor(stacking: zip(sequences, promptLengths).map { $0[nil, ($1 + step) ..< ($1 + step + 1)] }, along: 0)
            return decoder.decode(next, state: &state)
        }

        #expect(state.lengths == [7, 4, 6])
        #expect(state.count == 4 + steps)
        for (index, (sequence, promptLength)) in zip(sequences, promptLengths).enumerated() {
            let expected = decoder(TransformerDecoderInputs(input: sequence, lengths: [promptLength + steps]))
            expectClose(promptResults[index ..< index + 1, 0 ..< promptLength], expected[nil, 0 ..< promptLength], tolerance: 1e-20)
            expectClose(Tensor(stacking: stepResults.map { $0[index ..< index + 1] }, along: 1), expected[nil, promptLength ..< promptLength + steps], tolerance: 1e-20)
        }
    }

    @Test func selectedSequencesContinueTheirPrefixes() {
        let decoder = DecodingFixture.decoder(Double.self, attendsToEncoder: false)
        let prefixes = DecodingFixture.random([2, 3, DecodingFixture.hiddenDim], seed: 20)
        let next = DecodingFixture.random([3, 1, DecodingFixture.hiddenDim], seed: 21)
        let selection = [1, 1, 0]

        var state = decoder.makeState(batchSize: 2)
        _ = decoder.decode(prefixes, state: &state)
        state = state.selecting(sequences: selection)
        let result = decoder.decode(next, state: &state)

        for (row, source) in selection.enumerated() {
            let sequence = Tensor(stacking: [prefixes[source ..< source + 1], next[row ..< row + 1]], along: 1)
            let expected = decoder(TransformerDecoderInputs(input: sequence, lengths: [4]))
            expectClose(result[row ..< row + 1], expected[nil, 3 ..< 4], tolerance: 1e-20)
        }
    }

    @Test func sequencesShareAnEncoderOutputWithBatchSizeOne() {
        let decoder = DecodingFixture.decoder(Double.self, attendsToEncoder: true)
        let encoded = EncodedSequence(states: DecodingFixture.random([1, 5, DecodingFixture.hiddenDim], seed: 30), lengths: [4])
        let input = DecodingFixture.random([3, 2, DecodingFixture.hiddenDim], seed: 31)

        var state = decoder.makeState(batchSize: 3, encoded: encoded)
        let result = decoder.decode(input, state: &state)
        let selected = state.selecting(sequences: [2, 0])

        #expect(selected.batchSize == 2)
        for row in 0 ..< 3 {
            let expected = decoder(TransformerDecoderInputs(input: input[row ..< row + 1], lengths: [2], encoded: encoded))
            expectClose(result[row ..< row + 1], expected, tolerance: 1e-20)
        }
    }

    @Test func cachedAttentionMatchesTheAttentionLayer() {
        var generator = WyHash(seed: 40)
        let attention = MultiHeadAttention<Double, CPU>(heads: 4, keyValueHeads: 2, hiddenDim: DecodingFixture.hiddenDim, keyDim: 4, valueDim: 4, dropout: 0, using: &generator)
        let queries = DecodingFixture.random([2, 3, DecodingFixture.hiddenDim], seed: 41)
        let keys = DecodingFixture.random([2, 5, DecodingFixture.hiddenDim], seed: 42)
        let mask = Tensor<Double, CPU>([[0, 0, 1, 0, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]).view(as: 1, 1, 3, 5)

        var expectedAttention = attention
        expectedAttention.dropout.isActive = false
        var cachedAttention = attention
        cachedAttention.dropout.isActive = false
        let expected = expectedAttention((q: queries, k: keys, v: keys, mask: mask))
        let result = cachedAttention(queries: queries, cache: cachedAttention.cache(keys: keys, values: keys), mask: mask)
        expectClose(result, expected, tolerance: 1e-20)
    }

    @Test func cachesAppendCutAndMerge() {
        var generator = WyHash(seed: 50)
        let attention = MultiHeadAttention<Double, CPU>(heads: 4, keyValueHeads: 2, hiddenDim: DecodingFixture.hiddenDim, keyDim: 4, valueDim: 4, using: &generator)
        let inputs = DecodingFixture.random([2, 5, DecodingFixture.hiddenDim], seed: 51)
        let whole = attention.cache(keys: inputs, values: inputs)

        let appended = attention.cache(keys: inputs[nil, 0 ..< 3], values: inputs[nil, 0 ..< 3]).appending(attention.cache(keys: inputs[nil, 3 ..< 5], values: inputs[nil, 3 ..< 5]))
        expectClose(appended.keys, whole.keys, tolerance: 1e-20)
        expectClose(appended.values, whole.values, tolerance: 1e-20)

        let prefix = whole.prefix(count: 3)
        let expectedPrefix = attention.cache(keys: inputs[nil, 0 ..< 3], values: inputs[nil, 0 ..< 3])
        #expect(prefix.count == 3)
        expectClose(prefix.keys, expectedPrefix.keys, tolerance: 1e-20)
        expectClose(prefix.values, expectedPrefix.values, tolerance: 1e-20)

        let merged = whole.merging(prefix.selecting(batches: Tensor([1])))
        #expect(merged.batchSize == 3)
        #expect(merged.count == 5)
        expectClose(merged.keys[0 ..< 2], whole.keys, tolerance: 1e-20)
        expectClose(merged.keys[2 ..< 3, nil, 0 ..< 3], prefix.keys[1 ..< 2], tolerance: 1e-20)
        expectClose(merged.values[2 ..< 3, nil, 0 ..< 3], prefix.values[1 ..< 2], tolerance: 1e-20)
        #expect(merged.keys[2 ..< 3, nil, 3 ..< 5].elements.allSatisfy { $0 == 0 })
    }
}

struct TransformerLanguageModelTests {
    private static func model(dropout: Float = 0.1) -> TransformerLanguageModel<Double, CPU> {
        var generator = WyHash(seed: 60)
        var model = TransformerLanguageModel<Double, CPU>(layers: 2, vocabSize: 11, hiddenDim: DecodingFixture.hiddenDim, heads: 4, keyDim: 4, valueDim: 4, forwardDim: 32, dropout: dropout, using: &generator)
        model.modifyLayers(of: Dropout<Double, CPU>.self) { $0.isActive = false }
        return model
    }

    /// Greedy decoding that computes the whole sequence in every step.
    private static func wholeSequenceGreedy(_ model: TransformerLanguageModel<Double, CPU>, prompt: [Int32], count: Int) -> [Int32] {
        var tokens = prompt
        for _ in 0 ..< count {
            let next = model(TokenSequences([tokens]))[0, tokens.count - 1].argmax()
            tokens.append(Int32(next))
        }
        return Array(tokens.dropFirst(prompt.count))
    }

    @Test func generationMatchesGreedyDecodingOfWholeSequences() {
        let model = Self.model()
        let prompts: [[Int32]] = [[1, 2, 3], [4], [5, 6]]
        let expected = prompts.map { Self.wholeSequenceGreedy(model, prompt: $0, count: 5) }
        #expect(model.generate(prompts: prompts, maxLength: 5) == expected)

        // A sequence that generates the end token stops, and the others continue.
        let endToken = expected[0][1]
        let ended = expected.map { tokens in tokens.firstIndex(of: endToken).map { Array(tokens[...$0]) } ?? tokens }
        #expect(model.generate(prompts: prompts, maxLength: 5, endToken: endToken) == ended)
    }

    @Test func modelPredictsTheNextTokenOfEveryPosition() {
        let model = Self.model()
        let result = model(TokenSequences([[1, 2, 3], [4]]))
        #expect(result.shape == [2, 3, 11])
        let first = model(TokenSequences([[1, 2, 3]]))
        expectClose(result[0 ..< 1], first, tolerance: 1e-20)
    }

    @Test(.trainsModel)
    func modelLearnsToCount() {
        var generator = WyHash(seed: 70)
        var model = TransformerLanguageModel<Float, CPU>(layers: 2, vocabSize: 10, hiddenDim: 32, heads: 4, keyDim: 8, valueDim: 8, forwardDim: 64, dropout: 0, using: &generator)
        var optimizer = Adam<Float, CPU>(learningRate: 0.003)
        for _ in 0 ..< 400 {
            let sequences = (0 ..< 16).map { _ in
                let start = Int32.random(in: 0 ..< 10, using: &generator)
                return (0 ..< 8).map { (start + Int32($0)) % 10 }
            }
            let prediction = model(TokenSequences(sequences.map { Array($0.dropLast()) }))
            let loss = categoricalNegativeLogLikelihood(expected: Tensor(sequences.map { Array($0.dropFirst()) }), actual: prediction)
            model.update { parameters in
                optimizer.update(&parameters, along: loss.gradients(of: parameters))
            }
        }
        model.modifyLayers(of: Dropout<Float, CPU>.self) { $0.isActive = false }
        #expect(model.generate(prompts: [[3], [7, 8], [0, 1, 2]], maxLength: 4) == [[4, 5, 6, 7], [9, 0, 1, 2], [3, 4, 5, 6]])
    }
}
