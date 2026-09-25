//
//  GPUEngineTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 24.09.26.
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
import Foundation
import Testing

extension GPUTests {
    @Suite(.serialized)
    struct GPUEngineTests {
        private func compare(_ name: String, tolerance: Float = 1e-3, _ body: (Bool) -> [Tensor<Float, CPU>], sourceLocation: SourceLocation = #_sourceLocation) {
            GPUTest.compare(name, tolerance: tolerance, body, sourceLocation: sourceLocation)
        }

        private func random(_ shape: [Int], seed: UInt64, min: Float = -1, max: Float = 1) -> Tensor<Float, CPU> {
            GPUTest.random(shape, seed: seed, min: min, max: max)
        }

        private func run<Result>(on gpu: Bool, _ inputs: [Tensor<Float, CPU>], cpu: ([Tensor<Float, CPU>]) -> Result, gpu gpuBody: ([Tensor<Float, GPU>]) -> Result) -> Result {
            GPUTest.run(on: gpu, inputs, cpu: cpu, gpu: gpuBody)
        }

        // MARK: Tests

        @Test(arguments: GPUShaderSource.all.map(\.name))
        func kernelsCompile(group: String) throws {
            let source = try #require(GPUShaderSource.all.first { $0.name == group })
            let error = GPUContext.current.kernels.compile(source)
            #expect(error == nil, "\(String(describing: error))")
        }

        @Test func transferKeepsValues() {
            let values = random([3, 5, 7], seed: 1)
            let copy = Tensor<Float, CPU>(Tensor<Float, GPU>(values))
            #expect(copy.shape == values.shape)
            #expect(copy.elements == values.elements)
            let scalar = Tensor<Float, GPU>(42)
            #expect(scalar.item == 42)
        }

        @Test(arguments: [[1000], [37, 129], [4, 3, 1025]])
        func elementwiseOperationsMatchCPU(shape: [Int]) {
            let (a, b) = (random(shape, seed: 2), random(shape, seed: 3, min: 0.5, max: 2))
            func body<D: DeviceType>(_ a: Tensor<Float, D>, _ b: Tensor<Float, D>) -> [Tensor<Float, D>] {
                [a + b, a - b, a * b, a / b, -a, a.exp(), b.log(), b.sqrt(), a.rectifiedLinear(), a.heaviside(), a.tanh(), a.sine(), a.cosine(), a.sigmoid()]
            }
            compare("elementwise \(shape)") { gpu in
                run(on: gpu, [a, b], cpu: { body($0[0], $0[1]) }, gpu: { body($0[0], $0[1]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test(arguments: [
            ([5, 7], [7]), ([5, 7], [5, 1]), ([5, 7], []), ([], [5, 7]), ([3, 1, 5], [4, 1]), ([2, 3, 4, 5], [3, 1, 5]), ([1, 7], [5, 1]),
        ])
        func broadcastOperationsMatchCPU(lhsShape: [Int], rhsShape: [Int]) {
            let (a, b) = (random(lhsShape, seed: 4), random(rhsShape, seed: 5, min: 0.5, max: 2))
            func body<D: DeviceType>(_ a: Tensor<Float, D>, _ b: Tensor<Float, D>) -> [Tensor<Float, D>] {
                [a + b, a - b, a * b, a / b, b - a, b / a]
            }
            compare("broadcast \(lhsShape) \(rhsShape)") { gpu in
                run(on: gpu, [a, b], cpu: { body($0[0], $0[1]) }, gpu: { body($0[0], $0[1]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test(arguments: [
            ([100], [0]), ([33, 70], [0]), ([33, 70], [1]), ([33, 70], [0, 1]), ([4, 5, 6], [1]), ([4, 5, 6], [0, 2]), ([4, 5, 6, 7], [1, 3]),
            ([3, 40000], [1]), ([40000, 3], [0]), ([2, 3000, 4], [1]), ([1, 1, 5], [1]), ([7, 1, 3], [1]),
        ])
        func reductionsMatchCPU(shape: [Int], axes: [Int]) {
            let a = random(shape, seed: 6)
            func body<D: DeviceType>(_ a: Tensor<Float, D>) -> [Tensor<Float, D>] {
                [a.reduceSum(along: axes), a.reduceMean(along: axes), a.reduceMax(along: axes), a.reduceSum(), a.reduceMean()]
            }
            compare("reduce \(shape) \(axes)") { gpu in
                run(on: gpu, [a], cpu: { body($0[0]) }, gpu: { body($0[0]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test(arguments: [[5, 300], [300, 5], [7, 9, 11]])
        func argumentsOfMaximaMatchCPU(shape: [Int]) {
            let a = Tensor<Float, CPU>(random(shape, seed: 7), requiresGradient: true)
            // The gradient of a maximum scatters into the position of the maximum, which the context records.
            func body<D: DeviceType>(_ a: Tensor<Float, D>) -> [Tensor<Float, D>] {
                let maxima = a.reduceMax(along: [a.dim - 1])
                return [maxima, maxima.reduceSum().gradients(of: [a])[0]]
            }
            compare("argmax \(shape)") { gpu in
                run(on: gpu, [a], cpu: { body($0[0]) }, gpu: { body($0[0]).map { Tensor<Float, CPU>($0) } })
            }
            let indices = Tensor<Int32, CPU>(Tensor<Float, GPU>(a).argmax(along: a.dim - 1))
            #expect(indices.elements == a.argmax(along: a.dim - 1).elements)
        }

        @Test(arguments: [
            (1, 1, 1, false, false), (17, 33, 65, false, false), (64, 64, 64, true, false), (70, 130, 50, false, true), (129, 65, 257, true, true),
            (512, 512, 512, false, false), (300, 700, 400, true, false), (1024, 256, 768, false, true), (6, 25, 20000, false, true), (40, 30, 3000, true, false), (1, 70, 900, false, true), (1, 70, 900, true, false),
        ])
        func matrixProductsMatchCPU(rows: Int, columns: Int, inner: Int, transposeLeft: Bool, transposeRight: Bool) {
            let a = random(transposeLeft ? [inner, rows] : [rows, inner], seed: 8)
            let b = random(transposeRight ? [columns, inner] : [inner, columns], seed: 9)
            func body<D: DeviceType>(_ a: Tensor<Float, D>, _ b: Tensor<Float, D>) -> [Tensor<Float, D>] {
                [a.matrixMultiplied(with: b, transposeSelf: transposeLeft, transposeOther: transposeRight)]
            }
            compare("gemm \(rows)x\(columns)x\(inner) \(transposeLeft) \(transposeRight)") { gpu in
                run(on: gpu, [a, b], cpu: { body($0[0], $0[1]) }, gpu: { body($0[0], $0[1]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test(arguments: [
            (17, 33, 65, false, false), (70, 130, 50, false, true), (129, 65, 257, true, true), (40, 30, 300, true, false), (1, 70, 90, false, true),
        ])
        func productsOfMetalPerformanceShadersMatchCPU(rows: Int, columns: Int, inner: Int, transposeLeft: Bool, transposeRight: Bool) {
            GPUContext.current.supportsMatrixKernels = false
            defer { GPUContext.current.supportsMatrixKernels = true }
            let a = random([3] + (transposeLeft ? [inner, rows] : [rows, inner]), seed: 8)
            let b = random(transposeRight ? [columns, inner] : [inner, columns], seed: 9)
            let c = random([3] + (transposeRight ? [columns, inner] : [inner, columns]), seed: 10)
            func body<D: DeviceType>(_ a: Tensor<Float, D>, _ b: Tensor<Float, D>, _ c: Tensor<Float, D>) -> [Tensor<Float, D>] {
                [
                    a[0].matrixMultiplied(with: b, transposeSelf: transposeLeft, transposeOther: transposeRight),
                    a.broadcastMatrixMultiplied(with: c, transposeSelf: transposeLeft, transposeOther: transposeRight),
                ]
            }
            compare("mps \(rows)x\(columns)x\(inner) \(transposeLeft) \(transposeRight)") { gpu in
                run(on: gpu, [a, b, c], cpu: { body($0[0], $0[1], $0[2]) }, gpu: { body($0[0], $0[1], $0[2]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test(arguments: [([4, 3, 20, 30], [4, 3, 30, 10]), ([4, 3, 20, 30], [30, 10]), ([4, 1, 20, 30], [1, 3, 30, 10]), ([20, 30], [5, 30, 10])])
        func batchedProductsMatchCPU(lhsShape: [Int], rhsShape: [Int]) {
            let a = Tensor<Float, CPU>(random(lhsShape, seed: 11), requiresGradient: true)
            let b = Tensor<Float, CPU>(random(rhsShape, seed: 12), requiresGradient: true)
            func body<D: DeviceType>(_ a: Tensor<Float, D>, _ b: Tensor<Float, D>) -> [Tensor<Float, D>] {
                let product = a.broadcastMatrixMultiplied(with: b)
                return [product] + (product * product).reduceSum().gradients(of: [a, b])
            }
            compare("bmm \(lhsShape) \(rhsShape)") { gpu in
                run(on: gpu, [a, b], cpu: { body($0[0], $0[1]) }, gpu: { body($0[0], $0[1]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test func accumulatedProductsAddToResult() {
            let (a, b, c) = (random([40, 30], seed: 10), random([30, 50], seed: 11), random([40, 50], seed: 12))
            func body<D: DeviceType>(_ a: Tensor<Float, D>, _ b: Tensor<Float, D>, _ c: Tensor<Float, D>) -> [Tensor<Float, D>] {
                var accumulator: Tensor<Float, D>? = c + 0
                Tensor.accumulateProduct(a, b, into: &accumulator)
                return [accumulator!]
            }
            compare("gemm beta") { gpu in
                run(on: gpu, [a, b, c], cpu: { body($0[0], $0[1], $0[2]) }, gpu: { body($0[0], $0[1], $0[2]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test(arguments: [([2, 3, 4, 5], [0, 2, 1, 3]), ([2, 3, 4, 5], [3, 2, 1, 0]), ([33, 65], [1, 0]), ([7, 33, 65], [0, 2, 1]), ([4, 5, 6], [1, 0, 2])])
        func permutationsMatchCPU(shape: [Int], arrangement: [Int]) {
            let a = Tensor<Float, CPU>(random(shape, seed: 13), requiresGradient: true)
            func body<D: DeviceType>(_ a: Tensor<Float, D>) -> [Tensor<Float, D>] {
                let permuted = a.permuted(to: arrangement)
                return [permuted, (permuted * permuted).reduceSum().gradients(of: [a])[0]]
            }
            compare("permute \(shape) \(arrangement)") { gpu in
                run(on: gpu, [a], cpu: { body($0[0]) }, gpu: { body($0[0]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test func subscriptsStacksAndReversalsMatchCPU() {
            let a = Tensor<Float, CPU>(random([6, 5, 4], seed: 14), requiresGradient: true)
            let b = Tensor<Float, CPU>(random([3, 5, 4], seed: 15), requiresGradient: true)
            func body<D: DeviceType>(_ a: Tensor<Float, D>, _ b: Tensor<Float, D>) -> [Tensor<Float, D>] {
                let row = a[2]
                let column = a[nil, 3]
                let ranged = a[1 ..< 4, nil, 1 ..< 3]
                let stacked = stack([a, b], along: 0)
                let stackedInner = stack([a[0 ..< 3], b], along: 1)
                let reversed = a.reversed()
                let loss = (row.reduceSum() + column.reduceSum() * 2 + (ranged * ranged).reduceSum() + (stacked * stacked).reduceSum() + stackedInner.reduceSum() + (reversed * a).reduceSum())
                let gradients = loss.gradients(of: [a, b])
                return [row, column, ranged, stacked, stackedInner, reversed] + gradients
            }
            compare("subscripts") { gpu in
                run(on: gpu, [a, b], cpu: { body($0[0], $0[1]) }, gpu: { body($0[0], $0[1]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test func gathersAndScattersMatchCPU() {
            let a = Tensor<Float, CPU>(random([37, 11], seed: 16, min: 0.1, max: 1), requiresGradient: true)
            var generator = WyHash(seed: 17)
            let labels = Tensor<Int32, CPU>((0 ..< 37).map { $0 % 5 == 0 ? -1 : Int32.random(in: 0 ..< 11, using: &generator) })
            let gpuLabels = Tensor<Int32, GPU>(labels)
            compare("gather") { gpu in
                if gpu {
                    let input = Tensor<Float, GPU>(a, requiresGradient: true)
                    let loss = categoricalCrossEntropy(expected: gpuLabels, actual: input.softmax())
                    return [loss, loss.gradients(of: [input])[0]].map { Tensor<Float, CPU>($0) }
                }
                let loss = categoricalCrossEntropy(expected: labels, actual: a.softmax())
                return [loss, loss.gradients(of: [a])[0]]
            }
        }

        @Test(arguments: [(1, 3, 8, 8, 4, 3, 1, 1), (2, 5, 13, 11, 7, 3, 2, 1), (3, 2, 9, 9, 3, 5, 1, 2)])
        func convolutionsMatchCPU(batch: Int, channels: Int, height: Int, width: Int, filters: Int, kernel: Int, stride: Int, padding: Int) {
            let input = Tensor<Float, CPU>(random([batch, channels, height, width], seed: 18), requiresGradient: true)
            let weights = Tensor<Float, CPU>(random([filters, channels, kernel, kernel], seed: 19), requiresGradient: true)
            let bias = Tensor<Float, CPU>(random([filters], seed: 20), requiresGradient: true)
            func body<D: DeviceType>(_ input: Tensor<Float, D>, _ weights: Tensor<Float, D>, _ bias: Tensor<Float, D>) -> [Tensor<Float, D>] {
                let output = input.convolved2d(filters: weights, bias: bias, padding: padding, stride: stride)
                let pooled = output.maxPooled2d(windowSize: 2)
                let loss = (pooled * pooled).reduceSum() + output.reduceMean()
                return [output, pooled] + loss.gradients(of: [input, weights, bias])
            }
            compare("conv", tolerance: 2e-3) { gpu in
                run(on: gpu, [input, weights, bias], cpu: { body($0[0], $0[1], $0[2]) }, gpu: { body($0[0], $0[1], $0[2]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test func bandsAndDiagonalsMatchCPU() {
            let a = random([9, 7], seed: 21)
            let square = random([6, 6], seed: 22)
            func body<D: DeviceType>(_ a: Tensor<Float, D>, _ square: Tensor<Float, D>) -> [Tensor<Float, D>] {
                [a.bandMatrix(belowDiagonal: 2, aboveDiagonal: 1), a.bandMatrix(belowDiagonal: nil, aboveDiagonal: 0), square.diagonalElements(), square.diagonalElements().diagonalMatrix(), Tensor(fillingDiagonalWith: 3, size: 5)]
            }
            compare("band") { gpu in
                run(on: gpu, [a, square], cpu: { body($0[0], $0[1]) }, gpu: { body($0[0], $0[1]).map { Tensor<Float, CPU>($0) } })
            }
        }

        @Test func integerAndDoubleTensorsWork() {
            let integers = Tensor<Int32, GPU>([1, -2, 3, 4, -5, 6], shape: [2, 3])
            #expect((integers + integers).elements == [2, -4, 6, 8, -10, 12])
            #expect((integers * integers).reduceSum(along: [1]).elements == [14, 77])
            #expect(integers.reduceMax(along: [0]).elements == [4, -2, 6])
            let doubles = Tensor<Double, GPU>([1, 2, 3, 4], shape: [2, 2], requiresGradient: true)
            let product = doubles.matrixMultiplied(with: doubles)
            #expect(product.elements == [7, 10, 15, 22])
            #expect(product.reduceSum().gradients(of: [doubles])[0].elements == [7, 11, 9, 13])
        }

        @Test func trainingStepsMatchCPU() {
            let input = random([64, 20], seed: 23)
            let targets = random([64, 3], seed: 24)
            let w1 = Tensor<Float, CPU>(random([20, 32], seed: 25), requiresGradient: true)
            let w2 = Tensor<Float, CPU>(random([32, 3], seed: 26), requiresGradient: true)
            func body<D: DeviceType>(_ values: [Tensor<Float, D>]) -> [Tensor<Float, D>] {
                var (w1, w2) = (values[2], values[3])
                var losses: [Tensor<Float, D>] = []
                for _ in 0 ..< 5 {
                    let hidden = values[0].matrixMultiplied(with: w1).rectifiedLinear()
                    let loss = meanSquaredError(expected: values[1], actual: hidden.matrixMultiplied(with: w2))
                    let gradients = loss.gradients(of: [w1, w2])
                    w1 = (w1 - 0.01 * gradients[0]).detached()
                    w2 = (w2 - 0.01 * gradients[1]).detached()
                    w1.requiresGradient = true
                    w2.requiresGradient = true
                    losses.append(loss)
                }
                return losses + [w1, w2]
            }
            compare("training", tolerance: 2e-3) { gpu in
                run(on: gpu, [input, targets, w1, w2], cpu: { body($0) }, gpu: { body($0).map { Tensor<Float, CPU>($0) } })
            }
        }
    }
}
#endif
