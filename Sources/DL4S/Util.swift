//
//  Util.swift
//  DL4S
//
//  Created by Palle Klewitz on 07.03.19.
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

#if os(Linux)
/// Stub that just calls the passed function on Linux.
/// Foundation on Linux does not provide `autoreleasepool` because Linux has no Objective-C runtime.
func autoreleasepool<Result>(_ function: () -> Result) -> Result {
    function()
}
#endif

extension Sequence {
    func count(where predicate: (Element) throws -> Bool) rethrows -> Int {
        try lazy.filter(predicate).count
    }
}

func shapeForBroadcastedOperands(_ lhs: [Int], _ rhs: [Int]) -> [Int] {
    let dim = Swift.max(lhs.count, rhs.count)
    let pLhs = Array(repeating: 1, count: dim - lhs.count) + lhs
    let pRhs = Array(repeating: 1, count: dim - rhs.count) + rhs
    return zip(pLhs, pRhs).map(Swift.max)
}

/// Iteration over the indices of a shape, with running offsets in two memory layouts instead of index arrays.
enum StridedIteration {
    /// Calls `body` for every index of `shape` in row-major order, with the offset of the index in two layouts with the given strides.
    ///
    /// An empty shape has one index, the scalar.
    @inline(__always)
    static func forEachOffset(shape: [Int], strides first: [Int], _ second: [Int], _ body: (_ first: Int, _ second: Int) -> Void) {
        let dim = shape.count
        guard dim > 0 else {
            body(0, 0)
            return
        }
        let count = shape.reduce(1, *)
        guard count > 0 else {
            return
        }
        withUnsafeTemporaryAllocation(of: Int.self, capacity: dim) { counters in
            counters.initialize(repeating: 0)
            var (firstOffset, secondOffset) = (0, 0)
            for _ in 0 ..< count {
                body(firstOffset, secondOffset)
                var axis = dim &- 1
                while axis >= 0 {
                    counters[axis] &+= 1
                    firstOffset &+= first[axis]
                    secondOffset &+= second[axis]
                    if counters[axis] < shape[axis] {
                        break
                    }
                    firstOffset &-= first[axis] &* shape[axis]
                    secondOffset &-= second[axis] &* shape[axis]
                    counters[axis] = 0
                    axis &-= 1
                }
            }
        }
    }

    /// Describes the permutation of a contiguous source as a copy from strided source positions into a contiguous destination.
    ///
    /// Axes of size 1 are dropped, and neighboring destination axes that are also neighbors in the source are merged.
    /// - Parameters:
    ///   - sourceShape: Shape of the source
    ///   - arrangement: Destination axis of every source axis
    /// - Returns: The shape of the destination after merging, at least one axis, and the source stride of every axis.
    static func permutationLayout(sourceShape: [Int], arrangement: [Int]) -> (shape: [Int], sourceStrides: [Int]) {
        let sourceStrides = MemoryOps.strides(from: sourceShape)
        var shape = [Int](repeating: 1, count: sourceShape.count)
        var strides = [Int](repeating: 0, count: sourceShape.count)
        for axis in sourceShape.indices {
            shape[arrangement[axis]] = sourceShape[axis]
            strides[arrangement[axis]] = sourceStrides[axis]
        }
        // The destination is contiguous, so axes merge when they are also contiguous in the source.
        let merged = mergingAxes(shape: shape, strides: [strides])
        return merged.shape.isEmpty ? ([1], [1]) : (merged.shape, merged.strides[0])
    }

    /// Merges neighboring axes that are contiguous in every layout, and drops the axes of size 1.
    ///
    /// An axis and the axis after it are contiguous in a layout when the stride of the first axis is the stride of the second
    /// axis times its size. The merged axes visit the same offsets in the same order.
    /// - Parameters:
    ///   - shape: Shape of the iteration
    ///   - strides: Strides of every layout for the shape
    /// - Returns: The merged shape, and the strides of every layout for it
    static func mergingAxes(shape: [Int], strides: [[Int]]) -> (shape: [Int], strides: [[Int]]) {
        var mergedShape: [Int] = []
        var mergedStrides = [[Int]](repeating: [], count: strides.count)
        for axis in shape.indices where shape[axis] != 1 {
            let isContiguous = !mergedShape.isEmpty && strides.indices.allSatisfy { layout in
                mergedStrides[layout][mergedShape.count - 1] == strides[layout][axis] * shape[axis]
            }
            if isContiguous {
                mergedShape[mergedShape.count - 1] *= shape[axis]
                for layout in strides.indices {
                    mergedStrides[layout][mergedShape.count - 1] = strides[layout][axis]
                }
            } else {
                mergedShape.append(shape[axis])
                for layout in strides.indices {
                    mergedStrides[layout].append(strides[layout][axis])
                }
            }
        }
        return (mergedShape, mergedStrides)
    }
}

prefix func ! <Parameters>(predicate: @escaping (Parameters) -> Bool) -> (Parameters) -> Bool {
    { params in
        !predicate(params)
    }
}

public struct ProgressBar<UserInfo> {
    public let totalUnitCount: Int
    public private(set) var currentUnitCount: Int
    public let formatUserInfo: (UserInfo) -> String
    public let label: String
    private var startTime = Date()

    public init(totalUnitCount: Int, formatUserInfo: @escaping (UserInfo) -> String, label: String) {
        self.totalUnitCount = totalUnitCount
        self.formatUserInfo = formatUserInfo
        self.label = label
        currentUnitCount = 0
    }

    private static func formatRemainingTime(_ interval: TimeInterval) -> String {
        let remaining = Duration.seconds(interval).formatted(
            .units(allowed: [.days, .hours, .minutes, .seconds], width: .narrow),
        )
        return "About \(remaining) remaining"
    }

    public mutating func next(userInfo: UserInfo) {
        currentUnitCount += 1

        let interval = Date().timeIntervalSince(startTime)
        let perUnitDuration = interval / Double(currentUnitCount)
        let remainingDuration = perUnitDuration * Double(totalUnitCount - currentUnitCount)
        let remainingString = Self.formatRemainingTime(remainingDuration)

        let filled = String(repeating: "#", count: currentUnitCount * 30 / totalUnitCount)
        let empty = String(repeating: " ", count: 30 - (currentUnitCount * 30 / totalUnitCount))
        print("\r\u{1b}[K\(label) [\(filled)\(empty)] (\(currentUnitCount)/\(totalUnitCount) - \(remainingString)) \(formatUserInfo(userInfo))", terminator: "")
        fflush(nil)
    }

    public mutating func complete() {
        currentUnitCount = totalUnitCount
        print("\r\u{1b}[K\(label): Done.")
    }
}

public struct Progress<Element>: Sequence {
    private struct ProgressIterator: IteratorProtocol {
        var baseIterator: AnyIterator<Element>
        let totalUnitCount: Int
        var currentCount: Int
        let label: String?
        let unit: String?

        mutating func next() -> Element? {
            if let next = baseIterator.next() {
                currentCount += 1
                print(completed: currentCount, total: totalUnitCount)
                return next
            } else {
                printCompleted()
                return nil
            }
        }

        func print(completed: Int, total: Int) {
            let filled = String(repeating: "#", count: currentCount * 30 / totalUnitCount)
            let empty = String(repeating: " ", count: 30 - (currentCount * 30 / totalUnitCount))

            let label = label.map { "\($0) " } ?? ""
            let unitString = unit.map { "\($0) " } ?? ""
            let userInfo = "(\(unitString)\(currentCount)/\(totalUnitCount))"

            Swift.print("\r\033[K\(label)[\(filled)\(empty)] \(userInfo)", terminator: "")
            fflush(nil)
        }

        func printCompleted() {
            Swift.print("\r\033[K", terminator: "")
            if let label {
                Swift.print("\(label): ", terminator: "")
            }
            Swift.print("Done.")
        }
    }

    private let base: AnySequence<Element>
    public var label: String?
    public var unit: String?

    public init<S: Sequence>(_ sequence: S, label: String? = nil, unit: String? = nil) where S.Element == Element {
        base = AnySequence(sequence)
        self.label = label
        self.unit = unit
    }

    public consuming func makeIterator() -> AnyIterator<Element> {
        let baseIterator = base.makeIterator()
        let progressIterator = ProgressIterator(
            baseIterator: baseIterator,
            totalUnitCount: base.underestimatedCount,
            currentCount: 0,
            label: label,
            unit: unit,
        )
        return AnyIterator(progressIterator)
    }
}

public extension Collection {
    // @_specialize(where Self == Array<Int>)
    func dropLast(while predicate: (Element) throws -> Bool) rethrows -> SubSequence {
        var index = index(endIndex, offsetBy: -1)

        while index > startIndex {
            if try predicate(self[index]) {
                index = self.index(index, offsetBy: -1)
            } else {
                return self[...index]
            }
        }

        return self[..<index]
    }

    // @_specialize(where Self == Array<Int>)
    func suffix(while predicate: (Element) throws -> Bool) rethrows -> SubSequence {
        var index = endIndex

        while index > startIndex {
            let nextIndex = self.index(index, offsetBy: -1)
            if try predicate(self[nextIndex]) {
                index = nextIndex
            } else {
                return self[index...]
            }
        }

        return self[...]
    }
}

public extension Sequence {
    @inline(__always)
    func suffix(while predicate: (Element) throws -> Bool) rethrows -> ArraySlice<Element> {
        try Array(self).suffix(while: predicate)
    }

    @inline(__always)
    func dropLast(while predicate: (Element) throws -> Bool) rethrows -> ArraySlice<Element> {
        try Array(self).dropLast(while: predicate)
    }
}

/// Shapes of reductions and broadcasts.
enum ShapeUtil {
    /// The shape without the given axes.
    static func reducedShape(of shape: [Int], along axes: [Int]) -> [Int] {
        shape.indices.filter { !axes.contains($0) }.map { shape[$0] }
    }

    /// The shape with the size 1 for every one of the given axes.
    static func keptShape(of shape: [Int], along axes: [Int]) -> [Int] {
        shape.indices.map { axes.contains($0) ? 1 : shape[$0] }
    }

    /// Number of elements that a reduction along the given axes combines into one.
    static func elementCount(of shape: [Int], along axes: [Int]) -> Int {
        axes.map { shape[$0] }.reduce(1, *)
    }

    /// Shape of the product of the matrices of two operands, with broadcasting along all axes except the last two.
    static func batchedProductShape(_ lhs: [Int], _ rhs: [Int], lhsTransposed: Bool, rhsTransposed: Bool) -> [Int] {
        let dim = Swift.max(lhs.count, rhs.count)
        let batchShape = shapeForBroadcastedOperands(
            Array(repeating: 1, count: dim - lhs.count) + lhs.dropLast(2),
            Array(repeating: 1, count: dim - rhs.count) + rhs.dropLast(2),
        )
        return batchShape + [lhs[lhs.count - (lhsTransposed ? 1 : 2)], rhs[rhs.count - (rhsTransposed ? 2 : 1)]]
    }

    /// Whether a shape is broadcastable to another shape.
    static func broadcasts(_ shape: [Int], to target: [Int]) -> Bool {
        shape.count <= target.count && zip(shape.reversed(), target.reversed()).allSatisfy { $0 == $1 || $0 == 1 }
    }

    /// Strides of the axes of a contiguous shape, with 0 for the axes with one element, along which the shape broadcasts.
    static func broadcastStrides(_ shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 0, count: shape.count)
        var stride = 1
        for axis in shape.indices.reversed() {
            strides[axis] = shape[axis] == 1 ? 0 : stride
            stride *= shape[axis]
        }
        return strides
    }

    /// Axes of `target` that broadcasting expands from `shape`.
    static func broadcastAxes(from shape: [Int], to target: [Int]) -> [Int] {
        let padded = Array(repeating: 1, count: target.count - shape.count) + shape
        return target.indices.filter { padded[$0] == 1 && target[$0] > 1 }
    }
}

public enum ConvUtil {
    public static func outputShape(for inputShape: [Int], kernelCount: Int, kernelWidth: Int, kernelHeight: Int, stride: Int, padding: Int) -> [Int] {
        [
            kernelCount,
            outputSize(inputSize: inputShape[1], kernelSize: kernelHeight, padding: padding, stride: stride),
            outputSize(inputSize: inputShape[2], kernelSize: kernelWidth, padding: padding, stride: stride),
        ]
    }

    /// Number of positions of a window along an axis of a convolution or of pooling.
    @inline(__always)
    static func outputSize(inputSize: Int, kernelSize: Int, padding: Int, stride: Int) -> Int {
        (inputSize + 2 * padding - kernelSize) / stride + 1
    }

    /// Size of the result of a transposed convolution along an axis.
    @inline(__always)
    static func transposedOutputSize(inputSize: Int, kernelSize: Int, inset: Int, stride: Int) -> Int {
        (inputSize - 1) * stride - 2 * inset + kernelSize
    }
}

struct File: Sequence {
    struct LineIterator: IteratorProtocol {
        private var buffer: Data?
        private let handle: FileHandle?
        private var isCompleted = false

        init(handle: FileHandle?) {
            self.handle = handle
            buffer = nil
        }

        mutating func next() -> String? {
            autoreleasepool { () -> String? in
                guard let handle, !isCompleted else {
                    return nil
                }

                let chunkSize = 4096

                if let buffer, let index = buffer.firstIndex(of: Character("\n").asciiValue!) {
                    let line = String(data: buffer.prefix(upTo: index), encoding: .utf8)
                    self.buffer = Data(buffer.dropFirst(index + 1)) // creating a copy resets the indexing, otherwise index points to the wrong position
                    return line
                } else {
                    let nextChunk = handle.readData(ofLength: chunkSize)

                    let buffer = (buffer ?? Data()) + nextChunk
                    self.buffer = buffer

                    if nextChunk.count == 0 {
                        isCompleted = true
                        return String(data: buffer, encoding: .utf8)
                    }

                    return next()
                }
            }
        }
    }

    let url: URL

    consuming func makeIterator() -> LineIterator {
        LineIterator(handle: try? FileHandle(forReadingFrom: url))
    }
}
