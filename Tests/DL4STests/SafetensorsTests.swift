//
//  SafetensorsTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 23.09.26.
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

import DL4S
import Foundation
import Testing

/// A model with weights, frozen tensors, a sequence, an array of layers, a scalar, and an empty tensor.
@Layer
struct CheckpointModel<Element: RandomizableType> {
    typealias Device = CPU

    var encoder: Sequential<Dense<Element, CPU>, Relu<Element, CPU>, Dense<Element, CPU>>
    @Frozen var statistics: Tensor<Element, CPU>
    var heads: [Dense<Element, CPU>]
    var scale: Tensor<Element, CPU>
    var empty: Tensor<Element, CPU>

    init(seed: UInt64) {
        var generator = WyHash(seed: seed)
        encoder = Sequential {
            Dense<Element, CPU>(inputSize: 4, outputSize: 8, using: &generator)
            Relu<Element, CPU>()
            Dense<Element, CPU>(inputSize: 8, outputSize: 3, using: &generator)
        }
        statistics = Tensor(uniformlyDistributedWithShape: [2, 3], min: -1, max: 1, using: &generator)
        heads = [
            Dense(inputSize: 3, outputSize: 2, using: &generator),
            Dense(inputSize: 3, outputSize: 5, using: &generator),
        ]
        scale = Tensor(uniformlyDistributedWithShape: [], requiresGradient: true, using: &generator)
        empty = Tensor([], shape: [0])
    }

    func callAsFunction(_ inputs: Tensor<Element, CPU>) -> Tensor<Element, CPU> {
        heads[0](encoder(inputs)) * scale
    }
}

/// A directory that is deleted when the value is released.
private final class TemporaryDirectory {
    let url = FileManager.default.temporaryDirectory.appending(path: "DL4S-Safetensors-\(UUID().uuidString)", directoryHint: .isDirectory)

    init() throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    deinit {
        try? FileManager.default.removeItem(at: url)
    }
}

/// A tensor entry of a JSON header, as a test reads it with `JSONSerialization`.
private struct HeaderEntry: Equatable {
    var dtype: String
    var shape: [Int]
    var begin: Int
    var end: Int
}

/// A safetensors file, split into its parts as the format specification describes them.
private struct ParsedFile {
    var headerLength: Int
    var headerBytes: Data
    var entries: [String: HeaderEntry]
    var metadata: [String: String]?
    var dataSection: Data

    init(_ data: Data) throws {
        let bytes = [UInt8](data)
        try #require(bytes.count >= 8)
        headerLength = bytes[0 ..< 8].enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        try #require(8 + headerLength <= bytes.count)
        headerBytes = Data(bytes[8 ..< 8 + headerLength])
        dataSection = Data(bytes[(8 + headerLength)...])

        let object = try #require(JSONSerialization.jsonObject(with: headerBytes) as? [String: Any])
        metadata = object["__metadata__"] as? [String: String]
        entries = [:]
        for (key, value) in object where key != "__metadata__" {
            let entry = try #require(value as? [String: Any])
            let offsets = try #require(entry["data_offsets"] as? [Int])
            try #require(offsets.count == 2)
            entries[key] = try HeaderEntry(
                dtype: #require(entry["dtype"] as? String),
                shape: #require(entry["shape"] as? [Int]),
                begin: offsets[0],
                end: offsets[1],
            )
        }
    }
}

/// Builds a safetensors file from a JSON header and a data section, for files that the encoder does not write.
private func makeFile(header: String, data: [UInt8]) -> Data {
    var json = Data(header.utf8)
    while (8 + json.count) % 8 != 0 {
        json.append(UInt8(ascii: " "))
    }
    var file = Data()
    withUnsafeBytes(of: UInt64(json.count).littleEndian) { file.append(contentsOf: $0) }
    file.append(json)
    file.append(contentsOf: data)
    return file
}

/// Returns the raw bytes of every tensor of a layer, keyed by path.
private func tensorBytes<Layer: LayerType>(of layer: Layer) -> [String: [UInt8]] {
    var result: [String: [UInt8]] = [:]
    var copy = layer
    var visitor = TensorVisitor<Layer.Parameter, Layer.Device>(tensors: { tensor, _, path in
        result[path.description] = tensor.elements.withUnsafeBytes { Array($0) }
    })
    copy.visitTensors(&visitor)
    return result
}

/// Returns the elements of every tensor of a layer, keyed by path.
private func tensorElements<Layer: LayerType>(of layer: Layer) -> [String: [Layer.Parameter]] {
    var result: [String: [Layer.Parameter]] = [:]
    var copy = layer
    var visitor = TensorVisitor<Layer.Parameter, Layer.Device>(tensors: { tensor, _, path in
        result[path.description] = tensor.elements
    })
    copy.visitTensors(&visitor)
    return result
}

/// A model whose frozen tensor holds values that a conversion would change: NaN, signed zero, infinities, and a subnormal.
private func modelWithSpecialValues(seed: UInt64) -> CheckpointModel<Float> {
    var model = CheckpointModel<Float>(seed: seed)
    let values: [Float] = [.nan, -0.0, .infinity, -.infinity, .leastNonzeroMagnitude, 1 / 3]
    model.statistics = Tensor(values, shape: [2, 3])
    return model
}

struct SafetensorsTests {
    private let expectedKeys: Set<String> = [
        "encoder.0.weights", "encoder.0.bias", "encoder.2.weights", "encoder.2.bias",
        "statistics",
        "heads.0.weights", "heads.0.bias", "heads.1.weights", "heads.1.bias",
        "scale", "empty",
    ]

    // MARK: Round trips

    @Test func testDataRoundTripIsBitExact() throws {
        let model = modelWithSpecialValues(seed: 1)
        let data = try SafetensorsEncoder().encode(model)

        var restored = CheckpointModel<Float>(seed: 2)
        #expect(tensorBytes(of: restored) != tensorBytes(of: model))
        try SafetensorsDecoder().load(into: &restored, from: data)

        #expect(tensorBytes(of: restored) == tensorBytes(of: model))
        #expect(restored.weightPaths == model.weightPaths)
    }

    @Test func testDoubleRoundTripIsBitExact() throws {
        let model = CheckpointModel<Double>(seed: 1)
        var restored = CheckpointModel<Double>(seed: 2)
        try SafetensorsDecoder().load(into: &restored, from: SafetensorsEncoder().encode(model))
        #expect(tensorBytes(of: restored) == tensorBytes(of: model))
    }

    @Test func testFileRoundTripIsBitExact() throws {
        let directory = try TemporaryDirectory()
        let url = directory.url.appending(path: "model.safetensors")
        let model = modelWithSpecialValues(seed: 1)
        try SafetensorsEncoder().encode(model, to: url)

        var restored = CheckpointModel<Float>(seed: 2)
        try SafetensorsDecoder().load(into: &restored, from: url)
        #expect(tensorBytes(of: restored) == tensorBytes(of: model))

        // The file and the in-memory encoding are the same.
        #expect(try Data(contentsOf: url) == SafetensorsEncoder().encode(model))
    }

    @Test(arguments: [4, 64, 200, 1 << 20])
    func testShardedRoundTripByMaximumBytesIsBitExact(maximumBytes: Int) throws {
        let directory = try TemporaryDirectory()
        let model = modelWithSpecialValues(seed: 1)
        try SafetensorsEncoder(options: .init(sharding: .maximumBytes(maximumBytes))).encode(model, to: directory.url)

        for source in [directory.url, directory.url.appending(path: "model.safetensors.index.json")] {
            var restored = CheckpointModel<Float>(seed: 2)
            try SafetensorsDecoder().load(into: &restored, from: source)
            #expect(tensorBytes(of: restored) == tensorBytes(of: model))
        }
    }

    @Test func testShardedRoundTripByShardClosureIsBitExact() throws {
        let directory = try TemporaryDirectory()
        let model = modelWithSpecialValues(seed: 1)
        let sharding = SafetensorsEncoder.Sharding.byShard { path in
            "\(path.segments[0]).safetensors"
        }
        try SafetensorsEncoder(options: .init(sharding: sharding)).encode(model, to: directory.url)

        let files = try Set(FileManager.default.contentsOfDirectory(atPath: directory.url.path))
        #expect(files == [
            "encoder.safetensors", "statistics.safetensors", "heads.safetensors", "scale.safetensors", "empty.safetensors",
            "model.safetensors.index.json",
        ])

        var restored = CheckpointModel<Float>(seed: 2)
        try SafetensorsDecoder().load(into: &restored, from: directory.url)
        #expect(tensorBytes(of: restored) == tensorBytes(of: model))
    }

    @Test func testLoadKeepsTrainability() throws {
        var model = CheckpointModel<Float>(seed: 1)
        model.heads[1].freeze()
        var restored = model
        try SafetensorsDecoder().load(into: &restored, from: SafetensorsEncoder().encode(CheckpointModel<Float>(seed: 3)))

        #expect(restored.weightPaths == model.weightPaths)
        #expect(restored.parameters.allSatisfy { $0.requiresGradient })
        // The copy that shares storage with the restored model keeps its values.
        #expect(tensorBytes(of: model) == tensorBytes(of: CheckpointModel<Float>(seed: 1)))
    }

    // MARK: File layout

    @Test func testFileLayoutMatchesSpecification() throws {
        let model = modelWithSpecialValues(seed: 1)
        let data = try SafetensorsEncoder(options: .init(metadata: ["format": "dl4s", "note": "a/b"])).encode(model)
        let file = try ParsedFile(data)

        // The data section starts at a multiple of 8 bytes. The header is padded with spaces.
        #expect((8 + file.headerLength) % 8 == 0)
        let json = try #require(String(data: file.headerBytes, encoding: .utf8))
        #expect(json.first == "{")
        let objectEnd = try #require(json.lastIndex(of: "}"))
        let padding = json[json.index(after: objectEnd)...]
        #expect(padding.allSatisfy { $0 == " " })
        #expect(padding.count < 8)

        #expect(file.metadata == ["format": "dl4s", "note": "a/b"])
        #expect(Set(file.entries.keys) == expectedKeys)

        // The tensors fill the data section from the start, without gaps.
        let sorted = file.entries.values.sorted { ($0.begin, $0.end) < ($1.begin, $1.end) }
        var end = 0
        for entry in sorted {
            #expect(entry.begin == end)
            end = entry.end
        }
        #expect(end == file.dataSection.count)

        // Every tensor is F32 in row-major, little-endian order.
        let expected = tensorElements(of: model)
        for (key, entry) in file.entries {
            let elements = try #require(expected[key])
            #expect(entry.dtype == "F32")
            #expect(entry.end - entry.begin == elements.count * 4)
            let values = file.dataSection[entry.begin ..< entry.end].withUnsafeBytes { bytes in
                (0 ..< elements.count).map { bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self) }
            }
            #expect(values == elements.map { $0.bitPattern.littleEndian })
        }

        #expect(file.entries["encoder.0.weights"]?.shape == [4, 8])
        #expect(file.entries["statistics"]?.shape == [2, 3])
        #expect(file.entries["scale"]?.shape == [])
        #expect(file.entries["empty"].map { $0.shape == [0] && $0.begin == $0.end } == true)
    }

    @Test func testEncoderWritesElementTypes() throws {
        let double = try ParsedFile(SafetensorsEncoder().encode(Dense<Double, CPU>(inputSize: 2, outputSize: 3)))
        #expect(double.entries["weights"] == HeaderEntry(dtype: "F64", shape: [2, 3], begin: 0, end: 48))
        #expect(double.entries["bias"] == HeaderEntry(dtype: "F64", shape: [3], begin: 48, end: 72))
        #expect(double.metadata == nil)

        let indices = IndexLayer(indices: Tensor([3, -1, 7], shape: [3]))
        let integer = try ParsedFile(SafetensorsEncoder().encode(indices))
        #expect(integer.entries["indices"] == HeaderEntry(dtype: "I32", shape: [3], begin: 0, end: 12))
        #expect([UInt8](integer.dataSection) == [3, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 7, 0, 0, 0])

        var restored = IndexLayer(indices: Tensor(repeating: 0, shape: [3]))
        try SafetensorsDecoder().load(into: &restored, from: SafetensorsEncoder().encode(indices))
        #expect(restored.indices.elements == [3, -1, 7])
    }

    @Test func testShardedLayoutMatchesSpecification() throws {
        let directory = try TemporaryDirectory()
        let model = modelWithSpecialValues(seed: 1)
        let maximumBytes = 200
        try SafetensorsEncoder(options: .init(metadata: ["format": "dl4s"], sharding: .maximumBytes(maximumBytes))).encode(model, to: directory.url)

        let indexData = try Data(contentsOf: directory.url.appending(path: "model.safetensors.index.json"))
        let index = try #require(JSONSerialization.jsonObject(with: indexData) as? [String: Any])
        let weightMap = try #require(index["weight_map"] as? [String: String])
        let totalSize = try #require((index["metadata"] as? [String: Any])?["total_size"] as? Int)
        #expect(Set(weightMap.keys) == expectedKeys)

        let shardNames = Set(weightMap.values).sorted()
        #expect(shardNames.count > 1)
        #expect(shardNames == (1 ... shardNames.count).map { String(format: "model-%05d-of-%05d.safetensors", $0, shardNames.count) })

        var bytesInShards = 0
        for name in shardNames {
            let file = try ParsedFile(Data(contentsOf: directory.url.appending(path: name)))
            #expect((8 + file.headerLength) % 8 == 0)
            #expect(file.metadata == ["format": "dl4s"])
            #expect(Set(file.entries.keys) == Set(weightMap.filter { $0.value == name }.keys))
            // A shard exceeds the limit only when it holds one tensor that is larger than the limit.
            #expect(file.dataSection.count <= maximumBytes || file.entries.count == 1)
            bytesInShards += file.dataSection.count
        }
        #expect(bytesInShards == totalSize)
        #expect(totalSize == tensorElements(of: model).values.reduce(0) { $0 + $1.count * 4 })
    }

    @Test func testHeaderReadsEntriesAndMetadata() throws {
        let directory = try TemporaryDirectory()
        let url = directory.url.appending(path: "model.safetensors")
        let model = Dense<Float, CPU>(inputSize: 2, outputSize: 3)
        try SafetensorsEncoder(options: .init(metadata: ["epoch": "3"])).encode(model, to: url)

        let header = try SafetensorsDecoder().header(at: url)
        #expect(header.metadata == ["epoch": "3"])
        #expect(header.entries.map(\.name) == ["weights", "bias"])
        #expect(header["bias"]?.dtype == .f32)
        #expect(header["bias"]?.shape == [3])
        #expect(header["bias"]?.dataOffsets == 24 ..< 36)
        #expect(header.dataSectionOffset % 8 == 0)
        #expect(header["missing"] == nil)
    }

    // MARK: Matching

    @Test func testLoadConvertsElementTypes() throws {
        let source = Dense<Double, CPU>(inputSize: 3, outputSize: 4)
        var target = Dense<Float, CPU>(inputSize: 3, outputSize: 4)
        try SafetensorsDecoder().load(into: &target, from: SafetensorsEncoder().encode(source))
        #expect(target.weights.elements == source.weights.elements.map { Float($0) })
        #expect(target.bias.elements == source.bias.elements.map { Float($0) })

        var widened = Dense<Double, CPU>(inputSize: 3, outputSize: 4)
        try SafetensorsDecoder().load(into: &widened, from: SafetensorsEncoder().encode(target))
        #expect(widened.weights.elements == target.weights.elements.map { Double($0) })

        let integers = makeFile(header: #"{"bias":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#, data: [7, 0, 0, 0, 0xFE, 0xFF, 0xFF, 0xFF])
        var fromIntegers = Tensor<Float, CPU>(repeating: 0, shape: [2])
        var layer = BiasLayer(bias: fromIntegers)
        try SafetensorsDecoder().load(into: &layer, from: integers)
        fromIntegers = layer.bias
        #expect(fromIntegers.elements == [7, -2])
    }

    @Test func testMissingTensorThrowsAndKeepsModel() throws {
        let data = try SafetensorsEncoder().encode(Dense<Float, CPU>(inputSize: 4, outputSize: 8))
        let model = CheckpointModel<Float>(seed: 1)
        var target = model

        let error = try #require(throws: SafetensorsError.self) {
            try SafetensorsDecoder(options: .init(allowsUnusedTensors: true)).load(into: &target, from: data)
        }
        #expect(error.kind == .missingTensor)
        #expect(error.key == "encoder.0.weights")
        #expect(tensorBytes(of: target) == tensorBytes(of: model))
    }

    @Test func testAllowsMissingTensorsKeepsCurrentValues() throws {
        let source = CheckpointModel<Float>(seed: 1)
        let data = try SafetensorsEncoder().encode(source.heads[1])
        var target = CheckpointModel<Float>(seed: 2)
        let before = tensorBytes(of: target)

        let decoder = SafetensorsDecoder(options: .init(naming: { path in
            // Maps heads.1.weights to weights, so that only the second head matches the file.
            path.segments.starts(with: [.name("heads"), .index(1)]) ? TensorPath(Array(path.segments.dropFirst(2))).description : "unmatched.\(path)"
        }, allowsMissingTensors: true))
        try decoder.load(into: &target, from: data)

        let after = tensorBytes(of: target)
        #expect(after["heads.1.weights"] == tensorBytes(of: source)["heads.1.weights"])
        #expect(after["heads.1.bias"] == tensorBytes(of: source)["heads.1.bias"])
        #expect(after.filter { !$0.key.hasPrefix("heads.1.") } == before.filter { !$0.key.hasPrefix("heads.1.") })
    }

    @Test func testUnusedTensorThrows() throws {
        let data = try SafetensorsEncoder().encode(CheckpointModel<Float>(seed: 1))
        var target = Dense<Float, CPU>(inputSize: 4, outputSize: 8)
        let decoder = SafetensorsDecoder(options: .init(naming: { "encoder.0.\($0)" }))

        let error = try #require(throws: SafetensorsError.self) {
            try decoder.load(into: &target, from: data)
        }
        #expect(error.kind == .unusedTensor)

        let lenient = SafetensorsDecoder(options: .init(naming: { "encoder.0.\($0)" }, allowsUnusedTensors: true))
        try lenient.load(into: &target, from: data)
        #expect(target.weights.elements == CheckpointModel<Float>(seed: 1).encoder.first.weights.elements)
    }

    @Test func testShapeMismatchThrows() throws {
        let data = try SafetensorsEncoder().encode(Dense<Float, CPU>(inputSize: 4, outputSize: 8))
        var target = Dense<Float, CPU>(inputSize: 8, outputSize: 4)
        let error = try #require(throws: SafetensorsError.self) {
            try SafetensorsDecoder().load(into: &target, from: data)
        }
        #expect(error.kind == .shapeMismatch(file: [4, 8], model: [8, 4]))
        #expect(error.key == "weights")
    }

    @Test func testUnsupportedDTypeThrows() throws {
        let data = makeFile(header: #"{"bias":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#, data: [0, 0x3C, 0, 0x40])
        var layer = BiasLayer(bias: Tensor(repeating: 0, shape: [2]))
        let error = try #require(throws: SafetensorsError.self) {
            try SafetensorsDecoder().load(into: &layer, from: data)
        }
        #expect(error.kind == .unsupportedDType("F16"))
        #expect(error.key == "bias")
    }

    @Test func testDuplicateKeysThrow() throws {
        let encoder = SafetensorsEncoder(options: .init(naming: { _ in "same" }))
        let error = try #require(throws: SafetensorsError.self) {
            try encoder.encode(Dense<Float, CPU>(inputSize: 2, outputSize: 2))
        }
        #expect(error.kind == .duplicateKey)
        #expect(error.key == "same")
    }

    // MARK: Layout adoption

    @Test func testLoadCreatesSublayersFromTheLayout() throws {
        var generator = WyHash(seed: 4)
        let source = DynamicStack(blocks: [
            Dense(inputSize: 3, outputSize: 5, using: &generator),
            Dense(inputSize: 5, outputSize: 2, using: &generator),
            Dense(inputSize: 2, outputSize: 1, using: &generator),
        ])

        var restored = DynamicStack(blocks: [])
        try SafetensorsDecoder().load(into: &restored, from: SafetensorsEncoder().encode(source))

        #expect(restored.blocks.count == 3)
        #expect(tensorBytes(of: restored) == tensorBytes(of: source))
        // The adopted structure is what the layout describes.
        #expect(restored.tensorLayout == source.tensorLayout)
    }

    @Test func testFailedLoadDoesNotAdoptTheLayout() throws {
        var generator = WyHash(seed: 4)
        let source = DynamicStack(blocks: [Dense(inputSize: 3, outputSize: 5, using: &generator)])
        var data = try SafetensorsEncoder().encode(source)
        data.removeLast()

        var restored = DynamicStack(blocks: [])
        #expect(throws: SafetensorsError.self) {
            try SafetensorsDecoder().load(into: &restored, from: data)
        }
        #expect(restored.blocks.isEmpty)
    }

    @Test func testNamingClosureGivesAnEmptyLayout() throws {
        let source = DynamicStack(blocks: [Dense(inputSize: 3, outputSize: 5)])
        let data = try SafetensorsEncoder(options: .init(naming: { "model.\($0)" })).encode(source)

        var restored = DynamicStack(blocks: [])
        let decoder = SafetensorsDecoder(options: .init(naming: { "model.\($0)" }))
        let error = try #require(throws: SafetensorsError.self) {
            try decoder.load(into: &restored, from: data)
        }
        #expect(error.kind == .unusedTensor)
        #expect(restored.blocks.isEmpty)
    }

    @Test func testTensorArraysAreReportedByPosition() throws {
        var layer = TensorList()
        layer.tensors = (0 ..< 12).map { Tensor(repeating: Float($0), shape: [$0 % 3 + 1], requiresGradient: true) }

        let paths = layer.weightPaths.map(\.description)
        #expect(paths == (0 ..< 12).map { "tensors.\($0)" })
        // Layout entries sort indices by value, so "tensors.10" follows "tensors.9".
        #expect(layer.tensorLayout.entries.map(\.path.description) == paths)

        var restored = TensorList()
        restored.tensors = layer.tensors.map { Tensor(repeating: 0, shape: $0.shape) }
        try SafetensorsDecoder().load(into: &restored, from: SafetensorsEncoder().encode(layer))
        #expect(restored.tensors.map(\.elements) == layer.tensors.map(\.elements))
    }

    @Test func testTensorLayoutScopesPaths() {
        let layout = TensorLayout([
            "encoder.0.weights": [2, 3],
            "encoder.0.bias": [3],
            "encoder.1.weights": [3, 1],
            "head": [1],
        ])
        #expect(layout["encoder.0.bias"] == [3])
        #expect(layout["missing"] == nil)
        #expect(layout.children(of: "encoder").map(\.path.description) == ["0.bias", "0.weights", "1.weights"])
        #expect(layout.children(of: "head").isEmpty)

        let scoped = layout.scoped(to: "encoder.0")
        #expect(scoped["weights"] == [2, 3])
        #expect(scoped.entries == [
            TensorLayout.Entry(path: "bias", shape: [3]),
            TensorLayout.Entry(path: "weights", shape: [2, 3]),
        ])
        #expect(scoped == TensorLayout(["bias": [3], "weights": [2, 3]]))
    }

    // MARK: Invalid files

    @Test(arguments: [
        ("truncated length", Data([1, 0, 0])),
        ("header longer than file", makeFile(header: "{}", data: []).prefix(9)),
        ("not an object", makeFile(header: "[]", data: [])),
        ("invalid JSON", makeFile(header: "{\"bias\":", data: [])),
        ("gap between tensors", makeFile(header: #"{"bias":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#, data: Array(repeating: 0, count: 12))),
        ("data section too long", makeFile(header: #"{"bias":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, data: Array(repeating: 0, count: 12))),
        ("shape does not match byte count", makeFile(header: #"{"bias":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#, data: Array(repeating: 0, count: 8))),
        ("negative dimension", makeFile(header: #"{"bias":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}}"#, data: Array(repeating: 0, count: 8))),
    ])
    func testMalformedFileThrows(description: String, data: Data) throws {
        var layer = BiasLayer(bias: Tensor(repeating: 1, shape: [2]))
        let error = try #require(throws: SafetensorsError.self, "\(description)") {
            try SafetensorsDecoder().load(into: &layer, from: data)
        }
        guard case .malformedHeader = error.kind else {
            Issue.record("\(description): expected a malformed header, got \(error)")
            return
        }
        #expect(layer.bias.elements == [1, 1])
    }

    @Test func testMissingShardThrows() throws {
        let directory = try TemporaryDirectory()
        try SafetensorsEncoder(options: .init(sharding: .maximumBytes(8))).encode(Dense<Float, CPU>(inputSize: 2, outputSize: 2), to: directory.url)
        try FileManager.default.removeItem(at: directory.url.appending(path: "model-00002-of-00002.safetensors"))

        var target = Dense<Float, CPU>(inputSize: 2, outputSize: 2)
        let error = try #require(throws: SafetensorsError.self) {
            try SafetensorsDecoder().load(into: &target, from: directory.url)
        }
        #expect(error.kind == .shardNotFound("model-00002-of-00002.safetensors"))
    }
}

/// A layer with one tensor named `bias`, for files that tests write by hand.
@Layer
private struct BiasLayer {
    typealias Element = Float
    typealias Device = CPU

    var bias: Tensor<Float, CPU>

    func callAsFunction(_ inputs: Tensor<Float, CPU>) -> Tensor<Float, CPU> {
        inputs + bias
    }
}

/// A layer with one `Int32` tensor.
@Layer
private struct IndexLayer {
    typealias Element = Int32
    typealias Device = CPU

    var indices: Tensor<Int32, CPU>

    func callAsFunction(_ inputs: Tensor<Int32, CPU>) -> Tensor<Int32, CPU> {
        inputs + indices
    }
}

/// A stack of dense layers whose count and sizes come from the layout of a checkpoint.
@Layer
private struct DynamicStack {
    typealias Element = Float
    typealias Device = CPU

    var blocks: [Dense<Float, CPU>]

    mutating func adoptLayout(_ layout: TensorLayout) {
        let count = Set(layout.children(of: "blocks").compactMap { entry -> Int? in
            guard case let .index(index) = entry.path.segments.first else {
                return nil
            }
            return index
        }).count
        blocks = (0 ..< count).map { index in
            let shape = layout[TensorPath("blocks.\(index).weights")] ?? [0, 0]
            return Dense(inputSize: shape[0], outputSize: shape[1])
        }
    }

    func callAsFunction(_ inputs: Tensor<Float, CPU>) -> Tensor<Float, CPU> {
        blocks.reduce(inputs) { $1($0) }
    }
}

/// A layer with an array of tensors.
@Layer
private struct TensorList {
    typealias Element = Float
    typealias Device = CPU

    var tensors: [Tensor<Float, CPU>] = []

    func callAsFunction(_ inputs: Tensor<Float, CPU>) -> Tensor<Float, CPU> {
        inputs
    }
}
