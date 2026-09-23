//
//  SafetensorsEncoder.swift
//  DL4S
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

import Foundation

/// Writes the tensors of a model to a safetensors file.
///
/// The encoder writes every tensor that the model reports in ``TensorContainer/visitTensors(_:)``, weights and frozen
/// tensors alike. The key of a tensor is its ``TensorPath``, for example `encoder.blocks.3.Wq` or `0.weights`.
/// Hyperparameters, such as strides or rates, are not written: they are part of the code that creates the model.
/// `Float` tensors are written as `F32`, `Double` tensors as `F64`, and `Int32` tensors as `I32`.
///
/// ```swift
/// try SafetensorsEncoder().encode(model, to: URL(filePath: "mnist.safetensors"))
///
/// let encoder = SafetensorsEncoder(options: .init(sharding: .maximumBytes(2 << 30)))
/// try encoder.encode(transformer, to: URL(filePath: "checkpoint/", directoryHint: .isDirectory))
/// ```
///
/// Load the file with ``SafetensorsDecoder``.
public struct SafetensorsEncoder: Sendable {
    /// Options that control the output of the encoder.
    public struct Options: Sendable {
        /// Values for the `__metadata__` entry of the header. With sharding, every shard gets the metadata.
        public var metadata: [String: String]

        /// How to split the tensors into shards. With `nil`, the encoder writes one file.
        public var sharding: Sharding?

        /// Maps the path of a tensor to its key in the file.
        public var naming: @Sendable (TensorPath) -> String

        /// Creates options.
        /// - Parameters:
        ///   - metadata: Values for the `__metadata__` entry of the header.
        ///   - sharding: How to split the tensors into shards. With `nil`, the encoder writes one file.
        ///   - naming: Maps the path of a tensor to its key in the file. The default is the text form of the path.
        public init(
            metadata: [String: String] = [:],
            sharding: Sharding? = nil,
            naming: @escaping @Sendable (TensorPath) -> String = { $0.description },
        ) {
            self.metadata = metadata
            self.sharding = sharding
            self.naming = naming
        }
    }

    /// How the encoder splits the tensors of a model into shard files.
    ///
    /// A sharded checkpoint is a directory with the shard files and an index file `model.safetensors.index.json`,
    /// as in the Hugging Face layout. The index file maps the key of every tensor to the file name of its shard.
    public enum Sharding: Sendable {
        /// Starts a new shard when the next tensor does not fit into the current shard.
        ///
        /// The limit counts the tensor bytes of a shard. A tensor that is larger than the limit gets a shard of its
        /// own. The shards are named `model-00001-of-00003.safetensors` and so on.
        case maximumBytes(Int)

        /// Puts every tensor into the shard with the file name that the closure returns for its path.
        case byShard(@Sendable (TensorPath) -> String)
    }

    /// The options of the encoder.
    public var options: Options

    /// Creates an encoder.
    /// - Parameter options: Options that control the output.
    public init(options: Options = Options()) {
        self.options = options
    }

    /// Returns the tensors of a model as the contents of one safetensors file.
    ///
    /// This method ignores ``Options/sharding``.
    /// - Parameter layer: The model to encode.
    /// - Returns: The contents of the file.
    /// - Throws: ``SafetensorsError`` when two tensors get the same key, or when the element type of the model is not supported.
    public func encode<Layer: TensorContainer>(_ layer: Layer) throws -> Data {
        let tensors = try collectTensors(of: layer)
        let header = try SafetensorsFormat.encodeHeader(entries: entries(for: tensors), metadata: options.metadata)
        var data = Data(capacity: header.count + tensors.reduce(0) { $0 + $1.byteCount })
        data.append(header)
        for tensor in tensors {
            data.append(tensor.bytes())
        }
        return data
    }

    /// Writes the tensors of a model to a file, or to a directory of shards.
    ///
    /// Without ``Options/sharding``, `url` is the file to write. With sharding, `url` is a directory. The encoder
    /// creates the directory if necessary and writes the shards and the index file into it. Files that exist
    /// are replaced.
    ///
    /// - Parameters:
    ///   - layer: The model to encode.
    ///   - url: The file or the directory to write.
    /// - Throws: ``SafetensorsError`` when two tensors get the same key, or when the element type of the model
    ///   is not supported. File system errors when a file cannot be written.
    public func encode<Layer: TensorContainer>(_ layer: Layer, to url: URL) throws {
        let tensors = try collectTensors(of: layer)
        guard let sharding = options.sharding else {
            try write(tensors, to: url)
            return
        }

        let shards = shards(of: tensors, by: sharding)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        var weightMap: [String: String] = [:]
        for shard in shards {
            try write(shard.tensors, to: url.appending(path: shard.fileName))
            for tensor in shard.tensors {
                weightMap[tensor.key] = shard.fileName
            }
        }

        let index = SafetensorsIndex(totalSize: tensors.reduce(0) { $0 + $1.byteCount }, weightMap: weightMap)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
        try encoder.encode(index).write(to: url.appending(path: SafetensorsFormat.indexFileName))
    }

    // MARK: Implementation

    private func collectTensors<Layer: TensorContainer>(of layer: Layer) throws(SafetensorsError) -> [EncodedTensor<Layer.Parameter, Layer.Device>] {
        let dtype = try SafetensorsDType.of(Layer.Parameter.self)
        var tensors: [EncodedTensor<Layer.Parameter, Layer.Device>] = []
        var copy = layer
        var visitor = TensorVisitor<Layer.Parameter, Layer.Device>(tensors: { [naming = options.naming] tensor, _, path in
            tensors.append(EncodedTensor(key: naming(path), path: path, dtype: dtype, tensor: tensor))
        })
        copy.visitTensors(&visitor)

        var keys: Set<String> = [SafetensorsFormat.metadataKey]
        for tensor in tensors where !keys.insert(tensor.key).inserted {
            throw SafetensorsError(.duplicateKey, key: tensor.key)
        }
        return tensors
    }

    /// Assigns contiguous byte ranges in traversal order.
    private func entries<Element, Device>(for tensors: [EncodedTensor<Element, Device>]) -> [SafetensorsHeader.Entry] {
        var offset = 0
        return tensors.map { tensor in
            defer {
                offset += tensor.byteCount
            }
            return SafetensorsHeader.Entry(name: tensor.key, dtype: tensor.dtype, shape: tensor.tensor.shape, dataOffsets: offset ..< offset + tensor.byteCount)
        }
    }

    private func shards<Element, Device>(of tensors: [EncodedTensor<Element, Device>], by sharding: Sharding) -> [Shard<Element, Device>] {
        switch sharding {
        case let .maximumBytes(maximumBytes):
            precondition(maximumBytes > 0, "The maximum shard size must be positive, but it is \(maximumBytes).")
            var groups: [[EncodedTensor<Element, Device>]] = []
            var currentBytes = 0
            for tensor in tensors {
                if groups.isEmpty || (!groups[groups.count - 1].isEmpty && currentBytes + tensor.byteCount > maximumBytes) {
                    groups.append([])
                    currentBytes = 0
                }
                groups[groups.count - 1].append(tensor)
                currentBytes += tensor.byteCount
            }
            return groups.enumerated().map { index, group in
                Shard(fileName: String(format: "model-%05d-of-%05d.safetensors", index + 1, groups.count), tensors: group)
            }

        case let .byShard(shardName):
            var shards: [Shard<Element, Device>] = []
            var shardIndices: [String: Int] = [:]
            for tensor in tensors {
                let fileName = shardName(tensor.path)
                if let index = shardIndices[fileName] {
                    shards[index].tensors.append(tensor)
                } else {
                    shardIndices[fileName] = shards.count
                    shards.append(Shard(fileName: fileName, tensors: [tensor]))
                }
            }
            return shards
        }
    }

    // Writes one tensor at a time, so that the transient memory stays at the size of the largest tensor.
    private func write<Element, Device>(_ tensors: [EncodedTensor<Element, Device>], to url: URL) throws {
        let header = try SafetensorsFormat.encodeHeader(entries: entries(for: tensors), metadata: options.metadata)
        // Creates the file, or truncates a file that exists.
        try header.write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        for tensor in tensors where tensor.byteCount > 0 {
            try handle.write(contentsOf: tensor.bytes())
        }
        try handle.close()
    }
}

/// A tensor of a model with its key in the file.
private struct EncodedTensor<Element: NumericType, Device: DeviceType> {
    var key: String
    var path: TensorPath
    var dtype: SafetensorsDType
    var tensor: Tensor<Element, Device>

    var byteCount: Int {
        tensor.count * MemoryLayout<Element>.stride
    }

    /// The elements of the tensor in row-major order.
    func bytes() -> Data {
        var data = Data(count: byteCount)
        guard byteCount > 0 else {
            return data
        }
        data.withUnsafeMutableBytes { bytes in
            Device.Memory.assign(from: tensor.values.values, to: bytes.bindMemory(to: Element.self), count: tensor.count)
        }
        return data
    }
}

private struct Shard<Element: NumericType, Device: DeviceType> {
    var fileName: String
    var tensors: [EncodedTensor<Element, Device>]
}
