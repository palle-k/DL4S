//
//  SafetensorsDecoder.swift
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

/// Loads the tensors of a safetensors file into a model.
///
/// The decoder populates weights of an existing model, converting data types as needed.
///
/// ```swift
/// var restored = makeModel()
/// try SafetensorsDecoder().load(into: &restored, from: URL(filePath: "mnist.safetensors"))
/// ```
///
/// When the files do not match the model, the decoder throws an error, leaving the model unchanged.
///
/// If paths in the file do not correspond to tensor names, a `naming` function may be provided to map tensor names between DL4S and keys in the file.
/// Extraneous tensors may be ignored by setting `options.allowUnusedTensors` to true; `allowMissingTensors` leaves tensors missing in the .safetensors
/// file unchanged.
public struct SafetensorsDecoder: Sendable {
    /// Options that control how the decoder matches file entries to model tensors.
    public struct Options: Sendable {
        /// Maps the path of a model tensor to its key in the file. With `nil`, the key is the text form of the path.
        ///
        /// With a naming closure, the file keys cannot be mapped back to paths, so ``TensorContainer/adoptLayout(_:)``
        /// receives an empty layout.
        public var naming: (@Sendable (TensorPath) -> String)?

        /// Keeps the current value of a model tensor that the file does not have. When `false`, such a tensor is an error.
        public var allowsMissingTensors: Bool

        /// Ignores file entries that no model tensor uses. When `false`, such an entry is an error.
        public var allowsUnusedTensors: Bool

        /// Creates options.
        /// - Parameters:
        ///   - naming: Maps the path of a model tensor to its key in the file. With `nil`, the key is the text form of the path.
        ///   - allowsMissingTensors: Keeps the current value of a model tensor that the file does not have.
        ///   - allowsUnusedTensors: Ignores file entries that no model tensor uses.
        public init(
            naming: (@Sendable (TensorPath) -> String)? = nil,
            allowsMissingTensors: Bool = false,
            allowsUnusedTensors: Bool = false,
        ) {
            self.naming = naming
            self.allowsMissingTensors = allowsMissingTensors
            self.allowsUnusedTensors = allowsUnusedTensors
        }
    }

    /// The options of the decoder.
    public var options: Options

    /// Creates a decoder.
    /// - Parameter options: Options that control how file entries are matched to model tensors.
    public init(options: Options = Options()) {
        self.options = options
    }

    /// Loads the tensors of a file, or of a sharded checkpoint, into a model.
    ///
    /// `url` is a safetensors file, the index file `model.safetensors.index.json` of a sharded checkpoint, or a
    /// directory. A directory must contain an index file or a file `model.safetensors`.
    ///
    /// Before the decoder fills the model, it calls ``TensorContainer/adoptLayout(_:)`` with the paths and shapes
    /// of the stored tensors, first on the model and then on its sublayers.
    ///
    /// - Parameters:
    ///   - layer: The model to fill.
    ///   - url: The file, the index file, or the directory to read.
    /// - Throws: ``SafetensorsError`` when the files are not valid or do not match the model. File system errors
    ///   when a file cannot be read. When an error is thrown, the model is not changed.
    public func load<Layer: TensorContainer>(into layer: inout Layer, from url: URL) throws {
        let files = try sourceFiles(at: url)
        let headers = try files.map { file in
            let header = try header(at: file.url)
            try file.check(header)
            return header
        }
        let urls = files.map(\.url)
        let layout = storedLayout(of: headers)
        try validate(layer, adopting: layout, headers: headers, urls: urls)
        layer.adoptLayoutRecursively(layout)

        try withModelTensors(of: &layer) { tensors in
            let plan = try LoadPlan(tensors: tensors, headers: headers, options: options, urls: urls)
            for (fileIndex, file) in files.enumerated() where !plan.assignments[fileIndex].isEmpty {
                let mapping = try MappedFile(url: file.url)
                // A file that changed since its header was read cannot be read safely.
                guard mapping.bytes.count == headers[fileIndex].dataSectionOffset + (headers[fileIndex].entries.last?.dataOffsets.upperBound ?? 0) else {
                    throw SafetensorsError(.malformedHeader("The file changed while it was read."), url: file.url)
                }
                apply(plan.assignments[fileIndex], from: mapping.bytes, header: headers[fileIndex], to: &tensors, release: mapping.release(upTo:))
            }
        }
    }

    /// Loads the tensors of a safetensors file in memory into a model.
    ///
    /// - Parameters:
    ///   - layer: The model to fill.
    ///   - data: The contents of a safetensors file.
    /// - Throws: ``SafetensorsError`` when the data is not valid or does not match the model. When an error is
    ///   thrown, the model is not changed.
    public func load<Layer: TensorContainer>(into layer: inout Layer, from data: Data) throws {
        try data.withUnsafeBytes { bytes in
            let header = try SafetensorsFormat.decodeHeader(file: bytes)
            let layout = storedLayout(of: [header])
            try validate(layer, adopting: layout, headers: [header], urls: [nil])
            layer.adoptLayoutRecursively(layout)

            try withModelTensors(of: &layer) { tensors in
                let plan = try LoadPlan(tensors: tensors, headers: [header], options: options, urls: [nil])
                apply(plan.assignments[0], from: bytes, header: header, to: &tensors, release: { _ in })
            }
        }
    }

    /// Reads the header of a safetensors file in memory.
    ///
    /// - Parameter data: The contents of a safetensors file.
    /// - Returns: The names, types, shapes, and positions of the tensors, and the metadata.
    /// - Throws: ``SafetensorsError`` when the header is not valid.
    public func header(from data: Data) throws -> SafetensorsHeader {
        try data.withUnsafeBytes { bytes throws(SafetensorsError) in
            try SafetensorsFormat.decodeHeader(file: bytes)
        }
    }

    /// Reads the header of a safetensors file without the tensor data.
    ///
    /// - Parameter url: The safetensors file.
    /// - Returns: The names, types, shapes, and positions of the tensors, and the metadata.
    /// - Throws: ``SafetensorsError`` when the header is not valid. File system errors when the file cannot be read.
    public func header(at url: URL) throws -> SafetensorsHeader {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }
        do {
            let fileSize = try Int(handle.seekToEnd())
            try handle.seek(toOffset: 0)
            let prefix = try handle.read(upToCount: 8) ?? Data()
            let length = try prefix.withUnsafeBytes { bytes throws(SafetensorsError) in
                try SafetensorsFormat.headerLength(from: bytes, fileSize: fileSize)
            }
            let json = try handle.read(upToCount: length) ?? Data()
            guard json.count == length else {
                throw SafetensorsError(.malformedHeader("The file ends inside the header."))
            }
            return try SafetensorsFormat.decodeHeader(json: json, fileSize: fileSize)
        } catch let error as SafetensorsError {
            throw error.at(url)
        }
    }

    // MARK: Implementation

    /// Resolves the files to read and, for an index file, the keys that each shard must hold.
    private func sourceFiles(at url: URL) throws -> [SourceFile] {
        var isDirectory: ObjCBool = false
        let directoryExists = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
        if directoryExists {
            let index = url.appending(path: SafetensorsFormat.indexFileName)
            if FileManager.default.fileExists(atPath: index.path) {
                return try shardFiles(listedIn: index)
            }
            let single = url.appending(path: SafetensorsFormat.singleFileName)
            guard FileManager.default.fileExists(atPath: single.path) else {
                throw SafetensorsError(.shardNotFound(SafetensorsFormat.indexFileName), url: url)
            }
            return [SourceFile(url: single, expectedKeys: nil)]
        }
        if url.pathExtension.lowercased() == "json" {
            return try shardFiles(listedIn: url)
        }
        return [SourceFile(url: url, expectedKeys: nil)]
    }

    private func shardFiles(listedIn indexURL: URL) throws -> [SourceFile] {
        let index: SafetensorsIndex
        do {
            index = try JSONDecoder().decode(SafetensorsIndex.self, from: Data(contentsOf: indexURL))
        } catch let error as DecodingError {
            throw SafetensorsError(.malformedHeader("The index file is not valid: \(error)"), url: indexURL)
        }

        let directory = indexURL.deletingLastPathComponent()
        let keysByShard = Dictionary(grouping: index.weightMap.keys) { index.weightMap[$0] ?? "" }
        return try keysByShard.keys.sorted().map { fileName in
            let shardURL = directory.appending(path: fileName)
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                throw SafetensorsError(.shardNotFound(fileName), url: shardURL)
            }
            return SourceFile(url: shardURL, expectedKeys: Set(keysByShard[fileName] ?? []))
        }
    }

    private func key(for path: TensorPath) -> String {
        options.naming?(path) ?? path.description
    }

    /// The paths and shapes of the tensors in the files.
    private func storedLayout(of headers: [SafetensorsHeader]) -> TensorLayout {
        guard options.naming == nil else {
            return TensorLayout()
        }
        // A key that appears in two shards is reported by LoadPlan, so the first shape is kept here.
        let shapes = headers.flatMap(\.entries).map { (TensorPath($0.name), $0.shape) }
        return TensorLayout(Dictionary(shapes, uniquingKeysWith: { first, _ in first }))
    }

    // The layout is adopted by a copy, so that the model is not changed when the check fails. The copy shares the
    // storage of the model and is released before the model is filled, so the fill does not copy any tensor.

    /// Checks that the model, after it adopts the layout, matches the files.
    private func validate<Layer: TensorContainer>(_ layer: Layer, adopting layout: TensorLayout, headers: [SafetensorsHeader], urls: [URL?]) throws(SafetensorsError) {
        var candidate = layer
        candidate.adoptLayoutRecursively(layout)
        var tensors: [ModelTensor<Layer.Parameter, Layer.Device>] = []
        var collector = TensorVisitor<Layer.Parameter, Layer.Device>(tensors: { tensor, _, path in
            tensors.append(ModelTensor(key: key(for: path), tensor: tensor))
        })
        candidate.visitTensors(&collector)
        _ = try LoadPlan(tensors: tensors, headers: headers, options: options, urls: urls)
    }

    // While body runs, the model holds a placeholder instead of each tensor. The array then holds the only
    // reference to the storage, and a write into a tensor does not copy it.

    /// Takes the tensors out of the model, calls `body`, and puts the tensors back, also when `body` throws.
    private func withModelTensors<Layer: TensorContainer>(
        of layer: inout Layer,
        _ body: (inout [ModelTensor<Layer.Parameter, Layer.Device>]) throws -> Void,
    ) throws {
        let placeholder = Tensor<Layer.Parameter, Layer.Device>(repeating: 0, shape: [])
        var tensors: [ModelTensor<Layer.Parameter, Layer.Device>] = []
        var collector = TensorVisitor<Layer.Parameter, Layer.Device>(tensors: { tensor, _, path in
            tensors.append(ModelTensor(key: key(for: path), tensor: tensor))
            tensor = placeholder
        })
        layer.visitTensors(&collector)

        defer {
            var index = 0
            var writer = TensorVisitor<Layer.Parameter, Layer.Device>(tensors: { tensor, _, _ in
                tensor = tensors[index].tensor
                index += 1
            })
            layer.visitTensors(&writer)
        }
        try body(&tensors)
    }

    /// Copies the entries of one file into the model tensors, in file order.
    private func apply<Element, Device>(
        _ assignments: [LoadPlan.Assignment],
        from file: UnsafeRawBufferPointer,
        header: SafetensorsHeader,
        to tensors: inout [ModelTensor<Element, Device>],
        release: (Int) -> Void,
    ) {
        for assignment in assignments {
            let range = assignment.entry.dataOffsets
            let start = header.dataSectionOffset + range.lowerBound
            let source = UnsafeRawBufferPointer(rebasing: file[start ..< start + range.count])
            tensors[assignment.tensorIndex].tensor.assign(from: source, dtype: assignment.entry.dtype)
            release(start + range.count)
        }
    }
}

/// A file to read, with the keys that the index file lists for it.
private struct SourceFile {
    var url: URL
    var expectedKeys: Set<String>?

    func check(_ header: SafetensorsHeader) throws(SafetensorsError) {
        guard let expectedKeys else {
            return
        }
        let keys = Set(header.entries.map(\.name))
        if let key = expectedKeys.subtracting(keys).sorted().first {
            throw SafetensorsError(.malformedHeader("The index file lists the tensor in this shard, but the shard does not include it."), key: key, url: url)
        }
        if let key = keys.subtracting(expectedKeys).sorted().first {
            throw SafetensorsError(.malformedHeader("The shard includes the tensor, but the index file does not list it in this shard."), key: key, url: url)
        }
    }
}

/// A tensor of the model with its key in the file.
private struct ModelTensor<Element: NumericType, Device: DeviceType> {
    var key: String
    var tensor: Tensor<Element, Device>
}

/// The checked match between file entries and model tensors.
private struct LoadPlan {
    struct Assignment {
        var tensorIndex: Int
        var entry: SafetensorsHeader.Entry
    }

    /// One list per file, sorted by the position of the entry in the file.
    var assignments: [[Assignment]]

    init<Element, Device>(tensors: [ModelTensor<Element, Device>], headers: [SafetensorsHeader], options: SafetensorsDecoder.Options, urls: [URL?]) throws(SafetensorsError) {
        var tensorIndices: [String: Int] = [:]
        for (index, tensor) in tensors.enumerated() {
            guard tensorIndices.updateValue(index, forKey: tensor.key) == nil else {
                throw SafetensorsError(.duplicateKey, key: tensor.key)
            }
        }

        var foundKeys: Set<String> = []
        assignments = []
        for (header, url) in zip(headers, urls) {
            var fileAssignments: [Assignment] = []
            for entry in header.entries {
                guard foundKeys.insert(entry.name).inserted else {
                    throw SafetensorsError(.malformedHeader("Two shards have the tensor."), key: entry.name, url: url)
                }
                guard let tensorIndex = tensorIndices[entry.name] else {
                    if options.allowsUnusedTensors {
                        continue
                    }
                    throw SafetensorsError(.unusedTensor, key: entry.name, url: url)
                }
                guard entry.dtype.isLoadable else {
                    throw SafetensorsError(.unsupportedDType(entry.dtype.rawValue), key: entry.name, url: url)
                }
                let modelShape = tensors[tensorIndex].tensor.shape
                guard entry.shape == modelShape else {
                    throw SafetensorsError(.shapeMismatch(file: entry.shape, model: modelShape), key: entry.name, url: url)
                }
                fileAssignments.append(Assignment(tensorIndex: tensorIndex, entry: entry))
            }
            assignments.append(fileAssignments)
        }

        if !options.allowsMissingTensors, let missing = tensors.first(where: { !foundKeys.contains($0.key) }) {
            throw SafetensorsError(.missingTensor, key: missing.key, url: urls.count == 1 ? urls[0] : nil)
        }
    }
}

extension Tensor {
    /// Overwrites the elements of the tensor with little-endian values of the given type.
    ///
    /// Entries of the element type of the tensor are copied bit-exact.
    ///
    /// - Parameters:
    ///   - source: The bytes of the entry. It has `count` elements of type `dtype`.
    ///   - dtype: The element type of the entry. It must be `F32`, `F64`, or `I32`.
    mutating func assign(from source: UnsafeRawBufferPointer, dtype: SafetensorsDType) {
        let destination = mutableValues.values
        discardContext()
        guard count > 0 else {
            return
        }
        switch dtype {
        case .f32: Self.convert(source, as: Float.self, into: destination, count: count) { Element($0) }
        case .f64: Self.convert(source, as: Double.self, into: destination, count: count) { Element($0) }
        case .i32: Self.convert(source, as: Int32.self, into: destination, count: count) { Element($0) }
        default: preconditionFailure("The element type \(dtype) cannot be loaded.")
        }
    }

    // Converts through a small buffer, so that the conversion works on every device without a host copy of the tensor.
    private static func convert<Source>(
        _ source: UnsafeRawBufferPointer,
        as sourceType: Source.Type,
        into destination: MutableBuffer<Element, Device>,
        count: Int,
        _ transform: (Source) -> Element,
    ) {
        let base = source.baseAddress!
        if Source.self == Element.self, Int(bitPattern: base) % MemoryLayout<Element>.alignment == 0 {
            let elements = UnsafeBufferPointer(start: base.assumingMemoryBound(to: Element.self), count: count)
            Device.Memory.assign(from: elements, to: destination, count: count)
            return
        }

        let chunkSize = Swift.min(count, 16384)
        let chunk = UnsafeMutableBufferPointer<Element>.allocate(capacity: chunkSize)
        defer {
            chunk.deallocate()
        }
        var start = 0
        while start < count {
            let length = Swift.min(chunkSize, count - start)
            for index in 0 ..< length {
                chunk[index] = transform(source.loadUnaligned(fromByteOffset: (start + index) * MemoryLayout<Source>.size, as: Source.self))
            }
            Device.Memory.assign(from: UnsafeBufferPointer(rebasing: chunk[0 ..< length]), to: destination.advanced(by: start), count: length)
            start += length
        }
    }
}
