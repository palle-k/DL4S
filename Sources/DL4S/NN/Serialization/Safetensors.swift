//
//  Safetensors.swift
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

// The safetensors format stores little-endian values. All platforms that DL4S supports are little-endian.
#if _endian(big)
#error("Safetensors support requires a little-endian platform.")
#endif

/// The element type of a tensor in a safetensors file.
///
/// DL4S writes ``f32``, ``f64``, and ``i32``, and reads these three types into `Float`, `Double`, and `Int32`
/// tensors. ``SafetensorsHeader`` also reports the other types of the format, but a model cannot load them.
public struct SafetensorsDType: RawRepresentable, Hashable, Sendable, CustomStringConvertible {
    /// The name of the type in the header, for example `F32`.
    public let rawValue: String

    /// Creates a type from its name in the header.
    /// - Parameter rawValue: The name of the type, for example `F32`.
    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// 32-bit IEEE 754 floating point number.
    public static let f32 = SafetensorsDType(rawValue: "F32")
    /// 64-bit IEEE 754 floating point number.
    public static let f64 = SafetensorsDType(rawValue: "F64")
    /// 32-bit signed integer.
    public static let i32 = SafetensorsDType(rawValue: "I32")
    /// 16-bit IEEE 754 floating point number.
    public static let f16 = SafetensorsDType(rawValue: "F16")
    /// 16-bit brain floating point number.
    public static let bf16 = SafetensorsDType(rawValue: "BF16")
    /// 64-bit signed integer.
    public static let i64 = SafetensorsDType(rawValue: "I64")

    /// Number of bits of one element, or `nil` for a type that DL4S does not know.
    public var bitsPerElement: Int? {
        switch rawValue {
        case "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0": 8
        case "I16", "U16", "F16", "BF16": 16
        case "I32", "U32", "F32": 32
        case "I64", "U64", "F64", "C64": 64
        default: nil
        }
    }

    public var description: String {
        rawValue
    }

    /// The type that DL4S writes for tensors with elements of the given type.
    static func of<Element>(_ type: Element.Type) throws(SafetensorsError) -> SafetensorsDType {
        switch type {
        case is Float.Type: .f32
        case is Double.Type: .f64
        case is Int32.Type: .i32
        default: throw SafetensorsError(.unsupportedDType(String(describing: type)))
        }
    }

    /// Indicates whether DL4S can convert elements of this type into a tensor.
    var isLoadable: Bool {
        self == .f32 || self == .f64 || self == .i32
    }
}

/// The header of a safetensors file: the names, types, shapes, and positions of the tensors, and the metadata.
public struct SafetensorsHeader: Hashable, Sendable {
    /// The description of one tensor in the file.
    public struct Entry: Hashable, Sendable {
        /// The key of the tensor.
        public let name: String

        /// The element type of the tensor.
        public let dtype: SafetensorsDType

        /// The shape of the tensor. An empty shape is a scalar.
        public let shape: [Int]

        /// The byte range of the tensor, relative to the start of the data section.
        public let dataOffsets: Range<Int>
    }

    /// The tensors in the file, sorted by their position in the data section.
    public let entries: [Entry]

    /// The contents of the `__metadata__` entry. It is empty when the file has no metadata.
    public let metadata: [String: String]

    /// The byte offset of the data section in the file. It is the length of the header plus 8.
    public let dataSectionOffset: Int

    private let indicesByName: [String: Int]

    init(entries: [Entry], metadata: [String: String], dataSectionOffset: Int) {
        self.entries = entries.sorted { $0.dataOffsets.lowerBound < $1.dataOffsets.lowerBound }
        self.metadata = metadata
        self.dataSectionOffset = dataSectionOffset
        indicesByName = Dictionary(uniqueKeysWithValues: self.entries.enumerated().map { ($1.name, $0) })
    }

    /// Returns the entry of the tensor with the given key.
    /// - Parameter name: The key of the tensor.
    /// - Returns: The entry, or `nil` when the file has no tensor with that key.
    public subscript(name: String) -> Entry? {
        indicesByName[name].map { entries[$0] }
    }
}

/// An error that the safetensors encoder or decoder throws.
///
/// File system errors, such as a file that cannot be opened, are thrown as the errors of Foundation or as
/// `POSIXError` and not as `SafetensorsError`.
public struct SafetensorsError: Error, Sendable, CustomStringConvertible {
    /// The reason of the error.
    public enum Kind: Sendable, Equatable {
        /// The header is not valid. The value tells what is wrong.
        case malformedHeader(String)

        /// A tensor has an element type that cannot be converted. The value is the name of the type.
        case unsupportedDType(String)

        /// The file has no tensor for a model tensor, and ``SafetensorsDecoder/Options/allowsMissingTensors`` is off.
        case missingTensor

        /// The file has a tensor that no model tensor uses, and ``SafetensorsDecoder/Options/allowsUnusedTensors`` is off.
        case unusedTensor

        /// The tensor in the file and the model tensor have different shapes.
        case shapeMismatch(file: [Int], model: [Int])

        /// A shard that the index file lists does not exist. The value is the file name of the shard.
        case shardNotFound(String)

        /// Two tensors have the same key, because the naming closure maps two paths to the same key.
        case duplicateKey
    }

    /// The reason of the error.
    public let kind: Kind

    /// The key of the tensor that caused the error, if the error is about one tensor.
    public let key: String?

    /// The file that caused the error, if the error is about a file.
    public let url: URL?

    /// Creates an error.
    /// - Parameters:
    ///   - kind: The reason of the error.
    ///   - key: The key of the tensor that caused the error.
    ///   - url: The file that caused the error.
    public init(_ kind: Kind, key: String? = nil, url: URL? = nil) {
        self.kind = kind
        self.key = key
        self.url = url
    }

    public var description: String {
        var parts = ["Safetensors error: \(kind)"]
        if let key {
            parts.append("tensor \(key)")
        }
        if let url {
            parts.append("file \(url.path)")
        }
        return parts.joined(separator: ", ")
    }

    func at(_ url: URL?) -> SafetensorsError {
        SafetensorsError(kind, key: key, url: self.url ?? url)
    }
}

/// Reads and writes the binary layout of a safetensors file.
///
/// A file starts with the length N of the header as an unsigned 64-bit little-endian integer. The next N bytes
/// are a JSON object with one entry per tensor and an optional `__metadata__` entry. The data section follows
/// the header and holds the tensors in row-major order without gaps.
enum SafetensorsFormat {
    static let metadataKey = "__metadata__"

    // The reference implementation rejects larger headers.
    static let maximumHeaderLength = 100_000_000

    /// The data section starts at a multiple of this value. The header is padded with spaces.
    static let dataAlignment = 8

    /// The name of the index file of a sharded checkpoint, as in the Hugging Face layout.
    static let indexFileName = "model.safetensors.index.json"

    /// The name of the file that a directory without an index file holds.
    static let singleFileName = "model.safetensors"

    // MARK: Encoding

    /// Returns the header bytes, including the 8-byte length and the padding.
    static func encodeHeader(entries: [SafetensorsHeader.Entry], metadata: [String: String]) throws -> Data {
        var object: [String: HeaderValue] = [:]
        for entry in entries {
            object[entry.name] = .tensor(TensorJSON(
                dtype: entry.dtype.rawValue,
                shape: entry.shape,
                dataOffsets: [entry.dataOffsets.lowerBound, entry.dataOffsets.upperBound],
            ))
        }
        if !metadata.isEmpty {
            object[metadataKey] = .metadata(metadata)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var json = try encoder.encode(object)
        let padding = (dataAlignment - (8 + json.count) % dataAlignment) % dataAlignment
        json.append(contentsOf: repeatElement(UInt8(ascii: " "), count: padding))

        var header = Data(capacity: 8 + json.count)
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { header.append(contentsOf: $0) }
        header.append(json)
        return header
    }

    // MARK: Decoding

    /// Reads the header length from the first 8 bytes of a file.
    static func headerLength(from prefix: UnsafeRawBufferPointer, fileSize: Int) throws(SafetensorsError) -> Int {
        guard prefix.count >= 8, fileSize >= 8 else {
            throw SafetensorsError(.malformedHeader("The file has fewer than 8 bytes."))
        }
        let length = UInt64(littleEndian: prefix.loadUnaligned(as: UInt64.self))
        guard length <= UInt64(maximumHeaderLength) else {
            throw SafetensorsError(.malformedHeader("The header length \(length) is larger than \(maximumHeaderLength) bytes."))
        }
        guard Int(length) <= fileSize - 8 else {
            throw SafetensorsError(.malformedHeader("The header length \(length) is larger than the file."))
        }
        return Int(length)
    }

    /// Parses and validates the JSON header.
    ///
    /// - Parameters:
    ///   - json: The header bytes after the 8-byte length.
    ///   - fileSize: The size of the file, used to check that the tensors fill the data section.
    static func decodeHeader(json: Data, fileSize: Int) throws(SafetensorsError) -> SafetensorsHeader {
        guard json.first == UInt8(ascii: "{") else {
            throw SafetensorsError(.malformedHeader("The header is not a JSON object."))
        }
        let object: [String: HeaderValue]
        do {
            object = try JSONDecoder().decode([String: HeaderValue].self, from: json)
        } catch {
            throw SafetensorsError(.malformedHeader("The header is not valid JSON: \(error)"))
        }

        var entries: [SafetensorsHeader.Entry] = []
        var metadata: [String: String] = [:]
        for (name, value) in object {
            switch value {
            case let .metadata(values) where name == metadataKey:
                metadata = values
            case .metadata:
                throw SafetensorsError(.malformedHeader("The entry is not a tensor."), key: name)
            case .tensor where name == metadataKey:
                throw SafetensorsError(.malformedHeader("The metadata entry is not a map of strings."), key: name)
            case let .tensor(tensor):
                try entries.append(validatedEntry(named: name, tensor))
            }
        }

        let dataSectionOffset = 8 + json.count
        let header = SafetensorsHeader(entries: entries, metadata: metadata, dataSectionOffset: dataSectionOffset)

        // As in the reference implementation, the tensors must fill the data section from the start, without
        // gaps and overlaps.
        var end = 0
        for entry in header.entries {
            guard entry.dataOffsets.lowerBound == end else {
                throw SafetensorsError(.malformedHeader("The tensor starts at byte \(entry.dataOffsets.lowerBound), but the previous tensor ends at byte \(end)."), key: entry.name)
            }
            end = entry.dataOffsets.upperBound
        }
        guard dataSectionOffset + end == fileSize else {
            throw SafetensorsError(.malformedHeader("The tensors use \(end) bytes, but the data section has \(fileSize - dataSectionOffset) bytes."))
        }
        return header
    }

    /// Parses a complete file that is in memory.
    static func decodeHeader(file: UnsafeRawBufferPointer) throws(SafetensorsError) -> SafetensorsHeader {
        let length = try headerLength(from: file, fileSize: file.count)
        let json = Data(UnsafeRawBufferPointer(rebasing: file[8 ..< 8 + length]))
        return try decodeHeader(json: json, fileSize: file.count)
    }

    private static func validatedEntry(named name: String, _ tensor: TensorJSON) throws(SafetensorsError) -> SafetensorsHeader.Entry {
        guard tensor.shape.allSatisfy({ $0 >= 0 }) else {
            throw SafetensorsError(.malformedHeader("The shape \(tensor.shape) has a negative dimension."), key: name)
        }
        guard tensor.dataOffsets.count == 2, tensor.dataOffsets[0] >= 0, tensor.dataOffsets[0] <= tensor.dataOffsets[1] else {
            throw SafetensorsError(.malformedHeader("The data offsets \(tensor.dataOffsets) are not a valid range."), key: name)
        }
        let dtype = SafetensorsDType(rawValue: tensor.dtype)
        let range = tensor.dataOffsets[0] ..< tensor.dataOffsets[1]
        if let bits = dtype.bitsPerElement {
            let (count, overflow) = tensor.shape.reduce((1, false)) { partial, dimension in
                let product = partial.0.multipliedReportingOverflow(by: dimension)
                return (product.partialValue, partial.1 || product.overflow)
            }
            guard !overflow, count <= Int.max / bits, (count * bits + 7) / 8 == range.count else {
                throw SafetensorsError(.malformedHeader("The shape \(tensor.shape) of type \(dtype) does not match the \(range.count) bytes of the tensor."), key: name)
            }
        }
        return SafetensorsHeader.Entry(name: name, dtype: dtype, shape: tensor.shape, dataOffsets: range)
    }
}

/// One value of the header object: a tensor or the metadata.
private enum HeaderValue: Codable {
    case tensor(TensorJSON)
    case metadata([String: String])

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let metadata = try? container.decode([String: String].self) {
            self = .metadata(metadata)
        } else {
            self = try .tensor(container.decode(TensorJSON.self))
        }
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .tensor(tensor): try container.encode(tensor)
        case let .metadata(metadata): try container.encode(metadata)
        }
    }
}

private struct TensorJSON: Codable {
    var dtype: String
    var shape: [Int]
    var dataOffsets: [Int]

    enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case dataOffsets = "data_offsets"
    }
}

/// The index file of a sharded checkpoint (`model.safetensors.index.json`).
struct SafetensorsIndex: Codable {
    // Writers other than Hugging Face can put other values here, so the decoder does not read it.
    var metadata: SafetensorsIndexMetadata?

    /// Maps the key of every tensor to the file name of its shard.
    var weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case metadata
        case weightMap = "weight_map"
    }

    init(totalSize: Int, weightMap: [String: String]) {
        metadata = SafetensorsIndexMetadata(totalSize: totalSize)
        self.weightMap = weightMap
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        metadata = nil
        weightMap = try container.decode([String: String].self, forKey: .weightMap)
    }
}

/// The `metadata` entry of the index file.
struct SafetensorsIndexMetadata: Codable {
    /// The number of tensor bytes in all shards.
    var totalSize: Int

    enum CodingKeys: String, CodingKey {
        case totalSize = "total_size"
    }
}
