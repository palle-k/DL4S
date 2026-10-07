//
//  Embedding.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.10.19.
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

/// Transforms discrete values, such as word indices, into a lower dimensional embedding.
@Layer
public struct Embedding<Element: RandomizableType, Device: DeviceType>: Codable, Sendable {
    /// Matrix of embedding vectors, shape [inputFeatures, outputSize]
    public var embeddingMatrix: Tensor<Element, Device>

    /// Number of input features
    public var inputFeatures: Int {
        embeddingMatrix.shape[0]
    }

    /// Size of embedded feature vectors
    public var outputSize: Int {
        embeddingMatrix.shape[1]
    }

    /// Index for padding that is ignored in inputs to the layer
    public let ignoreIndex: Int

    /// Creates an embedding layer that has an input vocabulary of size `inputFeatures` and returns embeddings with the size `outputSize`.
    ///
    /// The layer expects categorial inputs with a shape of [batch size] and returns embeddings with a shape of [batch size, outputSize]
    ///
    /// - Parameters:
    ///   - inputFeatures: Vocabulary size.
    ///   - outputSize: Embedding dimensionality.
    ///   - ignoreIndex: Token index that is ignored when retreiving values from the embedding matrix
    public init(inputFeatures: Int, outputSize: Int, ignoreIndex: Int = -1) {
        var generator = WyHash()
        self.init(inputFeatures: inputFeatures, outputSize: outputSize, ignoreIndex: ignoreIndex, using: &generator)
    }

    /// Creates an embedding layer that has an input vocabulary of size `inputFeatures` and returns embeddings with the size `outputSize`.
    ///
    /// The layer expects categorial inputs with a shape of [batch size] and returns embeddings with a shape of [batch size, outputSize]
    ///
    /// - Parameters:
    ///   - inputFeatures: Vocabulary size.
    ///   - outputSize: Embedding dimensionality.
    ///   - ignoreIndex: Token index that is ignored when retreiving values from the embedding matrix
    ///   - generator: Random number generator that provides the initial weights.
    public init<Generator: RandomNumberGenerator>(inputFeatures: Int, outputSize: Int, ignoreIndex: Int = -1, using generator: inout Generator) {
        embeddingMatrix = Tensor<Element, Device>(heNormalWithShape: [inputFeatures, outputSize], requiresGradient: true, using: &generator)
        #if DEBUG
        embeddingMatrix.tag = "W"
        #endif
        self.ignoreIndex = ignoreIndex
    }

    /// Loads pretrained word embeddings from the space or tab separated values file at the given path
    /// and arranges them according to the order of words provided.
    ///
    /// The embeddings are expected to be arranged using the following format:
    ///
    ///     word1 num num num ... num
    ///     word2 num num num ... num
    ///     ...
    ///
    /// Row `i` of the embedding matrix is the vector of `words[i]`. Lines of words that are not in `words` are skipped,
    /// and when a word has more than one line, the first line is used. A word that the file does not have gets a
    /// random vector with the distribution of ``init(inputFeatures:outputSize:ignoreIndex:)``.
    ///
    /// - Parameters:
    ///   - words: Provided word order.
    ///   - embeddingsURL: Path to pretrained embeddings
    ///   - verbose: If set to true, print out loading progress
    ///   - ignoreIndex: Token index that is ignored when retreiving values from the embedding matrix
    /// - Throws: ``EmbeddingLoadingError`` when the file cannot be read, when a line of a word in `words` is not
    ///   valid, or when the file has no word of `words`.
    public init(words: [String], embeddingsURL: URL, verbose: Bool = false, ignoreIndex: Int = -1) throws {
        var generator = WyHash()
        try self.init(words: words, embeddingsURL: embeddingsURL, verbose: verbose, ignoreIndex: ignoreIndex, using: &generator)
    }

    /// Loads pretrained word embeddings from the space or tab separated values file at the given path
    /// and arranges them according to the order of words provided.
    ///
    /// The embeddings are expected to be arranged using the following format:
    ///
    ///     word1 num num num ... num
    ///     word2 num num num ... num
    ///     ...
    ///
    /// Row `i` of the embedding matrix is the vector of `words[i]`. Lines of words that are not in `words` are skipped,
    /// and when a word has more than one line, the first line is used. A word that the file does not have gets a
    /// random vector with the distribution of ``init(inputFeatures:outputSize:ignoreIndex:)``.
    ///
    /// - Parameters:
    ///   - words: Provided word order.
    ///   - embeddingsURL: Path to pretrained embeddings
    ///   - verbose: If set to true, print out loading progress
    ///   - ignoreIndex: Token index that is ignored when retreiving values from the embedding matrix
    ///   - generator: Random number generator that provides the vectors of words that the file does not have.
    /// - Throws: ``EmbeddingLoadingError`` when the file cannot be read, when a line of a word in `words` is not
    ///   valid, or when the file has no word of `words`.
    public init<Generator: RandomNumberGenerator>(
        words: [String],
        embeddingsURL: URL,
        verbose: Bool = false,
        ignoreIndex: Int = -1,
        using generator: inout Generator,
    ) throws {
        let data: Data
        do {
            data = try Data(contentsOf: embeddingsURL, options: .mappedIfSafe)
        } catch {
            throw EmbeddingLoadingError.unreadableFile(url: embeddingsURL, underlyingError: error)
        }

        // A word can occur more than once in the vocabulary, and all of its rows get its vector.
        let rowsByWord = Dictionary(grouping: words.indices) { words[$0] }
        var isLoaded = [Bool](repeating: false, count: words.count)
        var loadedWordCount = 0
        var vectorSize: Int?
        // The matrix is filled on the host and copied to the device once.
        var matrix: [Element] = []

        var progress = verbose ? ProgressBar<Void>(totalUnitCount: rowsByWord.count, formatUserInfo: { "" }, label: "loading embeddings") : nil

        try data.withUnsafeBytes { (file: UnsafeRawBufferPointer) in
            var lineStart = 0
            var lineNumber = 0
            while lineStart < file.count, loadedWordCount < rowsByWord.count {
                let lineEnd = file[lineStart...].firstIndex(of: UInt8(ascii: "\n")) ?? file.count
                let line = file[lineStart ..< lineEnd]
                lineStart = lineEnd + 1
                lineNumber += 1

                // A word that is not valid UTF-8 is not in the vocabulary, so its line is skipped and the lines
                // after it are read.
                guard let wordEnd = line.firstIndex(where: Self.isSeparator),
                      let word = String(bytes: file[line.startIndex ..< wordEnd], encoding: .utf8),
                      let rows = rowsByWord[word], !isLoaded[rows[0]]
                else {
                    continue
                }

                let fields = file[wordEnd ..< line.endIndex].split(omittingEmptySubsequences: true, whereSeparator: Self.isSeparator)
                let vector = try fields.map { field in
                    let text = String(bytes: field, encoding: .utf8)
                    guard let text, let value = Double(text) else {
                        throw EmbeddingLoadingError.invalidValue(url: embeddingsURL, line: lineNumber, word: word, value: text ?? String(describing: Array(field)))
                    }
                    return Element(value)
                }
                if vectorSize == nil {
                    guard !vector.isEmpty else {
                        throw EmbeddingLoadingError.sizeMismatch(url: embeddingsURL, line: lineNumber, word: word, expected: nil, found: 0)
                    }
                    vectorSize = vector.count
                    matrix = Tensor<Element, CPU>(heNormalWithShape: [words.count, vector.count], using: &generator).elements
                }
                guard vector.count == vectorSize else {
                    throw EmbeddingLoadingError.sizeMismatch(url: embeddingsURL, line: lineNumber, word: word, expected: vectorSize, found: vector.count)
                }

                for row in rows {
                    matrix.replaceSubrange(row * vector.count ..< (row + 1) * vector.count, with: vector)
                    isLoaded[row] = true
                }
                loadedWordCount += 1
                progress?.next(userInfo: ())
            }
        }

        progress?.complete()

        guard let vectorSize else {
            throw EmbeddingLoadingError.noWordFound(url: embeddingsURL)
        }

        if verbose {
            print("Unknown: \(isLoaded.count(where: { !$0 })) of \(words.count)")
            print("Embedding size: \(vectorSize)")
        }

        embeddingMatrix = Tensor<Element, Device>(matrix, shape: [words.count, vectorSize], requiresGradient: true)
        #if DEBUG
        embeddingMatrix.tag = "W"
        #endif
        self.ignoreIndex = ignoreIndex
    }

    /// Indicates whether a byte separates the fields of a line of an embeddings file.
    private static func isSeparator(_ byte: UInt8) -> Bool {
        byte == UInt8(ascii: " ") || byte == UInt8(ascii: "\t") || byte == UInt8(ascii: "\r")
    }

    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    public func callAsFunction(_ inputs: Tensor<Int32, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Embedding") {
            embeddingMatrix.gatheringRows(at: inputs, ignoreIndex: Int32(ignoreIndex))
        }
    }
}

/// An error that ``Embedding/init(words:embeddingsURL:verbose:ignoreIndex:using:)`` throws.
public enum EmbeddingLoadingError: Error, CustomStringConvertible {
    /// The file cannot be read. The underlying error tells why.
    case unreadableFile(url: URL, underlyingError: any Error)

    /// A line of a word in the vocabulary has a value that is not a number. The line number starts at 1.
    case invalidValue(url: URL, line: Int, word: String, value: String)

    /// The vector of a word has a different size than the vector of the first word that was loaded, or no values.
    /// `expected` is nil when the word is the first word that was loaded.
    case sizeMismatch(url: URL, line: Int, word: String, expected: Int?, found: Int)

    /// The file has no word of the vocabulary.
    case noWordFound(url: URL)

    public var description: String {
        switch self {
        case let .unreadableFile(url, underlyingError):
            "Cannot read the embeddings file \(url.path): \(underlyingError)"
        case let .invalidValue(url, line, word, value):
            "The value \"\(value)\" of the word \"\(word)\" in line \(line) of \(url.path) is not a number."
        case let .sizeMismatch(url, line, word, expected, found):
            if let expected {
                "The word \"\(word)\" in line \(line) of \(url.path) has \(found) values, but the words before it have \(expected)."
            } else {
                "The word \"\(word)\" in line \(line) of \(url.path) has no values."
            }
        case let .noWordFound(url):
            "The embeddings file \(url.path) has no word of the vocabulary."
        }
    }
}
