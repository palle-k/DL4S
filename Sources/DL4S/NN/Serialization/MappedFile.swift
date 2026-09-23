//
//  MappedFile.swift
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

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif
import Foundation

// mmap instead of Data(contentsOf:options: .alwaysMapped) enables more effective use of memory mapped files

/// A read-only memory mapping of a file, advised for sequential access.
///
/// Call ``release(upTo:)`` after the bytes before an offset are read.
final class MappedFile {
    /// The contents of the file.
    let bytes: UnsafeRawBufferPointer

    private let pageSize = Int(sysconf(Int32(_SC_PAGESIZE)))
    private var releasedUpTo = 0

    /// Maps a file into memory.
    /// - Parameter url: The file to map.
    /// - Throws: `POSIXError` when the file cannot be opened or mapped.
    init(url: URL) throws {
        let descriptor = url.withUnsafeFileSystemRepresentation { path in
            path.map { open($0, O_RDONLY) } ?? -1
        }
        guard descriptor >= 0 else {
            throw MappedFile.lastError()
        }
        // The mapping stays valid when the descriptor is closed.
        defer {
            close(descriptor)
        }

        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw MappedFile.lastError()
        }
        let size = Int(status.st_size)
        guard size > 0 else {
            // mmap rejects a length of 0.
            bytes = UnsafeRawBufferPointer(start: nil, count: 0)
            return
        }

        let address: UnsafeMutableRawPointer? = mmap(nil, size, PROT_READ, MAP_PRIVATE, descriptor, 0)
        guard let address, address != UnsafeMutableRawPointer(bitPattern: -1) else {
            throw MappedFile.lastError()
        }
        madvise(address, size, MADV_SEQUENTIAL)
        bytes = UnsafeRawBufferPointer(start: address, count: size)
    }

    deinit {
        if let address = bytes.baseAddress {
            munmap(UnsafeMutableRawPointer(mutating: address), bytes.count)
        }
    }

    /// Tells the operating system that the bytes before an offset are not needed again.
    ///
    /// Only full pages are released. The bytes stay readable after a release.
    /// - Parameter offset: The byte offset up to which the file was read.
    func release(upTo offset: Int) {
        guard let address = bytes.baseAddress else {
            return
        }
        let end = min(offset, bytes.count) / pageSize * pageSize
        guard end > releasedUpTo else {
            return
        }
        madvise(UnsafeMutableRawPointer(mutating: address + releasedUpTo), end - releasedUpTo, MADV_DONTNEED)
        releasedUpTo = end
    }

    private static func lastError() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}
