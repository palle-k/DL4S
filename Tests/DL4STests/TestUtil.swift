//
//  TestUtil.swift
//  DL4STests
//
//  Created by Palle Klewitz on 31.08.26.
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
import XCTest

extension XCTestCase {
    /// Skips the current test unless the `DL4S_LONG_TESTS` environment variable is set.
    ///
    /// Tests that perform longer optimization runs or measure performance may take a while.
    /// CI should not run them by default. Set `DL4S_LONG_TESTS=1` to activate them.
    func skipUnlessLongTestsEnabled() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["DL4S_LONG_TESTS"] == nil,
            "Skipping long test. Set DL4S_LONG_TESTS=1 to run."
        )
    }
}

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Reads memory statistics of the test process.
enum ProcessMemory {
    /// Resident set size of the process in bytes, or `nil` if the platform does not report it.
    static func residentBytes() -> Int? {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return nil
        }
        return Int(info.resident_size)
        #elseif os(Linux)
        // /proc/self/statm lists the total and resident size of the process in pages.
        guard let statm = try? String(contentsOfFile: "/proc/self/statm", encoding: .utf8) else {
            return nil
        }
        let fields = statm.split(separator: " ")
        guard fields.count > 1, let residentPages = Int(fields[1]) else {
            return nil
        }
        return residentPages * Int(sysconf(Int32(_SC_PAGESIZE)))
        #else
        return nil
        #endif
    }
}
