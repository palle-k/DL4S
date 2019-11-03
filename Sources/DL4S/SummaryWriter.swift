//
//  SummaryWriter.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.04.19.
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


/// Writes a summary of a training procedure to a CSV file.
public class SummaryWriter {
    private let eventsFile: FileHandle
    
    /// Writes a summary of a training procedure to a CSV file.
    /// The CSV file will be stored in a folder named `runName` relative to the given destination URL.
    /// - Parameters:
    ///   - destination: Destination URL to store the run directory in
    ///   - runName: Name of the run directory.
    public init(destination: URL, runName: String) throws {
        let dir = destination.appendingPathComponent(runName, isDirectory: true)
        let eventsFile = dir.appendingPathComponent("summary.csv")
        
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(atPath: dir.path, withIntermediateDirectories: true, attributes: nil)
        }
        if !FileManager.default.fileExists(atPath: eventsFile.path) {
            FileManager.default.createFile(atPath: eventsFile.path, contents: nil, attributes: nil)
        }
        
        self.eventsFile = try FileHandle(forWritingTo: eventsFile)
    }
    
    /// Writes a scalar with the given name into the summary
    /// - Parameters:
    ///   - scalar: Scalar value
    ///   - name: Name of the scalar
    ///   - iteration: Training iteration, at which the scalar was captured.
    public func write<Scalar: NumericType>(_ scalar: Scalar, named name: String, at iteration: Int) {
        let milliseconds = Int(Date().timeIntervalSince1970 * 1000)
        self.eventsFile.seekToEndOfFile()
        self.eventsFile.write("\(iteration),\(milliseconds),\(name),\(scalar)\n")
        self.eventsFile.synchronizeFile()
    }
}
