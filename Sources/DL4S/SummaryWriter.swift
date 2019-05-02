//
//  SummaryWriter.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.04.19.
//

import Foundation


public class SummaryWriter {
    private let eventsFile: FileHandle
    
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
    
    public func write<Scalar: NumericType>(_ scalar: Scalar, named name: String, at iteration: Int) {
        let milliseconds = Int(Date().timeIntervalSince1970 * 1000)
        self.eventsFile.seekToEndOfFile()
        self.eventsFile.write("\(iteration),\(milliseconds),\(name),\(scalar)\n")
        self.eventsFile.synchronizeFile()
    }
}
