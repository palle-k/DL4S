//
//  SummaryWriter.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.04.19.
//

import Foundation


public class SummaryWriter {
    private let eventsFile: FileHandle
    
    public init(destination: URL) throws {
        let eventsFile = destination.appendingPathComponent("events.csv")
        
        if !FileManager.default.fileExists(atPath: eventsFile.path) {
            FileManager.default.createFile(atPath: eventsFile.path, contents: nil, attributes: nil)
        }
        
        self.eventsFile = try FileHandle(forWritingTo: eventsFile)
    }
    
    public func write<Scalar: NumericType>(_ scalar: Scalar, named name: String, at iteration: Int) {
        let milliseconds = Int(Date().timeIntervalSince1970 * 1000)
        self.eventsFile.seekToEndOfFile()
        self.eventsFile.write("\(iteration),\(milliseconds),\(name),\(scalar)")
        self.eventsFile.synchronizeFile()
    }
}
