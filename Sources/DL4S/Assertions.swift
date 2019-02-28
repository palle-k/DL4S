//
//  Assertions.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.02.19.
//

import Foundation


extension FileHandle: TextOutputStream {
    public func write(_ string: String) {
        self.write(string.data(using: .utf8)!)
    }
    
    public static var stdout: FileHandle {
        get {
            return FileHandle.standardOutput
        }
        set {
            // noop
        }
    }
    
    public static var stderr: FileHandle {
        get {
            return FileHandle.standardError
        }
        set {
            // noop
        }
    }
}

func weakAssert(_ assertion: @autoclosure () -> Bool, message: String = "", line: Int = #line, function: String = #function, file: String = #file) {
    #if DEBUG
    if !assertion() {
        print("Assertion failed at \(file):\(function):\(line) \(message.count > 0 ? ": \(message)" : "")", to: &FileHandle.stderr)
    }
    #endif
}
