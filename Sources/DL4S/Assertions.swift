//
//  Assertions.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.02.19.
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
