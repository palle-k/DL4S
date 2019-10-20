//
//  Engine.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
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


public protocol DeviceType {
    associatedtype Memory: MemoryOperatorsType where Memory.Device == Self
    associatedtype Engine: EngineType where Engine.Device == Self
}


public protocol MemoryOperatorsType {
    associatedtype Device: DeviceType where Device.Memory == Self
    associatedtype RawBuffer: Hashable
    
    static func allocateBuffer<Element>(withCapacity: Int, type: Element.Type) -> Buffer<Element, Device>
    static func allocateBuffer<Element>(withShape shape: [Int], type: Element.Type) -> ShapedBuffer<Element, Device>
    static func free<Element>(_ buffer: Buffer<Element, Device>)
    static func free<Element>(_ buffer: ShapedBuffer<Element, Device>)
    
    static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, Device>, count: Int)
    static func assign<Element>(from source: Buffer<Element, Device>, to destination: Buffer<Element, Device>, count: Int)
    static func assign<Element>(from source: Buffer<Element, Device>, to destination: UnsafeMutableBufferPointer<Element>, count: Int)
    
    static func getValue<Element>(from source: Buffer<Element, Device>) -> Element
    static func getSize<Element>(of buffer: Buffer<Element, Device>) -> Int
    
    static func get<Element>(slice: [Int?], of buffer: Buffer<Element, Device>, with shape: [Int]) -> (Buffer<Element, Device>, Bool, [Int])
    static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, Device>, with shape: [Int]) -> (Buffer<Element, Device>, Bool, [Int])
    static func set<Element>(slice: [Int?], of buffer: Buffer<Element, Device>, with dstShape: [Int], from source: Buffer<Element, Device>, with sourceShape: [Int])
    static func set<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, Device>, with dstShape: [Int], from source: Buffer<Element, Device>, with sourceShape: [Int])
    
    static func setPointee<Element>(of buffer: Buffer<Element, Device>, to newValue: Element)
    
    static func advance<Element>(buffer: Buffer<Element, Device>, by advancement: Int) -> Buffer<Element, Device>
}
