//
//  Buffer.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


class Buffer<Element> {
    static func allocate(capacity: Int) -> Buffer<Element> {
        return Buffer(capacity: capacity)
    }
    
    private var parent: Buffer<Element>?
    private let capacity: Int
    let ptr: UnsafeMutablePointer<Element>
    
    private init(capacity: Int) {
        self.capacity = capacity
        ptr = UnsafeMutablePointer.allocate(capacity: capacity)
    }
    
    deinit {
        if parent == nil {
            ptr.deallocate()
        }
    }
    
    func deallocate() {
        if parent != nil {
            fatalError("Cannot deallocate unowned buffer")
        }
        ptr.deallocate()
    }
    
    var pointee: Element {
        return ptr.pointee
    }
    
    subscript (key: Int) -> Element {
        get {
            return ptr[key]
        }
        
        set (newValue) {
            ptr[key] = newValue
        }
    }
}
