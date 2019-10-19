//
//  VecOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
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

fileprivate struct ReplaceOperation<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    let location: [Int?]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        fatalError("\(#function) is not implemented.")
    }
    
    var symbol: String {
        return "IndexReplace"
    }
}

fileprivate struct SelectOperation<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    let location: [Int?]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        guard let (buffer, isCopy, _) = source.gradient(from: location) else {
            return
        }
        
        if isCopy {
            Device.Engine.vAdd(lhs: buffer, rhs: vectorGradient, result: buffer, count: vector.count)
            source.setGradient(at: location, source: buffer, sourceShape: vector.shape)
        }
    }
    
    var symbol: String {
        return "IndexSelect"
    }
}

fileprivate struct RangeReplaceOperation<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    let location: [Range<Int>?]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        fatalError("\(#function) is not implemented.")
    }
    
    var symbol: String {
        return "RangeReplace"
    }
}

fileprivate struct RangeSelectOperation<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var source: Tensor<Element, Device>
    
    let location: [Range<Int>?]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        guard let (buffer, isCopy, _) = source.gradient(from: location) else {
            return
        }
        Device.Engine.vAdd(lhs: buffer, rhs: vectorGradient, result: buffer, count: vector.count)
        
        if isCopy {
            source.setGradient(at: location, source: buffer, sourceShape: vector.shape)
        }
    }
    
    var symbol: String {
        return "RangeSelect"
    }
}

public extension Tensor {
    subscript(index: [Int?]) -> Tensor<Element, Device> {
        get {
            let index = zip(index, shape).map { idx, dim -> Int? in
                if let idx = idx, idx < 0 {
                    return dim + idx
                } else {
                    return idx
                }
            }
            let (val, isCopy, shape) = Device.Memory.get(slice: index, of: values, with: self.shape)
            let grad: Buffer<Element, Device>?
            if let gradient = self.gradient {
                let (g, _, _) = Device.Memory.get(slice: index, of: gradient, with: self.shape)
                grad = g
            } else {
                grad = nil
            }
            return Tensor(
                values: val,
                gradient: grad,
                shape: shape,
                parent: isCopy ? nil : self,
                context: requiresGradient ? SelectOperation(source: self, location: index).asAny() : nil
            )
        }
        set (slice) {
            let index = zip(index, shape).map { idx, dim -> Int? in
                if let idx = idx, idx < 0 {
                    return dim + idx
                } else {
                    return idx
                }
            }
            if slice.dim == 0 && dim - index.filter({$0 != nil}).count > 0 {
                fatalError("Assigning from a single value not supported yet.")
            }
            
            Device.Memory.set(slice: index, of: values, with: shape, from: slice.values, with: slice.shape)
            if let gradient = self.gradient, let sliceGradient = slice.gradient {
                Device.Memory.set(slice: index, of: gradient, with: shape, from: sliceGradient, with: slice.shape)
            }
            self.context = ReplaceOperation(source: slice, location: index).asAny()
        }
    }
    
    subscript(index: Int?...) -> Tensor<Element, Device> {
        get {
            return self[index]
        }
        set (slice) {
            self[index] = slice
        }
    }
}


public extension Tensor {
    subscript(index: [Range<Int>?]) -> Tensor<Element, Device> {
        get {
            let (val, isCopy, shape) = Device.Memory.get(slice: index, of: values, with: self.shape)
            let grad: Buffer<Element, Device>?
            if let gradient = self.gradient {
                let (g, _, _) = Device.Memory.get(slice: index, of: gradient, with: self.shape)
                grad = g
            } else {
                grad = nil
            }
            return Tensor(
                values: val,
                gradient: grad,
                shape: shape,
                parent: isCopy ? nil : self,
                context: requiresGradient ? RangeSelectOperation(source: self, location: index).asAny() : nil
            )
        }
        set (slice) {
            if slice.dim == 0 && dim - index.filter({$0 != nil}).count > 0 {
                fatalError("Assigning from a single value not supported yet.")
            }
            
            Device.Memory.set(slice: index, of: values, with: shape, from: slice.values, with: slice.shape)
            if let gradient = self.gradient, let sliceGradient = slice.gradient {
                Device.Memory.set(slice: index, of: gradient, with: shape, from: sliceGradient, with: slice.shape)
            }
            self.context = RangeReplaceOperation(source: slice, location: index).asAny()
        }
    }
    
    subscript(index: Range<Int>?...) -> Tensor<Element, Device> {
        get {
            return self[index]
        }
        set (slice) {
            self[index] = slice
        }
    }
}


public extension Tensor {
    func squeeze() -> Tensor<Element, Device> {
        return self.view(as: shape.filter {$0 != 1})
    }
    
    func squeeze(at axis: Int) -> Tensor<Element, Device> {
        guard shape[axis] == 1 else {
            return self
        }
        var newShape = shape
        newShape.remove(at: axis)
        return self.view(as: newShape)
    }
    
    func unsqueeze(at index: Int) -> Tensor<Element, Device> {
        var newShape = shape
        newShape.insert(1, at: index)
        
        return self.view(as: newShape)
    }
}
