//
//  Subscript.swift
//  DL4S
//
//  Created by Palle Klewitz on 15.10.19.
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

//MARK: Subscripting

public extension Tensor {
    /// Gets or sets a subtensor at the given index.
    ///
    /// When an element of the index is nil, all elements along the corresponding axis are read or written.
    ///
    /// Example:
    /// ```
    /// let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
    /// print(a[nil, 1]) // [2, 5]
    /// print(a[1]) // [4, 5, 6]
    /// print(a[1, nil] == a[1]) // true
    /// ```
    subscript(index: [Int?]) -> Self {
        get {
            let index = zip(index, shape).map { idx, dim -> Int? in
                if let idx = idx, idx < 0 {
                    return dim + idx
                } else {
                    return idx
                }
            }
            let (val, isCopy, shape) = Device.Memory.get(slice: index, of: values.values, with: self.shape)
            let handle = TensorHandle(values: val, parent: isCopy ? nil : self.handle)
            
            return Tensor(
                handle: handle,
                shape: shape,
                context: requiresGradient ? TensorContext(
                    tag: "read",
                    sources: [self],
                    backpropagateAccumulate: [{ resultGradient, acc in
                        var result = (acc ?? Self(repeating: 0, shape: self.shape))
                        result[index] = resultGradient
                        return result
                    }]
                ) : nil
            )
        }
        
        set (slice) {
            precondition(!requiresGradient, "Cannot write into tensor that requires gradient.")
            
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
            
            //TODO: Proper handling of replacement when gradient is computed.
            
            Device.Memory.set(slice: index, of: values.values, with: shape, from: slice.values.values, with: slice.shape)
            
            if slice.requiresGradient {
                self.requiresGradient = true
                self.context = TensorContext(
                    tag: "write",
                    sources: [slice],
                    backpropagate: [{ resultGradient in
                        resultGradient[index]
                    }]
                )
            }
        }
    }
    
    /// Gets or sets a subtensor at the given index.
    ///
    /// When an element of the index is nil, all elements along the corresponding axis are read or written.
    ///
    /// Example:
    /// ```
    /// let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
    /// print(a[nil, 1]) // [2, 5]
    /// print(a[1]) // [4, 5, 6]
    /// print(a[1, nil] == a[1]) // true
    /// ```
    subscript(index: Int?...) -> Self {
        get {self[index]}
        set (slice) {self[index] = slice}
    }
    
    /// Gets or sets a subtensor at the given window.
    ///
    /// When an element of the index is nil, all elements along the corresponding axis are read or written.
    ///
    /// Example:
    /// ```
    /// let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
    /// print(a[0 ..< 2]) // [[1, 2], [4, 5]]
    /// print(a[nil, 0 ..< 1]) // [[1, 2, 3]]
    /// ```
    subscript(index: [Range<Int>?]) -> Self {
        get {
            let (val, isCopy, shape) = Device.Memory.get(slice: index, of: values.values, with: self.shape)
            
            let handle: TensorHandle<Element, Device>
            if isCopy {
                handle = TensorHandle(values: val)
            } else {
                handle = TensorHandle(values: val, parent: self.handle)
            }
            
            return Tensor(
                handle: handle,
                shape: shape,
                context: requiresGradient ? TensorContext(
                    tag: "SubscriptRangeRead",
                    sources: [self],
                    backpropagateAccumulate: [{ resultGradient, acc in
                        var result = (acc ?? Self(repeating: 0, shape: self.shape))
                        result[index] = resultGradient
                        return result
                    }]
                ) : nil
            )
        }
        
        set (slice) {
            if slice.dim == 0 && dim - index.filter({$0 != nil}).count > 0 {
                fatalError("Assigning from a single value not supported yet.")
            }
            
            //TODO: Proper handling of replacement when gradient is computed.
            
            Device.Memory.set(slice: index, of: values.values, with: shape, from: slice.values.values, with: slice.shape)
            
            if slice.requiresGradient {
                self.requiresGradient = true
                self.context = TensorContext(
                    tag: "SubscriptRangeWrite",
                    sources: [slice],
                    backpropagate: [{ resultGradient in
                        resultGradient[index]
                    }]
                )
            }
        }
    }
    
    /// Gets or sets a subtensor at the given window.
    ///
    /// When an element of the index is nil, all elements along the corresponding axis are read or written.
    ///
    /// Example:
    /// ```
    /// let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
    /// print(a[0 ..< 2]) // [[1, 2], [4, 5]]
    /// print(a[nil, 0 ..< 1]) // [[1, 2, 3]]
    /// ```
    subscript(index: Range<Int>?...) -> Self {
        get {self[index]}
        set (slice) {self[index] = slice}
    }
}
