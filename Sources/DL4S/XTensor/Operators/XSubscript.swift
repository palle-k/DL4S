//
//  XSubscript.swift
//  DL4S
//
//  Created by Palle Klewitz on 15.10.19.
//

import Foundation


public extension XTensor {
    subscript(index: [Int?]) -> XTensor<Element, Device> {
        get {
            let index = zip(index, shape).map { idx, dim -> Int? in
                if let idx = idx, idx < 0 {
                    return dim + idx
                } else {
                    return idx
                }
            }
            let (val, isCopy, shape) = Device.Memory.get(slice: index, of: values.values, with: self.shape)
            let handle = XTensorHandle(values: val, parent: isCopy ? nil : self.handle)
            
            return XTensor(
                handle: handle,
                shape: shape,
                context: requiresGradient ? XTensorContext(
                    tag: "SubscriptRead",
                    sources: [self],
                    backpropagate: [{ resultGradient in
                        var result = XTensor<Element, Device>(repeating: 0, shape: self.shape)
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
                self.context = XTensorContext(
                    tag: "SubscriptWrite",
                    sources: [slice],
                    backpropagate: [{ resultGradient in
                        resultGradient[index]
                    }]
                )
            }
        }
    }
    
    subscript(index: Int?...) -> XTensor<Element, Device> {
        get {self[index]}
        set (slice) {self[index] = slice}
    }
}
