//
//  PackedSequence.swift
//  DL4S
//
//  Created by Palle Klewitz on 03.05.19.
//

import Foundation


public class PackedSequence<Element: NumericType, Device: DeviceType> {
    let tensor: Tensor<Element, Device>
    let lengths: [Int]
    let offsets: [Int]
    
    public var tensors: [Tensor<Element, Device>] {
        return zip(offsets, lengths).map { offset, length in
            tensor[offset ..< (offset + length)]
        }
    }
    
    public init(from tensor: Tensor<Element, Device>, withLengths lengths: [Int]) {
        precondition(tensor.dim >= 1, "Tensor must be at least 1-dimensional")
        precondition(lengths.reduce(0, +) == tensor.shape[0], "Length of packed sequences must sum up to tensor shape[0]")
        
        self.tensor = tensor
        self.lengths = lengths
        self.offsets = lengths
            .reduce(into: [0], {$0.append($0.last! + $1)})
            .dropLast()
    }
    
    public init(of tensors: [Tensor<Element, Device>]) {
        self.tensor = stack(tensors)
        self.lengths = tensors.map {$0.shape[0]}
        self.offsets = lengths
            .reduce(into: [0], {$0.append($0.last! + $1)})
            .dropLast()
    }
    
    public subscript(index: Int) -> Tensor<Element, Device> {
        let availableTensorOffsets = zip(offsets, lengths)
            .filter {index < $1}
            .map {$0.0}
        
        return stack(availableTensorOffsets.map { offset in
            self.tensor[offset + index].unsqueeze(at: 0)
        })
    }
}
