//
//  Util.swift
//  DL4S
//
//  Created by Palle Klewitz on 07.03.19.
//

import Foundation


extension Sequence {
    func count(where predicate: (Element) throws -> Bool) rethrows -> Int {
        return try lazy.filter(predicate).count
    }
}

func iterate(_ shape: [Int]) -> AnySequence<[Int]> {
    func increment<S: RandomAccessCollection>(_ list: S, shape: S) -> ([Int], Bool) where S.Element == Int {
        guard let first = list.first, let firstDim = shape.first else {
            return ([], true)
        }
        let (rest, overflows) = increment(list.dropFirst(), shape: shape.dropFirst())
        if overflows {
            if first + 1 >= firstDim {
                return ([0] + rest, true)
            } else {
                return ([first + 1] + rest, false)
            }
        } else {
            return ([first] + rest, false)
        }
    }
    
    return AnySequence(sequence(first: Array(repeating: 0, count: shape.count)) { state -> [Int]? in
        let (incremented, overflows) = increment(state, shape: shape)
        return overflows ? nil : incremented
    })
}

prefix func ! <Parameters>(predicate: @escaping (Parameters) -> Bool) -> (Parameters) -> Bool {
    return { params in
        !predicate(params)
    }
}

extension Slice: Equatable where Element: Hashable {
    public static func == (lhs: Slice<Base>, rhs: Slice<Base>) -> Bool {
        return lhs.count == rhs.count && !zip(lhs, rhs).map(==).contains(false)
    }
}

extension Slice: Hashable where Element: Hashable {
    
    
    public func hash(into hasher: inout Hasher) {
        for element in self {
            hasher.combine(element)
        }
    }
}
