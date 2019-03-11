//
//  Engine.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

import Foundation


public protocol Device {
    associatedtype MemoryOperatorType: MemoryOperators where MemoryOperatorType.DeviceType == Self
    associatedtype EngineType: Engine where EngineType.DeviceType == Self
}


public protocol MemoryOperators {
    associatedtype DeviceType: Device where DeviceType.MemoryOperatorType == Self
    associatedtype RawBufferType: Hashable
    
    static func allocateBuffer<Element>(withCapacity: Int, type: Element.Type) -> Buffer<Element, DeviceType>
    static func free<Element>(_ buffer: Buffer<Element, DeviceType>)
    
    static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, DeviceType>, count: Int)
    static func assign<Element>(from source: Buffer<Element, DeviceType>, to destination: Buffer<Element, DeviceType>, count: Int)
    static func assign<Element>(from source: Buffer<Element, DeviceType>, to destination: UnsafeMutableBufferPointer<Element>, count: Int)
    
    static func getValue<Element>(from source: Buffer<Element, DeviceType>) -> Element
    static func getSize<Element>(of buffer: Buffer<Element, DeviceType>) -> Int
    
    static func get<Element>(slice: [Int?], of buffer: Buffer<Element, DeviceType>, with shape: [Int]) -> (Buffer<Element, DeviceType>, Bool, [Int])
    static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, DeviceType>, with shape: [Int]) -> (Buffer<Element, DeviceType>, Bool, [Int])
    static func set<Element>(slice: [Int?], of buffer: Buffer<Element, DeviceType>, with dstShape: [Int], from source: Buffer<Element, DeviceType>, with sourceShape: [Int])
    static func set<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, DeviceType>, with dstShape: [Int], from source: Buffer<Element, DeviceType>, with sourceShape: [Int])
}


public protocol Engine {
    associatedtype DeviceType: Device where DeviceType.EngineType == Self
    
    static func sqrt<N: NumericType>(_ value: N) -> N
    static func log<N: NumericType>(_ value: N) -> N
    static func exp<N: NumericType>(_ value: N) -> N
    static func sin<N: NumericType>(_ value: N) -> N
    static func cos<N: NumericType>(_ value: N) -> N
    static func tan<N: NumericType>(_ value: N) -> N
    static func sinh<N: NumericType>(_ value: N) -> N
    static func cosh<N: NumericType>(_ value: N) -> N
    static func tanh<N: NumericType>(_ value: N) -> N
    static func pow<N: NumericType>(base: N, exponent: N) -> N
    
    static func fill<N: NumericType>(value: N, result: Buffer<N, DeviceType>, count: Int)
    static func fill<N: NumericType>(value: N, result: Buffer<N, DeviceType>, stride: Int, count: Int)
    static func transpose<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, srcRows: Int, srcCols: Int)
    static func vAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func vsAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, result: Buffer<N, DeviceType>, count: Int)
    static func vNeg<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func vSub<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func vMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func vMA<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, add: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func vsMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, result: Buffer<N, DeviceType>, count: Int)
    static func vDiv<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func svDiv<N: NumericType>(lhs: N, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func vSquare<N: NumericType>(values: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func matMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, lhsRows: Int, lhsCols: Int, rhsCols: Int)
    static func matMulAddInPlace<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), transposeFirst: Bool, transposeSecond: Bool)
    static func dot<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, count: Int) -> N
    static func vMulSA<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, add: N, result: Buffer<N, DeviceType>, count: Int)
    static func vsMulVAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, add: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func log<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func exp<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func relu<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func isPositive<N: NumericType>(val: Buffer<N, DeviceType>, result: UnsafeMutablePointer<N>, count: Int)
    static func tanh<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func sqrt<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func sum<N: NumericType>(val: Buffer<N, DeviceType>, count: Int) -> N
    static func copysign<N: NumericType>(values: Buffer<N, DeviceType>, signs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int)
    static func argmax<N: NumericType>(values: Buffer<N, DeviceType>, count: Int) -> (Int, N)
    static func conv2d<N: NumericType>(input: Buffer<N, DeviceType>, filter: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, width: Int, height: Int, kernelWidth: Int, kernelHeight: Int, kernelDepth: Int, kernelCount: Int)
}

extension Engine {
    public static func sqrt<N: NumericType>(_ value: N) -> N {
        return value.sqrt()
    }
    
    public static func log<N: NumericType>(_ value: N) -> N {
        return value.log()
    }
    
    public static func exp<N: NumericType>(_ value: N) -> N {
        return value.exp()
    }
    
    public static func sin<N: NumericType>(_ value: N) -> N {
        return value.sin()
    }
    
    public static func cos<N: NumericType>(_ value: N) -> N {
        return value.cos()
    }
    
    public static func tan<N: NumericType>(_ value: N) -> N {
        return value.tan()
    }
    
    public static func sinh<N: NumericType>(_ value: N) -> N {
        return value.sinh()
    }
    
    public static func cosh<N: NumericType>(_ value: N) -> N {
        return value.cosh()
    }
    
    public static func tanh<N: NumericType>(_ value: N) -> N {
        return value.tanh()
    }
    
    public static func pow<N: NumericType>(base: N, exponent: N) -> N {
        return N.pow(base: base, exponent: exponent)
    }
}


