//
//  VecOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
//

import Foundation

fileprivate struct ReshapeOperation<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        if !Vector.sameIdentity(source, vector.parent) {
            Element.vAdd(lhs: vector.gradient, rhs: source.gradient, result: source.gradient, count: source.count)
        }
        source._backwards()
    }
}

public extension Vector {
    func view(as shape: Int...) -> Vector<Element> {
        precondition(shape.count(where: {$0 == -1}) <= 1, "The size of at most one dimension can be unknown (-1).")
        precondition(shape.allSatisfy {$0 >= -1}, "All dimensions must be greater than or equal to -1.")
        
        var shape = shape
        if let idx = shape.firstIndex(of: -1) {
            let remaining = count / shape.lazy.filter {$0 >= 0}.reduce(1, *)
            shape[idx] = remaining
        }
        
        return Vector(values: values, gradient: gradient, shape: shape, parent: self, context: ReshapeOperation(source: self).asAny())
    }
    
}

fileprivate struct ReplaceOperation<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    let location: [Int?]
    
    func backwards(from vector: Vector<Element>) {
        
    }
}

fileprivate struct SelectOperation<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    let location: [Int?]
    
    func backwards(from vector: Vector<Element>) {
        let (buffer, isCopy, _) = source.gradient(from: location)
        Element.vAdd(lhs: buffer, rhs: vector.gradient, result: buffer, count: vector.count)
        
        if isCopy {
            source.setGradient(at: location, source: buffer, sourceShape: vector.shape)
        }
        source._backwards()
    }
}

fileprivate struct RangeReplaceOperation<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    let location: [Range<Int>?]
    
    func backwards(from vector: Vector<Element>) {
        
    }
}

fileprivate struct RangeSelectOperation<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    let location: [Range<Int>?]
    
    func backwards(from vector: Vector<Element>) {
        let (buffer, isCopy, _) = source.gradient(from: location)
        Element.vAdd(lhs: buffer, rhs: vector.gradient, result: buffer, count: vector.count)
        
        if isCopy {
            source.setGradient(at: location, source: buffer, sourceShape: vector.shape)
        }
        source._backwards()
    }
}

public extension Vector {
    subscript(index: Int?...) -> Vector<Element> {
        get {
            let (val, isCopy, shape) = MemoryOps.get(slice: index, of: values, with: self.shape)
            let (grad, _, _) = MemoryOps.get(slice: index, of: gradient, with: self.shape)
            return Vector(
                values: val,
                gradient: grad,
                shape: shape,
                parent: isCopy ? nil : self,
                context: SelectOperation(source: self, location: index).asAny()
            )
        }
        set (slice) {
            let origin = Vector(values: values, gradient: gradient, shape: shape, parent: self.parent, context: context)
            self.parent = origin
            self.ensureOwnership()
            
            if slice.dim == 0 && dim - index.filter({$0 != nil}).count > 0 {
                fatalError("Assigning from a single value not supported yet.")
            }
            
            MemoryOps.set(slice: index, of: values, with: shape, from: slice.values, with: slice.shape)
            MemoryOps.set(slice: index, of: gradient, with: shape, from: slice.gradient, with: slice.shape)
            
            self.context = ReplaceOperation(source: origin, location: index).asAny()
        }
    }
}


public extension Vector {
    subscript(index: Range<Int>?...) -> Vector<Element> {
        get {
            let (val, isCopy, shape) = MemoryOps.get(slice: index, of: values, with: self.shape)
            let (grad, _, _) = MemoryOps.get(slice: index, of: gradient, with: self.shape)
            return Vector(
                values: val,
                gradient: grad,
                shape: shape,
                parent: isCopy ? nil : self,
                context: RangeSelectOperation(source: self, location: index).asAny()
            )
        }
        set (slice) {
            let origin = Vector(values: values, gradient: gradient, shape: shape, parent: self.parent, context: context)
            self.parent = origin
            self.ensureOwnership()
            
            if slice.dim == 0 && dim - index.filter({$0 != nil}).count > 0 {
                fatalError("Assigning from a single value not supported yet.")
            }
            
            MemoryOps.set(slice: index, of: values, with: shape, from: slice.values, with: slice.shape)
            MemoryOps.set(slice: index, of: gradient, with: shape, from: slice.gradient, with: slice.shape)
            
            self.context = RangeReplaceOperation(source: origin, location: index).asAny()
        }
    }
}


