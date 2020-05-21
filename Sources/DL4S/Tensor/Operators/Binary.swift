//
//  Binary.swift
//  DL4S
//
//  Created by Palle Klewitz on 03.10.19.
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

//MARK: Broadcasting Operators

public extension Tensor {
    
    /// Element-wise broadcast adds the given tensors
    ///
    /// lhs and rhs must have matching shapes, such that dimensions of the shape are either equal or 1.
    /// Shapes are matched from the right. For example, the shapes [42, 3, 1] and [3, 8] can be broadcasted and will give a
    /// tensor with the result shape [42, 3, 8].
    ///
    /// For detailed broadcasting rules, follow the [numpy documentation](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    ///
    /// - Parameters:
    ///   - lhs: First tensor
    ///   - rhs: Second tensor
    /// - Returns: Broadcast added result
    static func + (lhs: Self, rhs: Self) -> Self {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        let resultValues = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        
        if lhs.shape == rhs.shape {
            Device.Engine.vAdd(lhs: lhs.values.values, rhs: rhs.values.values, result: resultValues.values, count: lhs.count)
        } else {
            Device.Engine.broadcastAdd(lhs: lhs.values, rhs: rhs.values, result: resultValues)
        }
        
        if lhs.requiresGradient || rhs.requiresGradient {
            func grad(a: Self, b: Self, grad: Self) -> Self {
                OperationGroup.capture(named: "∇+") {
                    let aPadded = Array(repeating: 1, count: grad.dim - a.dim) + a.shape
                    let aReducedAxes = zip(aPadded, grad.shape).enumerated()
                        .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                    
                    var tmpReducedShape = aPadded
                    
                    for a in aReducedAxes.reversed() {
                        tmpReducedShape.remove(at: a)
                    }
                    
                    let reduced = grad
                        .reduceSum(along: aReducedAxes)
                        .view(as: a.shape)
                    return reduced
                }
            }
            
            let resultContext = TensorContext<Element, Device>(
                tag: "+",
                sources: [lhs, rhs],
                backpropagate: [
                    { vectorGradient in
                        grad(a: lhs, b: rhs, grad: vectorGradient)
                    }, { vectorGradient in
                        grad(a: rhs, b: lhs, grad: vectorGradient)
                    }
                ]
            )
            
            return Tensor(using: resultValues, context: resultContext)
        } else {
            return Tensor(using: resultValues, context: nil)
        }
    }
    
    /// Element-wise broadcast multiplies the given tensors
    ///
    /// lhs and rhs must have matching shapes, such that dimensions of the shape are either equal or 1.
    /// Shapes are matched from the right. For example, the shapes [42, 3, 1] and [3, 8] can be broadcasted and will give a
    /// tensor with the result shape [42, 3, 8].
    ///
    /// For detailed broadcasting rules, follow the [numpy documentation](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    ///
    /// - Parameters:
    ///   - lhs: First tensor
    ///   - rhs: Second tensor
    /// - Returns: Broadcast multiplied result
    static func * (lhs: Self, rhs: Self) -> Self {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        let resultValues = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        
        if lhs.shape == rhs.shape {
            Device.Engine.vMul(lhs: lhs.values.values, rhs: rhs.values.values, result: resultValues.values, count: lhs.count)
        } else {
            Device.Engine.broadcastMul(lhs: lhs.values, rhs: rhs.values, result: resultValues)
        }
        
        if lhs.requiresGradient || rhs.requiresGradient {
            func grad(a: Self, b: Self, grad: Self) -> Self {
                OperationGroup.capture(named: "∇⊙") {
                    let aPadded = Array(repeating: 1, count: grad.dim - a.dim) + a.shape
                    let aReducedAxes = zip(aPadded, grad.shape).enumerated()
                        .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                    
                    var tmp1reducedShape = aPadded
                    
                    for a in aReducedAxes.reversed() {
                        tmp1reducedShape.remove(at: a)
                    }
                    
                    return (b * grad).reduceSum(along: aReducedAxes).view(as: a.shape)
                }
            }
            
            let resultContext = TensorContext<Element, Device>(
                tag: "⊙",
                sources: [lhs, rhs],
                backpropagate: [
                    { vectorGradient in
                        grad(a: lhs, b: rhs, grad: vectorGradient)
                    }, { vectorGradient in
                        grad(a: rhs, b: lhs, grad: vectorGradient)
                    }
                ]
            )
            
            return Tensor(using: resultValues, context: resultContext)
        } else {
            return Tensor(using: resultValues, context: nil)
        }
    }
    
    /// Element-wise broadcast subtracts the given tensors
    ///
    /// lhs and rhs must have matching shapes, such that dimensions of the shape are either equal or 1.
    /// Shapes are matched from the right. For example, the shapes [42, 3, 1] and [3, 8] can be broadcasted and will give a
    /// tensor with the result shape [42, 3, 8].
    ///
    /// For detailed broadcasting rules, follow the [numpy documentation](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    ///
    /// - Parameters:
    ///   - lhs: First tensor
    ///   - rhs: Second tensor
    /// - Returns: Broadcast  difference
    static func - (lhs: Self, rhs: Self) -> Self {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        
        if lhs.shape == rhs.shape {
            Device.Engine.vSub(lhs: lhs.values.values, rhs: rhs.values.values, result: resultBuffer.values, count: lhs.count)
        } else {
            Device.Engine.broadcastSub(lhs: lhs.values, rhs: rhs.values, result: resultBuffer)
        }
        
        if lhs.requiresGradient || rhs.requiresGradient {
            let resultContext = TensorContext(
                tag: "-",
                sources: [lhs, rhs],
                backpropagateAccumulate: [
                    { resultGradient, acc in
                        OperationGroup.capture(named: "∇₁-") {
                            let lhsPadded = Array(repeating: 1, count: resultGradient.dim - lhs.dim) + lhs.shape
                            let lhsReducedAxes = zip(lhsPadded, resultGradient.shape).enumerated()
                                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                            
                            var tmpReducedShape = lhsPadded
                            
                            for a in lhsReducedAxes.reversed() {
                                tmpReducedShape.remove(at: a)
                            }
                            
                            if let acc = acc {
                                return acc + resultGradient.reduceSum(along: lhsReducedAxes).view(as: lhs.shape)
                            } else {
                                return resultGradient.reduceSum(along: lhsReducedAxes).view(as: lhs.shape)
                            }
                            
                        }
                    }, { resultGradient, acc in
                        OperationGroup.capture(named: "∇₂-") {
                            let rhsPadded = Array(repeating: 1, count: resultGradient.dim - rhs.dim) + rhs.shape
                            let rhsReducedAxes = zip(rhsPadded, resultGradient.shape).enumerated()
                                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                            
                            var tmpReducedShape = rhsPadded
                            
                            for a in rhsReducedAxes.reversed() {
                                tmpReducedShape.remove(at: a)
                            }
                            
                            if let acc = acc {
                                return acc - resultGradient.reduceSum(along: rhsReducedAxes).view(as: rhs.shape)
                            } else {
                                return -resultGradient.reduceSum(along: rhsReducedAxes).view(as: rhs.shape)
                            }
                        }
                    }
                ]
            )
             
            return Tensor(using: resultBuffer, context: resultContext)
        } else {
            return Tensor(using: resultBuffer, context: nil)
        }
    }
    
    /// Element-wise broadcast divides the given tensors
    ///
    /// lhs and rhs must have matching shapes, such that dimensions of the shape are either equal or 1.
    /// Shapes are matched from the right. For example, the shapes [42, 3, 1] and [3, 8] can be broadcasted and will give a
    /// tensor with the result shape [42, 3, 8].
    ///
    /// For detailed broadcasting rules, follow the [numpy documentation](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    ///
    /// - Parameters:
    ///   - lhs: First tensor
    ///   - rhs: Second tensor
    /// - Returns: Broadcast quotient
    static func / (lhs: Self, rhs: Self) -> Self {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        
        if lhs.shape == rhs.shape {
            Device.Engine.vDiv(lhs: lhs.values.values, rhs: rhs.values.values, result: resultBuffer.values, count: lhs.count)
        } else {
            Device.Engine.broadcastDiv(lhs: lhs.values, rhs: rhs.values, result: resultBuffer)
        }
        
        if lhs.requiresGradient || rhs.requiresGradient {
            let context = TensorContext(
                tag: "÷",
                sources: [lhs, rhs],
                backpropagateAccumulate: [
                    { resultGradient, acc -> Self in
                        OperationGroup.capture(named: "∇₁÷") {
                            let lhsPadded = Array(repeating: 1, count: resultGradient.dim - lhs.dim) + lhs.shape
                            let lhsReducedAxes = zip(lhsPadded, resultGradient.shape).enumerated()
                                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                            
                            var tmp1reducedShape = lhsPadded
                            
                            for a in lhsReducedAxes.reversed() {
                                tmp1reducedShape.remove(at: a)
                            }
                            
                            let d = resultGradient / rhs
                            if let acc = acc {
                                return acc + d.reduceSum(along: lhsReducedAxes).view(as: lhs.shape)
                            } else {
                                return d.reduceSum(along: lhsReducedAxes).view(as: lhs.shape)
                            }
                            
                        }
                    }, { resultGradient, acc -> Self in
                        OperationGroup.capture(named: "∇₂÷") {
                            let rhsPadded = Array(repeating: 1, count: resultGradient.dim - rhs.dim) + rhs.shape
                            let rhsReducedAxes = zip(rhsPadded, resultGradient.shape).enumerated()
                                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                            
                            var tmp1reducedShape = rhsPadded
                            
                            for a in rhsReducedAxes.reversed() {
                                tmp1reducedShape.remove(at: a)
                            }
                            
                            let m = resultGradient * lhs
                            let d = m / (rhs * rhs)
                            if let acc = acc {
                                return acc - d.reduceSum(along: rhsReducedAxes).view(as: rhs.shape)
                            } else {
                                return -d.reduceSum(along: rhsReducedAxes).view(as: rhs.shape)
                            }
                        }
                    }
                ]
            )
            
            return Tensor(using: resultBuffer, context: context)
        } else {
            return Tensor(using: resultBuffer, context: nil)
        }
    }
    
    
    /// Negates every element of the given tensor.
    ///
    /// - Parameter value: Tensor to negate
    /// - Returns: Negated tensor
    static prefix func - (value: Self) -> Self {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: value.shape, type: Element.self)
        Device.Engine.vNeg(val: value.values.values, result: resultBuffer.values, count: value.count)
        
        return Tensor(
            using: resultBuffer,
            context: value.requiresGradient ? TensorContext(
                tag: "neg",
                sources: [value],
                backpropagate: [{ resultGradient in
                    -resultGradient
                }]
            ) : nil
        )
    }
    
    /// In-place broadcast adds the given tensors.
    /// This operation requires the resulting broadcast shape to be equivalent to the shape of lhs.
    ///
    /// For detailed broadcasting rules, follow the [numpy documentation](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    ///
    /// - Parameters:
    ///   - lhs: Tensor to update
    ///   - rhs: Tensor to add to lhs
    static func += (lhs: inout Self, rhs: Self) {
        let originalShape = lhs.shape
        #if DEBUG
        let tag = lhs.tag
        #endif
        lhs = lhs + rhs
        #if DEBUG
        lhs.tag = tag
        #endif
        assert(originalShape == lhs.shape, "In-place addition has modified shape.")
    }
    
    /// In-place broadcast subtracts the given tensors.
    /// This operation requires the resulting broadcast shape to be equivalent to the shape of lhs.
    ///
    /// For detailed broadcasting rules, follow the [numpy documentation](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    ///
    /// - Parameters:
    ///   - lhs: Tensor to update
    ///   - rhs: Tensor to subtract from lhs
    static func -= (lhs: inout Self, rhs: Self) {
        let originalShape = lhs.shape
        #if DEBUG
        let tag = lhs.tag
        #endif
        lhs = lhs - rhs
        #if DEBUG
        lhs.tag = tag
        #endif
        assert(originalShape == lhs.shape, "In-place subtraction has modified shape.")
    }
    
    /// In-place broadcast multiplies the given tensors.
    /// This operation requires the resulting broadcast shape to be equivalent to the shape of lhs.
    ///
    /// For detailed broadcasting rules, follow the [numpy documentation](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    ///
    /// - Parameters:
    ///   - lhs: Tensor to update
    ///   - rhs: Tensor to multiply with lhs
    static func *= (lhs: inout Self, rhs: Self) {
        let originalShape = lhs.shape
        #if DEBUG
        let tag = lhs.tag
        #endif
        lhs = lhs * rhs
        #if DEBUG
        lhs.tag = tag
        #endif
        assert(originalShape == lhs.shape, "In-place multiplication has modified shape.")
    }
    
    /// In-place broadcast divides the given tensors.
    /// This operation requires the resulting broadcast shape to be equivalent to the shape of lhs.
    ///
    /// For detailed broadcasting rules, follow the [numpy documentation](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    ///
    /// - Parameters:
    ///   - lhs: Tensor to update
    ///   - rhs: Tensor to divide lhs with
    static func /= (lhs: inout Self, rhs: Self) {
        let originalShape = lhs.shape
        #if DEBUG
        let tag = lhs.tag
        #endif
        lhs = lhs / rhs
        #if DEBUG
        lhs.tag = tag
        #endif
        assert(originalShape == lhs.shape, "In-place division has modified shape.")
    }
    
    /// Performs a broadcasted exponentiation between self (base) and power (exponent).
    ///
    /// For detailed broadcasting rules, follow the [numpy documentation](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    ///
    /// - Parameters:
    ///   - power: Exponent
    /// - Returns: self broadcast exponentiated by power
    func raised(toPowerOf power: Self) -> Self {
        (self.log() * power).exp()
    }
    
    /// Computes the elementwise maxima between the given tensors
    ///
    /// - Parameters:
    ///   - first: First tensor
    ///   - second: Second tensor
    /// - Returns: Element wise maxima between first and second value tensors
    static func max(_ first: Self, _ second: Self) -> Self {
        precondition(first.shape == second.shape, "Shapes must be equal")
        let resultBuffer = Device.Memory.allocateBuffer(withShape: first.shape, type: Element.self)
        
        if first.requiresGradient || second.requiresGradient {
            let contextBuffer = Device.Memory.allocateBuffer(withShape: first.shape, type: Element.self)
            Device.Engine.max(first.values, second.values, result: resultBuffer, context: contextBuffer)
            let contextTensor = Tensor(using: contextBuffer, context: nil)
            
            let context = TensorContext(
                tag: "max",
                sources: [first, second],
                backpropagate: [
                    { targetGrad in
                        targetGrad * (1 - contextTensor)
                    },
                    { targetGrad in
                        targetGrad * contextTensor
                    }
                ]
            )
            
            return Tensor(using: resultBuffer, context: context)
        } else {
            Device.Engine.max(first.values, second.values, result: resultBuffer)
            return Tensor(using: resultBuffer, context: nil)
        }
    }
    
    /// Computes the elementwise minima between the given tensors
    ///
    /// - Parameters:
    ///   - first: First tensor
    ///   - second: Other tensors
    /// - Returns: Element wise minima between first and second value tensors
    static func min(_ first: Self, _ second: Self) -> Self {
        precondition(first.shape == second.shape, "Shapes must be equal")
        let resultBuffer = Device.Memory.allocateBuffer(withShape: first.shape, type: Element.self)
        
        if first.requiresGradient || second.requiresGradient {
            let contextBuffer = Device.Memory.allocateBuffer(withShape: first.shape, type: Element.self)
            Device.Engine.min(first.values, second.values, result: resultBuffer, context: contextBuffer)
            let contextTensor = Tensor(using: contextBuffer, context: nil)
            
            let context = TensorContext(
                tag: "max",
                sources: [first, second],
                backpropagate: [
                    { targetGrad in
                        targetGrad * (1 - contextTensor)
                    },
                    { targetGrad in
                        targetGrad * contextTensor
                    }
                ]
            )
            
            return Tensor(using: resultBuffer, context: context)
        } else {
            Device.Engine.min(first.values, second.values, result: resultBuffer)
            return Tensor(using: resultBuffer, context: nil)
        }
    }
}
