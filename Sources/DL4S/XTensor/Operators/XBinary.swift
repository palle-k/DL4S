//
//  XBinary.swift
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

public extension XTensor {
    static func + (lhs: XTensor<Element, Device>, rhs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        let resultValues = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        
        Device.Engine.broadcastAdd(lhs: lhs.values, rhs: rhs.values, result: resultValues)
        
        if lhs.requiresGradient || rhs.requiresGradient {
            func grad(a: XTensor<Element, Device>, b: XTensor<Element, Device>, grad: XTensor<Element, Device>) -> XTensor<Element, Device> {
                OperationGroup.capture(named: "BroadcastAddGrad") {
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
            
            let resultContext = XTensorContext<Element, Device>(
                tag: "BroadcastAdd",
                sources: [lhs, rhs],
                backpropagate: [
                    { vectorGradient in
                        grad(a: lhs, b: rhs, grad: vectorGradient)
                    }, { vectorGradient in
                        grad(a: rhs, b: lhs, grad: vectorGradient)
                    }
                ]
            )
            
            return XTensor(using: resultValues, context: resultContext)
        } else {
            return XTensor(using: resultValues, context: nil)
        }
    }
    
    static func * (lhs: XTensor<Element, Device>, rhs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        let resultValues = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        
        Device.Engine.broadcastMul(lhs: lhs.values, rhs: rhs.values, result: resultValues)
        
        if lhs.requiresGradient || rhs.requiresGradient {
            func grad(a: XTensor<Element, Device>, b: XTensor<Element, Device>, grad: XTensor<Element, Device>) -> XTensor<Element, Device> {
                OperationGroup.capture(named: "BroadcastMultiplyGrad") {
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
            
            let resultContext = XTensorContext<Element, Device>(
                tag: "BroadcastMultiply",
                sources: [lhs, rhs],
                backpropagate: [
                    { vectorGradient in
                        grad(a: lhs, b: rhs, grad: vectorGradient)
                    }, { vectorGradient in
                        grad(a: rhs, b: lhs, grad: vectorGradient)
                    }
                ]
            )
            
            return XTensor(using: resultValues, context: resultContext)
        } else {
            return XTensor(using: resultValues, context: nil)
        }
    }
    
    static func - (lhs: XTensor<Element, Device>, rhs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.broadcastSub(lhs: lhs.values, rhs: rhs.values, result: resultBuffer)
        
        if lhs.requiresGradient || rhs.requiresGradient {
            let resultContext = XTensorContext(
                tag: "BroadcastSubtract",
                sources: [lhs, rhs],
                backpropagate: [
                    { resultGradient in
                        OperationGroup.capture(named: "BroadcastSubtractGradLhs") {
                            let lhsPadded = Array(repeating: 1, count: resultGradient.dim - lhs.dim) + lhs.shape
                            let lhsReducedAxes = zip(lhsPadded, resultGradient.shape).enumerated()
                                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                            
                            var tmpReducedShape = lhsPadded
                            
                            for a in lhsReducedAxes.reversed() {
                                tmpReducedShape.remove(at: a)
                            }
                            
                            return resultGradient.reduceSum(along: lhsReducedAxes).view(as: tmpReducedShape)
                        }
                    }, { resultGradient in
                        OperationGroup.capture(named: "BroadcastSubtractGradRhs") {
                            let rhsPadded = Array(repeating: 1, count: resultGradient.dim - rhs.dim) + rhs.shape
                            let rhsReducedAxes = zip(rhsPadded, resultGradient.shape).enumerated()
                                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                            
                            var tmpReducedShape = rhsPadded
                            
                            for a in rhsReducedAxes.reversed() {
                                tmpReducedShape.remove(at: a)
                            }
                            
                            return 0 - resultGradient.reduceSum(along: rhsReducedAxes).view(as: tmpReducedShape)
                        }
                    }
                ]
            )
             
            return XTensor(using: resultBuffer, context: resultContext)
        } else {
            return XTensor(using: resultBuffer, context: nil)
        }
    }
    
    static func / (lhs: XTensor<Element, Device>, rhs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.broadcastDiv(lhs: lhs.values, rhs: rhs.values, result: resultBuffer)
        
        if lhs.requiresGradient || rhs.requiresGradient {
            let context = XTensorContext(
                tag: "BroadcastDivide",
                sources: [lhs, rhs],
                backpropagate: [
                    { resultGradient -> XTensor<Element, Device> in
                        OperationGroup.capture(named: "BroadcastDivideLhs") {
                            let lhsPadded = Array(repeating: 1, count: resultGradient.dim - lhs.dim) + lhs.shape
                            let lhsReducedAxes = zip(lhsPadded, resultGradient.shape).enumerated()
                                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                            
                            var tmp1reducedShape = lhsPadded
                            
                            for a in lhsReducedAxes.reversed() {
                                tmp1reducedShape.remove(at: a)
                            }
                            
                            let d = resultGradient / rhs
                            return d.reduceSum(along: lhsReducedAxes).view(as: lhs.shape)
                        }
                    }, { resultGradient -> XTensor<Element, Device> in
                        OperationGroup.capture(named: "BroadcastDivideRhs") {
                            let rhsPadded = Array(repeating: 1, count: resultGradient.dim - rhs.dim) + rhs.shape
                            let rhsReducedAxes = zip(rhsPadded, resultGradient.shape).enumerated()
                                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                            
                            var tmp1reducedShape = rhsPadded
                            
                            for a in rhsReducedAxes.reversed() {
                                tmp1reducedShape.remove(at: a)
                            }
                            
                            let m = resultGradient * lhs
                            let d = m / (rhs * rhs)
                            return -d.reduceSum(along: rhsReducedAxes).view(as: rhs.shape)
                        }
                    }
                ]
            )
            
            return XTensor(using: resultBuffer, context: context)
        } else {
            return XTensor(using: resultBuffer, context: nil)
        }
    }
    
    static prefix func - (value: XTensor<Element, Device>) -> XTensor<Element, Device> {
        return 0 - value
    }
    
    static func += (lhs: inout XTensor<Element, Device>, rhs: XTensor<Element, Device>) {
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
    
    static func -= (lhs: inout XTensor<Element, Device>, rhs: XTensor<Element, Device>) {
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
    
    static func *= (lhs: inout XTensor<Element, Device>, rhs: XTensor<Element, Device>) {
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
    
    static func /= (lhs: inout XTensor<Element, Device>, rhs: XTensor<Element, Device>) {
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
    
    func raised(toPowerOf power: Self) -> Self {
        (self.log() * power).exp()
    }
}
