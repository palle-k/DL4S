//
//  Optim.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


protocol Optimizer {
    associatedtype Element: NumericType
    func step()
}


public class SGDOptimizer<Element: NumericType>: Optimizer {
    public var learningRate: Element
    public var regularizationL2: Element = 0
    
    public let parameters: [Vector<Element>]
    
    public init(learningRate: Element, parameters: [Vector<Element>]) {
        self.learningRate = learningRate
        self.parameters = parameters
    }
    
    public func step() {
        for parameter in self.parameters {
            Element.vsMulVAdd(lhs: parameter.gradient.immutable, rhs: -self.learningRate, add: parameter.values.immutable, result: parameter.values, count: parameter.count)
        }
    }
}

