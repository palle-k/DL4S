//
//  LearningRate.swift
//  DL4S
//
//  Created by Palle Klewitz on 20.05.20.
//  Copyright (c) 2020 - Palle Klewitz
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

public protocol LearningRateScheduler {
    func learningRate<Element: NumericType>(atStep step: Int) -> Element
}

public struct NoamScheduler: LearningRateScheduler {
    public let warmupSteps: Int
    public let modelDim: Int
    
    public init(warmupSteps: Int, modelDim: Int) {
        self.warmupSteps = warmupSteps
        self.modelDim = modelDim
    }
    
    public func learningRate<Element: NumericType>(atStep step: Int) -> Element {
        let step = Float(step)
        let warmupSteps = Float(self.warmupSteps)
        let modelDim = Float(self.modelDim)
        
        return Element(1 / sqrt(modelDim) * min(1 / sqrt(step), step * pow(warmupSteps, -1.5)))
    }
}
