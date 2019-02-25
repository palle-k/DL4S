//
//  Operation.swift
//  DL4S
//
//  Created by Palle Klewitz on 25.02.19.
//

import Foundation

protocol OperationContext {
    func zeroGradient()
    func backwards(from parameter: Variable)
}

protocol BinaryOperationContext: OperationContext {
    var lhs: Variable { get }
    var rhs: Variable { get }
}

extension BinaryOperationContext {
    func zeroGradient() {
        lhs.zeroGradient()
        rhs.zeroGradient()
    }
}

protocol UnaryOperationContext: OperationContext {
    var param: Variable { get }
}

extension UnaryOperationContext {
    func zeroGradient() {
        param.zeroGradient()
    }
}


public func + (lhs: Variable, rhs: Variable) -> Variable {
    struct AdditionOperationContext: BinaryOperationContext {
        var lhs: Variable
        var rhs: Variable
        
        func backwards(from parameter: Variable) {
            let gradient = parameter.gradient
            
            lhs.gradient += gradient
            rhs.gradient += gradient
            
            lhs._backwards()
            rhs._backwards()
        }
    }
    
    return Variable(value: lhs.value + rhs.value, context: AdditionOperationContext(lhs: lhs, rhs: rhs))
}

public func * (lhs: Variable, rhs: Variable) -> Variable {
    struct MultiplicationOperationContext: BinaryOperationContext {
        var lhs: Variable
        var rhs: Variable
        
        func backwards(from parameter: Variable) {
            let gradient = parameter.gradient
            
            lhs.gradient += gradient * rhs.value
            rhs.gradient += gradient * lhs.value
            
            lhs._backwards()
            rhs._backwards()
        }
    }
    
    return Variable(value: lhs.value * rhs.value, context: MultiplicationOperationContext(lhs: lhs, rhs: rhs))
}

public prefix func - (value: Variable) -> Variable {
    struct NegationOperationContext: UnaryOperationContext {
        var param: Variable
        
        func backwards(from parameter: Variable) {
            let gradient = parameter.gradient
            
            param.gradient += -gradient
            
            param._backwards()
        }
    }
    
    return Variable(value: -value.value, context: NegationOperationContext(param: value))
}

public func - (lhs: Variable, rhs: Variable) -> Variable {
    return lhs + -rhs
}

public func / (lhs: Variable, rhs: Variable) -> Variable {
    struct DivisionOperationContext: BinaryOperationContext {
        var lhs: Variable
        var rhs: Variable
        
        func backwards(from parameter: Variable) {
            let gradient = parameter.gradient
            
            lhs.gradient += gradient / rhs.value
            rhs.gradient += gradient * (-lhs.value / (rhs.value * rhs.value))
            
            lhs._backwards()
            rhs._backwards()
        }
    }
    
    return Variable(value: lhs.value / rhs.value, context: DivisionOperationContext(lhs: lhs, rhs: rhs))
}

public func exp(_ parameter: Variable) -> Variable {
    struct ExponentiationOperationContext: UnaryOperationContext {
        var param: Variable
        
        func backwards(from parameter: Variable) {
            let gradient = parameter.gradient
            
            param.gradient += gradient * exp(param.value)
            param._backwards()
        }
    }
    
    return Variable(value: exp(parameter.value), context: ExponentiationOperationContext(param: parameter))
}

public func log(_ parameter: Variable) -> Variable {
    struct LogOperationContext: UnaryOperationContext {
        var param: Variable
        
        func backwards(from parameter: Variable) {
            let gradient = parameter.gradient
            
            param.gradient += gradient / param.value
            param._backwards()
        }
    }
    
    return Variable(value: log(parameter.value), context: LogOperationContext(param: parameter))
}

public func sum(_ parameters: [Variable]) -> Variable {
    struct SumOperationContext: OperationContext {
        var parameters: [Variable]
        
        func backwards(from parameter: Variable) {
            let gradient = parameter.gradient
            
            parameters.forEach {$0.gradient += gradient}
            parameters.forEach {$0._backwards()}
        }
        
        func zeroGradient() {
            parameters.forEach {$0.zeroGradient()}
        }
    }
    
    return Variable(
        value: parameters.map {$0.value}.reduce(0, +),
        context: SumOperationContext(parameters: parameters)
    )
}

public func sigmoid(_ parameter: Variable) -> Variable {
    return 1 / (1 + exp(-parameter))
}

public func relu(_ parameter: Variable) -> Variable {
    struct ReluOperationContext: UnaryOperationContext {
        var param: Variable
        
        func backwards(from parameter: Variable) {
            let gradient = parameter.gradient
            
            param.gradient += gradient * (param.value >= 0 ? 1 : 0)
            param._backwards()
        }
    }
    
    return Variable(
        value: max(parameter.value, 0),
        context: ReluOperationContext(param: parameter)
    )
}

public func tanh(_ parameter: Variable) -> Variable {
    struct TanhOperationContext: UnaryOperationContext {
        var param: Variable
        
        func backwards(from parameter: Variable) {
            let gradient = parameter.gradient
            
            let imm = tanh(param.value)
            param.gradient += gradient * (1 - imm * imm)
            
            param._backwards()
        }
    }
    
    return Variable(value: tanh(parameter.value), context: TanhOperationContext(param: parameter))
}


public func softmax(_ parameters: [Variable]) -> [Variable] {
    let exponentiated = parameters.map(exp)
    let expSum = sum(exponentiated)
    
    return exponentiated.map {$0 / expSum}
}

public func dot(_ x: [Variable], _ y: [Variable]) -> Variable {
    return sum(zip(x, y).map(*))
}

public func + (lhs: [Variable], rhs: [Variable]) -> [Variable] {
    return zip(lhs, rhs).map(+)
}

public func - (lhs: [Variable], rhs: [Variable]) -> [Variable] {
    return zip(lhs, rhs).map(-)
}

public func + (lhs: [[Variable]], rhs: [[Variable]]) -> [[Variable]] {
    return zip(lhs, rhs).map {zip($0, $1).map(+)}
}

public func - (lhs: [[Variable]], rhs: [[Variable]]) -> [[Variable]] {
    return zip(lhs, rhs).map {zip($0, $1).map(-)}
}

public func * (_ x: [[Variable]], _ y: [[Variable]]) -> [[Variable]] {
    var result: [[Variable]] = Array(repeating: [], count: x.count)
    for row in 0 ..< x.count {
        result[row].reserveCapacity(y[0].count)
        for column in 0 ..< y[0].count {
            result[row].append(sum((0 ..< y.count).map {x[row][$0] * y[$0][column]}))
        }
    }
    return result
}

public func * (_ x: [[Variable]], _ y: [Variable]) -> [Variable] {
    var result: [Variable] = []
    result.reserveCapacity(x.count)
    for row in 0 ..< x.count {
        result.append(sum((0 ..< y.count).map {x[row][$0] * y[$0]}))
    }
    return result
}

func sigmoid(_ val: [Variable]) -> [Variable] {
    return val.map(sigmoid)
}

func exp(_ val: [Variable]) -> [Variable] {
    return val.map(exp)
}

func tanh(_ val: [Variable]) -> [Variable] {
    return val.map(tanh)
}

func log(_ val: [Variable]) -> [Variable] {
    return val.map(log)
}

func convolve(image: [[Variable]], filter: [[Variable]]) -> [[Variable]] {
    var result: [[Variable]] = []
    result.reserveCapacity(image.count - filter.count)
    
    for i in 0 ..< (image.count - filter.count) {
        var row: [Variable] = []
        row.reserveCapacity(image[0].count - filter[0].count)
        for j in 0 ..< (image[0].count - filter[0].count) {
            row.append(
                sum((0 ..< filter.count)
                    .flatMap {(y: Int) in (0 ..< filter[0].count).map {(y, $0)}}
                    .map { (y: Int, x: Int) -> Variable in
                        filter[y][x] * image[y + i][x + j]
                    }
                )
            )
        }
        result.append(row)
    }
    return result
}
