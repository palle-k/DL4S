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
    var symbol: String { get }
    var dependencies: [Variable] { get }
}

protocol BinaryOperationContext: OperationContext {
    var lhs: Variable { get }
    var rhs: Variable { get }
    var symbol: String { get }
}

extension BinaryOperationContext {
    func zeroGradient() {
        lhs.zeroGradient()
        rhs.zeroGradient()
    }
    
    var dependencies: [Variable] {
        return [lhs, rhs]
    }
}

protocol UnaryOperationContext: OperationContext {
    var param: Variable { get }
    var symbol: String { get }
}

extension UnaryOperationContext {
    func zeroGradient() {
        param.zeroGradient()
    }
    
    var dependencies: [Variable] {
        return [param]
    }
}


extension OperationContext {
    private var id: String {
        return String(dependencies.hashValue ^ symbol.hashValue).replacingOccurrences(of: "-", with: "_")
    }
    
    private func collectVariables() -> Set<Variable> {
        return Set(dependencies).union(dependencies.map {$0.context?.collectVariables() ?? []}.reduce(Set(), {$0.union($1)}))
    }
    
    private func collectEdges() -> [String] {
        let dependencyEdges = Array(dependencies.map {$0.context?.collectEdges() ?? []}.joined())
        let edges = dependencies.compactMap {v in v.context.map {(v, $0)}}.map { v, ctx in
            "node\(ctx.id) -> node\(self.id) [label=\"\(v.value), \(v.gradient)\"]"
        }
        let emptyEdges = dependencies.filter {$0.context == nil}.map { v in
            "var\(String(v.hashValue).replacingOccurrences(of: "-", with: "_")) -> node\(self.id)"
        }
        return dependencyEdges + edges + emptyEdges
    }
    
    private func collectNodes() -> [String] {
        return Array(dependencies.compactMap {$0.context?.collectNodes()}.joined()) + ["\(id) [label=\"\(symbol)\"]"]
    }
    
    public var description: String {
        return """
        digraph G {
            \(collectVariables().map {"var\(String($0.hashValue).replacingOccurrences(of: "-", with: "_")) [label=\"v(\($0.value), \($0.gradient))\"]"}.joined(separator: "\n\t"))
            \(collectNodes().joined(separator: "\n\t"))
            \(collectEdges().joined(separator: "\n\t"))
        }
        """
    }
}


public func + (lhs: Variable, rhs: Variable) -> Variable {
    struct AdditionOperationContext: BinaryOperationContext {
        var symbol: String {
            return "+"
        }
        
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
        var symbol: String {
            return "*"
        }
        
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
        var symbol: String {
            return "*(-1)"
        }
        
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
        var symbol: String {
            return "/"
        }
        
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

public func += (lhs: inout Variable, rhs: Variable) {
    lhs = lhs + rhs
}

public func -= (lhs: inout Variable, rhs: Variable) {
    lhs = lhs - rhs
}

public func *= (lhs: inout Variable, rhs: Variable) {
    lhs = lhs * rhs
}

public func /= (lhs: inout Variable, rhs: Variable) {
    lhs = lhs / rhs
}

public func exp(_ parameter: Variable) -> Variable {
    struct ExponentiationOperationContext: UnaryOperationContext {
        var symbol: String {
            return "exp"
        }
        
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
        var symbol: String {
            return "log"
        }
        
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
        var symbol: String {
            return "sum"
        }
        
        var parameters: [Variable]
        
        func backwards(from parameter: Variable) {
            let gradient = parameter.gradient
            
            parameters.forEach {$0.gradient += gradient}
            parameters.forEach {$0._backwards()}
        }
        
        func zeroGradient() {
            parameters.forEach {$0.zeroGradient()}
        }
        
        var dependencies: [Variable] {
            return parameters
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
        var symbol: String {
            return "relu"
        }
        
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
        var symbol: String {
            return "tanh"
        }
        
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
    precondition(x.count == y.count)
    
    return sum(zip(x, y).map(*))
}

infix operator .+ : AdditionPrecedence
infix operator .* : MultiplicationPrecedence
infix operator .- : AdditionPrecedence

public func .+ (lhs: [Variable], rhs: [Variable]) -> [Variable] {
    precondition(lhs.count == rhs.count)
    
    return zip(lhs, rhs).map(+)
}

public func .- (lhs: [Variable], rhs: [Variable]) -> [Variable] {
    precondition(lhs.count == rhs.count)
    
    return zip(lhs, rhs).map(-)
}

public func .+ (lhs: [[Variable]], rhs: [[Variable]]) -> [[Variable]] {
    precondition(lhs.count == rhs.count)
    precondition(lhs.isEmpty && rhs.isEmpty || lhs[0].count == rhs[0].count)
    
    return zip(lhs, rhs).map {zip($0, $1).map(+)}
}

public func .- (lhs: [[Variable]], rhs: [[Variable]]) -> [[Variable]] {
    precondition(lhs.count == rhs.count)
    precondition(lhs.isEmpty && rhs.isEmpty || lhs[0].count == rhs[0].count)
    
    return zip(lhs, rhs).map {zip($0, $1).map(-)}
}

public func * (_ x: [[Variable]], _ y: [[Variable]]) -> [[Variable]] {
    precondition(x[0].count == y.count)
    
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
    precondition(x.allSatisfy {$0.count == y.count})
    
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

func relu(_ val: [Variable]) -> [Variable] {
    return val.map(relu)
}

func convolve(image: [[Variable]], filter: [[Variable]], stride: Int = 1) -> [[Variable]] {
    var result: [[Variable]] = []
    result.reserveCapacity(image.count - filter.count)
    
    for i in Swift.stride(from: 0, to: image.count - filter.count, by: stride) {
        var row: [Variable] = []
        row.reserveCapacity(image[0].count - filter[0].count)
        for j in Swift.stride(from: 0, to: image[0].count - filter[0].count, by: stride) {
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

func convolve(image: [[[Variable]]], filter: [[[Variable]]], stride: Int = 1) -> [[Variable]] {
    var result: [[Variable]] = Array(repeating: 0, rows: image.count - filter.count, columns: image[0].count - filter[0].count)
    for z in 0 ..< image.count {
        result = result .+ convolve(image: image[z], filter: filter[z], stride: stride)
    }
    return result
}

func convolve(image: [[[Variable]]], filters: [[[[Variable]]]], stride: Int = 1) -> [[[Variable]]] {
    return filters.reduce(into: []) { acc, filter in
        acc.append(convolve(image: image, filter: filter, stride: stride))
    }
}

func binaryCrossEntropy(expected: Variable, actual: Variable) -> Variable {
    return -(expected * log(actual) + (1 - expected) * log(1 - actual))
}

func categoricalCrossEntropy(expected: [Variable], actual: [Variable]) -> Variable {
    return -sum(zip(expected, actual.map(log)).map(*))
}
