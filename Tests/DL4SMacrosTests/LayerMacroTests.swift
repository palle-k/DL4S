//
//  LayerMacroTests.swift
//  DL4SMacrosTests
//
//  Created by Palle Klewitz on 21.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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

import DL4SMacros
import SwiftSyntaxMacroExpansion
import SwiftSyntaxMacros
import SwiftSyntaxMacrosGenericTestSupport
import Testing

struct LayerMacroTests {
    private let macros: [String: MacroSpec] = [
        "Layer": MacroSpec(type: LayerMacro.self, conformances: ["LayerType"]),
        "Frozen": MacroSpec(type: FrozenMacro.self),
    ]

    /// Runs the macros on the source and compares the expansion and the diagnostics.
    private func expectExpansion(of source: String, to expansion: String, diagnostics: [DiagnosticSpec] = [], sourceLocation: SourceLocation = #_sourceLocation) {
        assertMacroExpansion(
            source,
            expandedSource: expansion,
            diagnostics: diagnostics,
            macroSpecs: macros,
            failureHandler: { failure in
                Issue.record(Comment(rawValue: failure.message), sourceLocation: sourceLocation)
            },
        )
    }

    @Test func testGeneratesVisitTensorsForStoredVarProperties() {
        expectExpansion(
            of: """
            @Layer
            public struct Dense<Element: RandomizableType, Device: DeviceType>: Codable {
                public var weights: Tensor<Element, Device>
                public var bias: Tensor<Element, Device>
                public let inputSize: Int
                public var outputSize: Int {
                    bias.shape[0]
                }
                public static var shared = 1
                var rate: Float = 0.5 {
                    didSet {
                        rate = min(rate, 1)
                    }
                }
                var a, b: Tensor<Element, Device>
            }
            """,
            to: """
            public struct Dense<Element: RandomizableType, Device: DeviceType>: Codable {
                public var weights: Tensor<Element, Device>
                public var bias: Tensor<Element, Device>
                public let inputSize: Int
                public var outputSize: Int {
                    bias.shape[0]
                }
                public static var shared = 1
                var rate: Float = 0.5 {
                    didSet {
                        rate = min(rate, 1)
                    }
                }
                var a, b: Tensor<Element, Device>

                public mutating func visitTensors(_ visitor: inout DL4S.TensorVisitor<Element, Device>) {
                    visitor.stored(&self.weights, named: "weights")
                    visitor.stored(&self.bias, named: "bias")
                    visitor.stored(&self.rate, named: "rate")
                    visitor.stored(&self.a, named: "a")
                    visitor.stored(&self.b, named: "b")
                }
            }

            extension Dense: DL4S.LayerType {
            }
            """,
        )
    }

    @Test func testFrozenPropertiesGetTheFrozenRole() {
        expectExpansion(
            of: """
            @Layer
            struct Classifier {
                typealias Element = Float
                typealias Device = CPU
                @Frozen var backbone: ResNet18<Float, CPU>
                @Frozen var `default`: Tensor<Float, CPU>
                var head: Dense<Float, CPU>
            }
            """,
            to: """
            struct Classifier {
                typealias Element = Float
                typealias Device = CPU
                var backbone: ResNet18<Float, CPU>
                var `default`: Tensor<Float, CPU>
                var head: Dense<Float, CPU>

                mutating func visitTensors(_ visitor: inout DL4S.TensorVisitor<Element, Device>) {
                    visitor.stored(&self.backbone, named: "backbone", role: .frozen)
                    visitor.stored(&self.`default`, named: "default", role: .frozen)
                    visitor.stored(&self.head, named: "head")
                }
            }

            extension Classifier: DL4S.LayerType {
            }
            """,
        )
    }

    @Test func testEmptyLayerGetsAnEmptyTraversal() {
        expectExpansion(
            of: """
            @Layer
            package struct Relu<Element: NumericType, Device: DeviceType>: Codable {
                package init() {}
            }
            """,
            to: """
            package struct Relu<Element: NumericType, Device: DeviceType>: Codable {
                package init() {}

                package mutating func visitTensors(_ visitor: inout DL4S.TensorVisitor<Element, Device>) {
                }
            }

            extension Relu: DL4S.LayerType {
            }
            """,
        )
    }

    @Test func testClassesGetANonMutatingMethod() {
        expectExpansion(
            of: """
            @Layer
            public final class Dense<Element, Device> {
                public var weights: Tensor<Element, Device>
                let cache: Int
            }
            """,
            to: """
            public final class Dense<Element, Device> {
                public var weights: Tensor<Element, Device>
                let cache: Int

                public func visitTensors(_ visitor: inout DL4S.TensorVisitor<Element, Device>) {
                    visitor.stored(&self.weights, named: "weights")
                }
            }

            extension Dense: DL4S.LayerType {
            }
            """,
        )
    }

    @Test func testRejectsTypesThatAreNotStructsOrClasses() {
        expectExpansion(
            of: """
            @Layer
            enum Activation<Element, Device> {
                case relu
            }
            """,
            to: """
            enum Activation<Element, Device> {
                case relu
            }
            """,
            diagnostics: [
                DiagnosticSpec(message: "'@Layer' can only be applied to a struct or a class", line: 1, column: 1),
            ],
        )
    }

    @Test func testRejectsStructsWithoutElementAndDeviceTypes() {
        expectExpansion(
            of: """
            @Layer
            struct Classifier {
                typealias Device = CPU
                var head: Dense<Float, CPU>
            }
            """,
            to: """
            struct Classifier {
                typealias Device = CPU
                var head: Dense<Float, CPU>
            }
            """,
            diagnostics: [
                DiagnosticSpec(message: "'@Layer' needs the types 'Element' and 'Device' in the scope of 'Classifier', but 'Element' is missing. Add generic parameters or typealiases, or implement 'LayerType' manually.", line: 1, column: 1),
            ],
        )
    }

    @Test func testFrozenRejectsConstantsAndComputedProperties() {
        expectExpansion(
            of: """
            struct Classifier {
                @Frozen let mask: Tensor<Float, CPU>
                @Frozen var scale: Tensor<Float, CPU> {
                    Tensor(1)
                }
                @Frozen func run() {}
            }
            """,
            to: """
            struct Classifier {
                let mask: Tensor<Float, CPU>
                var scale: Tensor<Float, CPU> {
                    Tensor(1)
                }
                func run() {}
            }
            """,
            diagnostics: [
                DiagnosticSpec(message: "'@Frozen' can only be applied to a 'var' property, because a 'let' property is a constant and is not visited", line: 2, column: 5),
                DiagnosticSpec(message: "'@Frozen' can only be applied to a stored property", line: 3, column: 5),
                DiagnosticSpec(message: "'@Frozen' can only be applied to a stored property", line: 6, column: 5),
            ],
        )
    }
}
