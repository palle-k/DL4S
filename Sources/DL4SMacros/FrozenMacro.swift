//
//  FrozenMacro.swift
//  DL4SMacros
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

import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxMacros

/// Marks a stored property of a `@Layer` struct as frozen.
public struct FrozenMacro: PeerMacro {
    // The macro produces no code. `LayerMacro` checks for the presence of the `@Frozen` the attribute when it generates `visitTensors(_:)`.
    // The expansion checks that the attribute is on a stored `var` property.

    public static func expansion(
        of node: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext,
    ) throws -> [DeclSyntax] {
        guard let variable = declaration.as(VariableDeclSyntax.self) else {
            context.diagnose(Diagnostic(node: node, message: MacroDiagnostic("'@Frozen' can only be applied to a stored property", id: "frozenNotAProperty")))
            return []
        }
        guard variable.bindingSpecifier.tokenKind == .keyword(.var) else {
            context.diagnose(Diagnostic(node: node, message: MacroDiagnostic("'@Frozen' can only be applied to a 'var' property, because a 'let' property is a constant and is not visited", id: "frozenOnLet")))
            return []
        }
        guard variable.storedBindings.count == variable.bindings.count else {
            context.diagnose(Diagnostic(node: node, message: MacroDiagnostic("'@Frozen' can only be applied to a stored property", id: "frozenOnComputedProperty")))
            return []
        }
        return []
    }
}
