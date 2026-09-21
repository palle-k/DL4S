//
//  LayerMacro.swift
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
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

/// Adds the `LayerType` conformance to a struct or class and generates `visitTensors(_:)` from its stored properties.
public struct LayerMacro: ExtensionMacro, MemberMacro {
    private static let requiredTypeNames = ["Element", "Device"]

    /// The parts of a struct or class declaration that the macro reads.
    private struct LayerDeclaration {
        let name: String
        let genericParameters: GenericParameterClauseSyntax?
        let members: MemberBlockItemListSyntax
        let modifiers: DeclModifierListSyntax
        let isClass: Bool

        init?(_ declaration: some DeclGroupSyntax) {
            if let structDecl = declaration.as(StructDeclSyntax.self) {
                name = structDecl.name.text
                genericParameters = structDecl.genericParameterClause
                members = structDecl.memberBlock.members
                modifiers = structDecl.modifiers
                isClass = false
            } else if let classDecl = declaration.as(ClassDeclSyntax.self) {
                name = classDecl.name.text
                genericParameters = classDecl.genericParameterClause
                members = classDecl.memberBlock.members
                modifiers = classDecl.modifiers
                isClass = true
            } else {
                return nil
            }
        }
    }

    public static func expansion(
        of node: AttributeSyntax,
        attachedTo declaration: some DeclGroupSyntax,
        providingExtensionsOf type: some TypeSyntaxProtocol,
        conformingTo protocols: [TypeSyntax],
        in context: some MacroExpansionContext,
    ) throws -> [ExtensionDeclSyntax] {
        // The member expansion reports a declaration that is not a struct or class, or that lacks the required
        // types. The compiler passes only the protocols that the type does not conform to yet, so an empty list
        // needs no extension.
        guard let layer = LayerDeclaration(declaration), !protocols.isEmpty,
              missingRequiredTypeNames(in: layer, context: context).isEmpty
        else {
            return []
        }
        let conformance: DeclSyntax = "extension \(type.trimmed): DL4S.LayerType {}"
        return [conformance.cast(ExtensionDeclSyntax.self)]
    }

    public static func expansion(
        of node: AttributeSyntax,
        providingMembersOf declaration: some DeclGroupSyntax,
        conformingTo protocols: [TypeSyntax],
        in context: some MacroExpansionContext,
    ) throws -> [DeclSyntax] {
        guard let layer = LayerDeclaration(declaration) else {
            context.diagnose(Diagnostic(node: node, message: MacroDiagnostic("'@Layer' can only be applied to a struct or a class", id: "layerNotAStructOrClass")))
            return []
        }

        let missingTypeNames = missingRequiredTypeNames(in: layer, context: context)
        if !missingTypeNames.isEmpty {
            let names = missingTypeNames.map { "'\($0)'" }.joined(separator: " and ")
            context.diagnose(Diagnostic(
                node: node,
                message: MacroDiagnostic("'@Layer' needs the types 'Element' and 'Device' in the scope of '\(layer.name)', but \(names) \(missingTypeNames.count == 1 ? "is" : "are") missing. Add generic parameters or typealiases, or implement 'LayerType' manually.", id: "layerMissingTypes"),
            ))
            return []
        }

        // The macro has no type information. The generated code calls `visitor.stored(&self.name, named: "name")` for
        // every stored `var` property, and the overloads of `stored` in `TensorVisitor` select the role of the
        // property by its type.
        let calls = storedProperties(of: layer).map { property in
            let role = property.isFrozen ? ", role: .frozen" : ""
            return "visitor.stored(&self.\(property.reference), named: \"\(property.name)\"\(role))"
        }
        let body = calls.isEmpty ? "" : "\n" + calls.map { "    " + $0 }.joined(separator: "\n") + "\n"
        let access = accessModifier(of: layer)
        // A class satisfies the `mutating` requirement with a plain method.
        let mutating = layer.isClass ? "" : "mutating "
        let method: DeclSyntax = """
        \(raw: access)\(raw: mutating)func visitTensors(_ visitor: inout DL4S.TensorVisitor<Element, Device>) {\(raw: body)}
        """
        return [method]
    }

    /// A stored `var` property of the struct.
    private struct StoredProperty {
        /// Name of the property without backticks, used in the path of the tensor.
        let name: String

        /// Name of the property as written, used to reference it in code.
        let reference: String

        /// Whether the property has the `@Frozen` attribute.
        let isFrozen: Bool
    }

    private static func storedProperties(of layer: LayerDeclaration) -> [StoredProperty] {
        layer.members.flatMap { member -> [StoredProperty] in
            guard let variable = member.decl.as(VariableDeclSyntax.self),
                  variable.bindingSpecifier.tokenKind == .keyword(.var),
                  !variable.modifiers.contains(where: { $0.name.tokenKind == .keyword(.static) || $0.name.tokenKind == .keyword(.lazy) })
            else {
                return []
            }
            let isFrozen = variable.attributes.contains { attribute in
                let name = attribute.as(AttributeSyntax.self)?.attributeName.trimmedDescription
                return name == "Frozen" || name == "DL4S.Frozen"
            }
            return variable.storedBindings.compactMap { binding -> StoredProperty? in
                guard let pattern = binding.pattern.as(IdentifierPatternSyntax.self) else {
                    return nil
                }
                let reference = pattern.identifier.trimmed.text
                let name = String(reference.drop(while: { $0 == "`" }).reversed().drop(while: { $0 == "`" }).reversed())
                return StoredProperty(name: name, reference: reference, isFrozen: isFrozen)
            }
        }
    }

    // The generated method uses the names Element and Device. A type that is nested in a generic type may
    // take them from the outer type, so the check is skipped in that case.
    private static func missingRequiredTypeNames(in layer: LayerDeclaration, context: some MacroExpansionContext) -> [String] {
        let isNestedInGenericType = context.lexicalContext.contains { outer in
            outer.as(StructDeclSyntax.self)?.genericParameterClause != nil
                || outer.as(ClassDeclSyntax.self)?.genericParameterClause != nil
                || outer.as(EnumDeclSyntax.self)?.genericParameterClause != nil
                || outer.as(ActorDeclSyntax.self)?.genericParameterClause != nil
        }
        if isNestedInGenericType {
            return []
        }

        var availableNames = Set<String>()
        for parameter in layer.genericParameters?.parameters ?? [] {
            availableNames.insert(parameter.name.text)
        }
        for member in layer.members {
            if let alias = member.decl.as(TypeAliasDeclSyntax.self) {
                availableNames.insert(alias.name.text)
            }
        }
        return requiredTypeNames.filter { !availableNames.contains($0) }
    }

    // The witness of a protocol requirement must be as visible as the conformance of the type.
    private static func accessModifier(of layer: LayerDeclaration) -> String {
        for modifier in layer.modifiers {
            switch modifier.name.tokenKind {
            case .keyword(.public), .keyword(.open):
                return "public "
            case .keyword(.package):
                return "package "
            default:
                continue
            }
        }
        return ""
    }
}

extension VariableDeclSyntax {
    /// The bindings of the declaration that are stored properties.
    ///
    /// A binding with a getter, or with `get` or `set` accessors, is a computed property. A binding with only
    /// `willSet` or `didSet` observers is stored.
    var storedBindings: [PatternBindingSyntax] {
        bindings.filter { binding in
            guard let accessorBlock = binding.accessorBlock else {
                return true
            }
            switch accessorBlock.accessors {
            case .getter:
                return false
            case let .accessors(accessors):
                return !accessors.contains { accessor in
                    switch accessor.accessorSpecifier.tokenKind {
                    case .keyword(.get), .keyword(.set), .keyword(._read), .keyword(._modify), .keyword(.unsafeAddress), .keyword(.unsafeMutableAddress):
                        true
                    default:
                        false
                    }
                }
            }
        }
    }
}
