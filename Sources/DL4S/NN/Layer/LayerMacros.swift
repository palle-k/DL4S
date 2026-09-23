//
//  LayerMacros.swift
//  DL4S
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

/// Makes a struct or a class a layer.
///
/// The macro adds the ``LayerType`` conformance and generates ``TensorContainer/visitTensors(_:)`` from the stored
/// `var` properties of the type:
///
/// - Each `Tensor<Element, Device>` not marked as `@Frozen` is reported as a weight.
/// - A layer with the same `Parameter` and `Device` types is traversed as a sub-layer.
/// - An optional or an array of either is handled the same way. A `nil` value is skipped.
/// - Every other property is ignored. `let` properties are constants and are not visited.
///
/// Mark a tensor that is saved but not updated during training with ``Frozen()``.
///
/// ```swift
/// @Layer
/// public struct Dense<Element: RandomizableType, Device: DeviceType>: Codable {
///     public var weights: Tensor<Element, Device>
///     public var bias: Tensor<Element, Device>
///
///     public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
///         inputs.matrixMultiplied(with: weights) + bias
///     }
/// }
/// ```
///
/// The generated method has the signature `visitTensors(_ visitor: inout TensorVisitor<Element, Device>)`.
/// The type must have types named `Element` and `Device` in scope, as generic parameters or as typealiases.
/// Implement ``LayerType`` by hand to get fine-grained control.
@attached(extension, conformances: LayerType)
@attached(member, names: named(visitTensors))
public macro Layer() = #externalMacro(module: "DL4SMacros", type: "LayerMacro")

/// Marks a stored property of a ``Layer()`` type as frozen.
///
/// A frozen tensor is saved with the model but never trained. When used on a sublayer, all weights of the sublayer are
/// treated as frozen.
///
/// ```swift
/// @Layer
/// struct Classifier: Codable {
///     typealias Element = Float
///     typealias Device = CPU
///
///     @Frozen var backbone: ResNet18<Float, CPU>
///     var head: Dense<Float, CPU>
///     ...
/// }
/// ```
@attached(peer)
public macro Frozen() = #externalMacro(module: "DL4SMacros", type: "FrozenMacro")
