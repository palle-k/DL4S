# AGENTS.md

Guidance for coding agents that work in this repository.

## What this is

DL4S is a pure Swift deep learning library with built-in reverse-mode automatic differentiation (dynamic compute graphs, no special toolchain). It provides tensor operations, NN layers, optimizers, losses, and reference architectures (ResNet18, VGG, AlexNet, Transformer). It supports macOS, iOS, tvOS, watchOS, and Linux. It has no external dependencies.

## Commands

```bash
swift build                                          # build
swift test                                           # run all tests
swift test --filter DL4STests.GradientTests          # run one test class
swift test --filter DL4STests.GradientTests/testMul  # run one test method
swift test --generate-linuxmain                      # regenerate XCTestManifests.swift after adding tests
```

There is no linter or formatter configuration in this repo. CI is a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs `swift test` in debug and release configuration (`-c release -Xswiftc -enable-testing`) on macOS and Ubuntu.

On Linux, acceleration comes from Intel MKL/IPP instead of Accelerate. Build with `swift build -c release -Xswiftc -DMKL_ENABLE -Xlinker -L${MKLROOT}/lib/intel64 -Xlinker -L${IPPROOT}/lib/intel64` (see README for setup). Without MKL or Accelerate, a slow generic fallback is used.

Tests are XCTest classes in `Tests/DL4STests`. The MNIST idx files in that directory are bundled as test resources. Tests that train real models (`MNISTTests`, `TransformerTests`, `ModelTests`) or measure performance are slow and are skipped unless the `DL4S_LONG_TESTS` environment variable is set, so CI does not run them. Run them locally with `DL4S_LONG_TESTS=1 swift test`.

## Architecture

Two targets: `MKL` (a C shim that only exposes Intel MKL/IPP headers through `include/module.modulemap`; `placeholder.c` is empty on purpose) and `DL4S`, which depends on it. `Package.swift` sets no build flags; all acceleration configuration happens through command-line `-Xswiftc`/`-Xlinker` flags.

### Generic core: Tensor over Element and Device

Everything is generic over two parameters: `Tensor<Element: NumericType, Device: DeviceType>` (`Sources/DL4S/Tensor/Tensor.swift`). Valid elements are `Float`, `Double`, and `Int32` (`Sources/DL4S/Numerics/`).

- `DeviceType` (`Sources/DL4S/Engine/Engine.swift`) bundles two associated types: `Memory: MemoryOperatorsType` (raw allocation, slicing) and `Engine: EngineType` (the full kernel catalogue: broadcast ops, gemm, conv, reductions, scatter/gather, etc.).
- `CPU` (`Engine/CPU/`) is the only device. `CPUEngine` methods are thin shims that forward to static methods on the element type (`CPUNumeric` protocol). The per-type implementations in `Engine/CPU/Numeric/` select between three variants with conditional compilation: `#if MKL_ENABLE`, `#elseif canImport(Accelerate)`, and a generic Swift fallback (`CPUGeneric.swift`). A GPU backend can conform to the same protocols, but none exists.
- Kernels that accumulate into an existing gradient have fused `...Add` variants (`permuteAxesAdd`, `subscriptWriteAdd`, `reverseAdd`, ...). Backward passes use these to avoid a separate add kernel.
- `Tensor` wraps a `TensorHandle` class with copy-on-write (`ensureOwnership()`); views share the parent buffer, and only the root handle frees it.

### Automatic differentiation

Autograd is closure-based and lives in `Sources/DL4S/Tensor/`:

- Each differentiable operation (all in `Tensor/Operators/*.swift`) computes its forward result through the engine, then attaches a `TensorContext` (`Context.swift`) that holds the source tensors and one backpropagation closure per source. Capture only happens when an operand `requiresGradient`.
- `tensor.gradients(of:retainBackwardsGraph:)` (`Tensor.swift`) topologically sorts the graph by `backpropID` and walks it backwards. Backward closures are written with normal tensor operations, so the backward pass builds its own graph when `retainBackwardsGraph: true`, which enables second and higher derivatives. With `false`, accumulated gradients are detached.
- New tensor operation checklist: add the primitive to `EngineType`, implement it in `CPUEngine` (usually delegating to a `CPUNumeric` static, implemented in `CPUFloat`/`CPUDouble`/`CPUInt32`/`CPUGeneric`), then add the public `Tensor` method in the matching `Tensor/Operators/*.swift` file with its `TensorContext` gradient closures. Add a gradient check to `Tests/DL4STests/GradientTests.swift` and tick the README feature list.
- Backward closures must not capture the result tensor directly (retain cycle). See `exp`/`tanh` in `Unary.swift`: they capture a copy and recompute the forward value when the backward graph itself needs gradients.
- Debug-only graph tooling: `Tensor.tag`, `OperationGroup.capture(named:)`, and `tensor.graph()` (Graphviz DOT output) are gated behind `#if DEBUG`.

### NN layer system

- `LayerType` (`NN/Layer/Layer.swift`) has associated `Inputs`/`Outputs` types (not fixed to tensors, which is how RNNs return tuples), `callAsFunction`, and two parameter accessors: `parameters` and `parameterPaths` (writable key paths into the layer struct).
- Layers are value types. Optimizers (`NN/Optimizer/`) copy the model and mutate its parameters through `parameterPaths`. This is why usage code must call `optimizer.model(input)`, never the original `model` variable.
- `Sequential` (`NN/Layer/Sequential.swift`) is a result builder that folds a block of layers into nested `Sequential<Sequential<A, B>, C>` pairs.
- Reference architectures live in `NN/Models/`.

## Conventions

- Every file starts with the MIT license header (`// <Filename>.swift / DL4S / Created by ... / Copyright ...`). New files get the same header.
- Public APIs carry `///` doc comments with `- Parameters:` / `- Returns:`. The `docs/` directory is Jazzy output; do not hand-edit it.
- Hot generic functions use `@inline(__always)` and `@_specialize(where Element == Float, Device == CPU)`.
- Engine primitives use terse names (`vAdd`, `vsMul`, `gemm`, `img2col`); public tensor methods are spelled out (`matrixMultiplied(with:)`, `permuted(to:)`, `reduceSum(along:)`).

## Debugging

`util/debugger_support/tensor.py` is an LLDB script that adds readable summaries for `Tensor` and `ShapedBuffer` values. Load it with `command script import` in LLDB or from `~/.lldbinit`.

## Documentation & Communication
- All communication, comments, documentation, etc. must use ASD-STE100 Simple Technical English.
- Documentation must follow the Google Developer Documentation style guide. This includes spelling, terminology, choice of words, inclusive language, phrasing and  sentence construction.
- Avoid em-dashes and en-dashes in sentence constructions. Write concisely. 
- Phrases to avoid: "load-bearing", "gated", "the X is real", "X is doing a lot of work", "it's not X, it's Y", "genuinely"
