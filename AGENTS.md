# AGENTS.md
Guidance for coding agents that work in this repository.

## What this is
DL4S is a pure Swift deep learning library with built-in reverse-mode automatic differentiation (dynamic compute graphs, no special toolchain). It provides tensor operations, NN layers, optimizers, losses, and reference architectures (ResNet18, VGG, AlexNet, Transformer). It supports macOS, iOS, tvOS, watchOS, and Linux.

## Commands
```bash
swift build                                          # build
swift test                                           # run all tests
swift test --filter GradientTests                    # run one test suite
swift test --filter GradientTests/testMatMul         # run one test function
DL4S_LONG_TESTS=1 swift test                         # include the long training runs
swift test --filter Concurrency                      # run the thread-safety stress suite
swift test --sanitize=thread --filter Concurrency    # run it under thread sanitizer
swift test --traits MKL --filter VecTests            # x86_64 Linux with oneAPI: run the MKL smoke test
swift package -Xswiftc -DDL4S_SKIP_MKL_PLATFORM_CHECK generate-documentation --target DL4S   # build the DocC documentation
```

There is no linter or formatter configuration in this repo. CI is a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs `swift test` in debug and release configuration (`-c release -Xswiftc -enable-testing`) on macOS and Ubuntu, and runs the concurrency suite under the thread sanitizer. A second workflow (`.github/workflows/tsan-full.yml`) runs the full test suite under the thread sanitizer on every push to master and on manual dispatch. Both sanitizer jobs run on macOS only: on Linux, TSan does not see `Synchronization.Mutex` and reports every access under the lock as a race (verified with Swift 6.1 and 6.3.3).

On x86_64 Linux, acceleration comes from Intel oneAPI MKL/IPP instead of Accelerate through the `MKL` package trait (off by default): `source /opt/intel/oneapi/setvars.sh`, `export CPATH=${IPPROOT}/include:${CPATH}`, then `swift build -c release --traits MKL` (see README for setup). The trait defines `MKL_ENABLE` for the `DL4S` and `DL4STests` targets. Without MKL or Accelerate, an unoptimized generic fallback is used. A third workflow (`.github/workflows/mkl-integration.yml`) runs `swift test --traits MKL --filter VecTests` on ubuntu-24.04 with oneAPI from the Intel apt repository; it runs on pull requests and pushes to master that touch the CPU backend, `Sources/CMKL`, or `Package.swift`, and on manual dispatch. A fourth workflow (`.github/workflows/docs.yml`) builds the DocC documentation with the plugin flag from the Commands block on every push to master and on manual dispatch, and publishes it to the `gh-pages` branch. 

Allocation tracing is a debugging aid that is compiled in only with `-Xswiftc -DDL4S_TRACE_ALLOCATIONS`. With the flag, `CPUMemoryOperators.setAllocationTracing(true)` records the call stack of every allocation and prints the call stack of a buffer that is not freed after 5 seconds. Builds without the flag contain no tracing code. Run the tracing test with `swift test --sanitize=thread -Xswiftc -DDL4S_TRACE_ALLOCATIONS --filter Concurrency`.

Tests use Swift Testing (`@Suite` structs with `@Test` functions) in `Tests/DL4STests`. Tests that train real models for minutes (`ModelTests`, `TransformerTests`, and the full-set runs in `MNISTTests` and `TransformerMNISTTests`) have the `.longRunning` trait from `TestUtil.swift`, which skips them unless the `DL4S_LONG_TESTS` environment variable is set, and run `.serialized`. Run them locally with `DL4S_LONG_TESTS=1 swift test`. The short MNIST training runs in `MNISTTests` carry the `.trainsModel` trait: they run with an accelerated backend, in release builds, or with `DL4S_LONG_TESTS` set, because the generic fallback in a debug build needs more than an hour for them. The MNIST idx files in the test directory are bundled as test resources; `MNIST.sample` (5,000 training images) and `MNIST.full` in `MNIST.swift` load them once per process. `TestUtil.swift` also has `expectEqual(_:_:accuracy:)`, `expectClose(_:_:tolerance:)` for tensors, and `numericalGradient(of:at:)` for finite-difference gradient checks.

`ConcurrencyTests` runs inference, backpropagation, dropout, and weight initialization from several raw threads at the same time. It is the acceptance test for the thread-safety work. Run the suite under the thread sanitizer to see data races as reports.

## Architecture
Two targets: `CMKL` (a system library target: `module.modulemap` plus `shim.h`, which includes `mkl.h` and `ipp.h`; the MKL include path and link line come from the `mkl-dynamic-lp64-gomp` pkg-config file, the IPP libraries from `link` directives in the module map) and `DL4S`, which depends on `CMKL` only when the `MKL` trait is on. `Package.swift` declares `CMKL` only on x86_64 Linux hosts, so builds on other hosts do not look for the pkg-config file. Accelerate needs no configuration.

### Generic core: Tensor over Element and Device
Everything is generic over two parameters: `Tensor<Element: NumericType, Device: DeviceType>` (`Sources/DL4S/Tensor/Tensor.swift`). Valid elements are `Float`, `Double`, and `Int32` (`Sources/DL4S/Numerics/`).
- `DeviceType` (`Sources/DL4S/Engine/Engine.swift`) bundles two associated types: `Memory: MemoryOperatorsType` (raw allocation, slicing) and `Engine: EngineType` (the full kernel catalogue: broadcast ops, gemm, conv, reductions, scatter/gather, etc.).
- `CPU` (`Engine/CPU/`) is the only device. `CPUEngine` methods are thin shims that forward to static methods on the element type (`CPUNumeric` protocol). The per-type implementations in `Engine/CPU/Numeric/` select between three variants with conditional compilation: `#if MKL_ENABLE`, `#elseif canImport(Accelerate)`, and a generic Swift fallback (`CPUGeneric.swift`). A GPU backend can conform to the same protocols, but none exists.
- Kernels that accumulate into an existing gradient have fused `...Add` variants (`permuteAxesAdd`, `subscriptWriteAdd`, `reverseAdd`, ...). Backward passes use these to avoid a separate add kernel.
- `Tensor` wraps a `TensorHandle` class with copy-on-write (`ensureOwnership()`); views share the parent buffer, and only the root handle frees it.

### Automatic differentiation
Autograd is closure-based and lives in `Sources/DL4S/Tensor/`:
- Each differentiable operation (all in `Tensor/Operators/*.swift`) computes its forward result through the engine, then attaches a `TensorContext` (`Context.swift`) that holds the source tensors and the backpropagation closures. The context has two forms: one closure per source (the default), or one closure for all sources (`backpropagateAll:`) that owns the accumulators and returns every source gradient at once. Use the second form when one kernel produces all source gradients, as `stack` does, so the kernel runs once per backward visit and the closures share no state. Capture only happens when an operand `requiresGradient`.
- `tensor.gradients(of:retainBackwardsGraph:)` (`Tensor.swift`) topologically sorts the graph by `backpropID` and walks it backwards. Backward closures are written with normal tensor operations, so the backward pass builds its own graph when `retainBackwardsGraph: true`, which enables second and higher derivatives. With `false`, accumulated gradients are detached.
- New tensor operation checklist: add the primitive to `EngineType`, implement it in `CPUEngine` (usually delegating to a `CPUNumeric` static, implemented in `CPUFloat`/`CPUDouble`/`CPUInt32`/`CPUGeneric`), then add the public `Tensor` method in the matching `Tensor/Operators/*.swift` file with its `TensorContext` gradient closures. Add a gradient check to `Tests/DL4STests/GradientTests.swift` and tick the README feature list.
- Backward closures must not capture the result tensor directly (retain cycle). See `exp`/`tanh` in `Unary.swift`: they capture a copy and recompute the forward value when the backward graph itself needs gradients.
- Debug-only graph tooling: `Tensor.tag`, `OperationGroup.capture(named:)`, and `tensor.graph()` (Graphviz DOT output) are compiled only in `#if DEBUG`. The operation stack that `capture` records is a `@TaskLocal`, so each thread or task has its own stack.
- Each tensor gets its `backpropID` from a process-wide atomic counter (`UniqueID`). Copies keep the id. `ensureOwnership` creates a new tensor with a new id when the buffer is shared.

### NN layer system
- `LayerType` (`NN/Layer/Layer.swift`) has associated `Inputs`/`Outputs` types (not fixed to tensors, which is how RNNs return tuples), `callAsFunction`, and two parameter accessors: `parameters` and `parameterPaths` (writable key paths into the layer struct).
- Layers are value types. Optimizers (`NN/Optimizer/`) copy the model and mutate its parameters through `parameterPaths`. This is why usage code must call `optimizer.model(input)`, never the original `model` variable.
- `Sequential` (`NN/Layer/Sequential.swift`) is a result builder that folds a block of layers into nested `Sequential<Sequential<A, B>, C>` pairs.
- Reference architectures live in `NN/Models/`.

## Conventions
- Every file starts with the MIT license header (`// <Filename>.swift / DL4S / Created by ... / Copyright ...`). New files get the same header.
- Public APIs carry `///` doc comments with `- Parameters:` / `- Returns:`. The reference documentation is DocC output published from CI.
- The package builds in Swift 6 language mode (`swiftLanguageModes: [.v6]` in `Package.swift`), so concurrency diagnostics are errors.
- Hot generic functions use `@inline(__always)` and `@_specialize(where Element == Float, Device == CPU)`.
- Engine primitives use terse names (`vAdd`, `vsMul`, `gemm`, `img2col`); public tensor methods are spelled out (`matrixMultiplied(with:)`, `permuted(to:)`, `reduceSum(along:)`).

## Debugging
`util/debugger_support/tensor.py` is an LLDB script that adds readable summaries for `Tensor` and `ShapedBuffer` values. Load it with `command script import` in LLDB or from `~/.lldbinit`.

## Documentation & Communication
- All communication, comments, documentation, etc. must use ASD-STE100 Simple Technical English.
- Documentation must follow the Google Developer Documentation style guide. This includes spelling, terminology, choice of words, inclusive language, phrasing and  sentence construction.
- Avoid em-dashes and en-dashes in sentence constructions. Write concisely. 
- Phrases to avoid: "load-bearing", "gated", "the X is real", "X is doing a lot of work", "it's not X, it's Y", "genuinely", "{Ticket}/{fix} lands" / "{fix} has landed"
- Do not include ticket IDs or phases in any code, including unit tests.
- Avoid mannered prose: writing that uses metaphor or a striking phrase where a plain statement would do. Examples: "a dial worth turning" for "a parameter worth varying"; "this point earns its keep" for "this point still matters." Such phrases draw attention to the writing rather than the idea, and they are less precise, because a metaphor carries associations the writer did not intend. When a literal phrase is available, use it.
