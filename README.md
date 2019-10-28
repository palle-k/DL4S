# DL4S
[![license](https://img.shields.io/github/license/palle-k/DL4S.svg)](https://github.com/palle-k/DL4S/blob/master/License)
[![tag](https://img.shields.io/github/v/tag/palle-k/DL4S)](https://github.com/palle-k/DL4S/releases)

This framework provides reverse mode automatic differentiation,
vectorized implementations of common matrix and vector operators and high level neural network operations,
such as convolution, recurrent units, and more.

## Overview
1. [Installation](#installation)
2. [Features](#features)
    1. Layers
    2. Optimizers
    3. Losses
    4. Tensor Operations
    5. Engines
    6. Architectures
3. [Examples](#examples)


## Installation

### iOS / tvOS / macOS

1. In Xcode, select "File" > "Swift Packages" > "Add Package Dependency"
2. Enter `https://github.com/palle-k/DL4S.git` into the Package URL field and click "Next".
3. Select "Version", "Up to Next Major", 1.0.0 and click "Next".
4. Enable the Package Product DL4S, your app in the "Add to Target" column and click "Next". 

**Note**: Installation via CocoaPods is no longer supported for newer versions.

### Swift Package
Add the dependency to your `Package.swift` file:

```swift
.package(url: "https://github.com/palle-k/DL4S.git", .branch("master"))
```

Then add `DL4S` as a dependency to your target:

```swift
.target(name: "MyPackage", dependencies: ["DL4S"])
```

#### MKL / IPP Support

DL4S can be accelerated with Intel's Math Kernel Library and Integrated Performance Primitives ([Installation Instructions](https://software.intel.com/en-us/get-started-with-mkl-for-linux)).

On Apple devices, DL4S uses vectorized functions provided by the builtin Accelerate framework by default.
If no acceleration library is available, a generic fallback is used.

Compiling with MKL/IPP:
```bash
export MKLROOT=/opt/intel/mkl  # change depending on your system configuration
export IPPROOT=/opt/intel/ipp

export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$IPPROOT/lib/intel64:$LD_LIBRARY_PATH

swift build -c release \
    -Xlinker -L${MKLROOT}/lib/intel64 \
    -Xlinker -L${IPPROOT}/lib/intel64 \
    -Xlinker -lmkl_intel_lp64 \
    -Xlinker -lmkl_sequential \
    -Xlinker -lmkl_core \
    -Xlinker -lpthread \
    -Xlinker -lippcore \
    -Xlinker -lippvm \
    -Xlinker -lipps \                
    -Xlinker -lm \
    -Xlinker -ldl \
    -Xcc -m64 \
    -Xcc -DMKL_ENABLE \
    -Xcc -I${MKLROOT}/include \
    -Xcc -I${IPPROOT}/include
```




## Features

<details>
<summary>
Layers
</summary>
<p>

- [x] Convolution
- [x] Dense/Linear/Fully Connected
- [x] LSTM
- [x] Gated Recurrent Unit (GRU)
- [x] Vanilla RNN
- [x] Bidirectional RNNs
- [x] Max Pooling
- [x] Average Pooling
- [x] Relu
- [x] Tanh
- [x] Sigmoid
- [x] Softmax
- [x] Embedding
- [x] Batch Norm
- [x] Layer Norm
- [x] Lambda 
- [x] Sequential
- [x] Dropout

</p>
</details>

<details>
<summary>
Optimizers
</summary>
<p>

- [x] SGD
- [x] Momentum
- [x] Adam
- [x] AdaGrad
- [x] AdaDelta
- [x] RMSProp

</p>
</details>

<details>
<summary>
Losses
</summary>
<p>

- [x] Binary Cross-Entropy
- [x] Categorical Cross-Entropy
- [x] MSE
- [x] L1 & L2 regularization

</p>
</details>

<details>
<summary>
Tensor Operations
</summary>
<p>

- [x] broadcast-add
- [x] broadcast-sub
- [x] broadcast-mul 
- [x] broadcast-div
- [x] matmul
- [x] neg
- [x] exp
- [x] pow
- [x] log
- [x] sqrt
- [x] sin
- [x] cos
- [x] tan
- [x] tanh
- [x] l1 norm
- [x] l2 norm
- [x] sum
- [x] max
- [x] relu
- [x] leaky relu
- [x] reduce sum
- [x] reduce max
- [x] conv2d
- [x] max pool
- [x] avg pool
- [x] subscript
- [x] subscript range
- [x] transpose
- [x] axis permute
- [x] reverse
- [x] im2col
- [x] col2im
- [x] stack / concat

</p>
</details>

<details>
<summary>
Engines
</summary>
<p>

- [x] CPU (Accelerate framework for Apple Devices)
- [x] CPU (Intel Math Kernel Library and Integrated Performance Primitives)
- [x] CPU (Generic)
- [ ] GPU (Metal)

For an experimental, early stage Metal implementation, check out `feature/metal`.

</p>
</details>

<details>
<summary>
Architectures
</summary>
<p>

Default implementations are provided for the following architectures:

- [x] ResNet18
- [x] VGG (11, 13, 16, 19)
- [x] AlexNet

</p>
</details>


## Examples

Some high level examples have been implemented in other repositories:

- [Neural Machine Translation](https://github.com/palle-k/Seq2Seq-DL4S) based on seq2seq with Attention
- [Generative Adversarial Networks](https://github.com/palle-k/DL4S-WGAN-GP) - Wasserstein GAN with Gradient Penalty (WGAN-GP)

### Arithmetic & Differentiation

DL4S provides a high-level interface to many vectorized operations on tensors.

```swift
let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
let prod = a.transposed().matrixMultipled(with: a)
let s = prod.reduceSum()
let l = log(s)
print(l) // 5.1873856
```

When a tensor is marked to require a gradient, a compute graph will be captured. 
The graph stores all operations, which use that tensor directly or indirectly as an operand.

It is then possible to backpropagate through that graph using the `gradients(of:)` function:
```swift
// Backpropagate
let dl_da = l.gradients(of: [a])[0]

print(dl_da)
/*
[[0.034, 0.034]
 [0.078, 0.078]
 [0.123, 0.123]]
*/
```

#### Second derivatives

The operations used during backpropagation are themselves differentiable. 
Therefore, second derivatives can be computed by computing the gradient of the gradient.

When higher order derivatives are required, the compute graph of the backwards pass has to be explicitly retained.
```swift
let t = Tensor<Float, CPU>([1,2,3,4], requiresGradient: true)

let result = t * t * t
print(result) // [1, 8, 27, 64]

let grad = result.gradients(of: [t], retainBackwardsGraph: true)[0]
print(grad) // [3, 12, 27, 48]

let secondGrad = grad.gradients(of: [t], retainBackwardsGraph: true)[0]
print(secondGrad) // [6, 12, 18, 24]

let thirdGrad = secondGrad.gradients(of: [t])[0]
print(thirdGrad) // [6, 6, 6, 6]
```


### Convolutional Networks

Example for MNIST classification

```swift
// Input must be 1x28x28
var model = Sequential {
   Convolution2D<Float, CPU>(inputChannels: 1, outputChannels: 6, kernelSize: (5, 5))
   Relu<Float, CPU>()
   MaxPool2D<Float, CPU>(windowSize: 2, stride: 2)
   
   Convolution2D<Float, CPU>(inputChannels: 6, outputChannels: 16, kernelSize: (5, 5))
   Relu<Float, CPU>()
   MaxPool2D<Float, CPU>(windowSize: 2, stride: 2)
   
   Flatten<Float, CPU>()
   
   Dense<Float, CPU>(inputSize: 256, outputSize: 120)
   Relu<Float, CPU>()
   
   Dense<Float, CPU>(inputSize: 120, outputSize: 10)
   Softmax<Float, CPU>()
}

var optimizer = Adam(model: model, learningRate: 0.001)

// Single iteration of minibatch gradient descent
let batch: Tensor<Float, CPU> = ... // shape: [batchSize, 28, 28]
let y_true: Tensor<Int32, CPU> = ... // shape: [batchSize]

// use optimizer.model, not model
let pred = optimizer.model(batch)
let loss = categoricalCrossEntropy(expected: y_true, actual: pred)

let gradients = loss.gradients(of: optimizer.model.parameters)
optimizer.update(along: gradients)
```

### Recurrent Networks

Example for MNIST classification

The Gated Reccurent Unit scans the image from top to bottom and uses the final hidden state for classification.

```swift
let model = Sequential {
    GRU<Float, CPU>(inputSize: 28, hiddenSize: 128, direction: .forward)
    Lambda<GRU<Float, CPU>.Outputs, Tensor<Float, CPU>, Float, CPU> { inputs in
        inputs.0
    }
    Dense<Float, CPU>(inputSize: 128, outputSize: 10)
    Softmax<Float, CPU>()
}

var optimizer = Adam(model: model, learningRate: 0.001)

let batch: Tensor<Float, CPU> = ... // shape: [batchSize, 28, 28]
let y_true: Tensor<Int32, CPU> = ... // shape: [batchSize]

let x = batch.permuted(to: 1, 0, 2) // Swap first and second axis
let pred = optimizer.model(x)
let loss = categoricalCrossEntropy(expected: y_true, actual: pred)

let gradients = loss.gradients(of: optimizer.model.parameters)
optimizer.update(along: gradients)
```
