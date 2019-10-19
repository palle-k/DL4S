# DL4S

[![CocoaPods](https://img.shields.io/cocoapods/v/DL4S.svg)](https://cocoapods.org/pods/DL4S)
![CocoaPods](https://img.shields.io/cocoapods/p/DL4S.svg)
[![license](https://img.shields.io/github/license/palle-k/DL4S.svg)](https://github.com/palle-k/DL4S/blob/master/License)

This framework provides reverse mode automatic differentiation,
vectorized implementations of common matrix and vector operators and high level neural network operations,
such as convolution, recurrent units, and more.

## Overview
1. [Installation](#installation)
2. [Features](#features)
    1. [Layers](#layers)
    2. [Optimizers](#optimizers)
    3. [Losses](#losses)
    4. [Tensor Operations](#tensor-operations)
    5. [Engines](#engines)
    6. [Architectures](#architectures)
3. [Examples](#examples)
    1. [Convolutional Networks](#convolutional-networks)
    2. [Recurrent Network (LSTM)](#recurrent-networks)
    3. [Generative Adversarial Network](#generative-adversarial-networks)


## Installation

### CocoaPods

```ruby
target 'Your-App-Name' do
    use_frameworks!
    pod 'DL4S', '~> 0.1.0'
end
```


### Swift Package Manager
Add the dependency to your `Package.swift` file:

```swift
.package(url: "https://github.com/palle-k/DL4S.git", .branch("master"))
```

Then add `DL4S` as a dependency to your target:

```swift
.target(name: "MyPackage", dependencies: ["DL4S"])
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

- [x] CPU (Accelerate framework)
- [ ] GPU (Metal)

</p>
</details>

<details>
<summary>
Architectures
</summary>
<p>

Default implementations are provided for the following architectures:

- [x] ResNet (currently only ResNet-18)
- [x] VGG
- [x] AlexNet

</p>
</details>


## Examples

### Arithmetic & Differentiation

DL4S provides a high-level interface to many vectorized operations on tensors.

```swift
let a = XTensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
let prod = a.transposed().matMul(a)
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
Therefore, second derivatives (diagonal of Hessian) can be computed by computing the gradient of the gradient.

When higher order derivatives are required, the compute graph of the backwards pass has to be explicitly retained.
Otherwise it will be automatically discarded.
```swift
let t = XTensor<Float, CPU>([1,2,3,4], requiresGradient: true)

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
let model = Sequential<Float, CPU>(
    Conv2D(inputChannels: 1, outputChannels: 6, kernelSize: 5, padding: 0).asAny(), // 4x24x24
    Relu().asAny(),
    MaxPool2D(windowSize: 2, stride: 2).asAny(), // 4x12x12
    Conv2D(inputChannels: 6, outputChannels: 16, kernelSize: 5, padding: 0).asAny(), // 16x8x8
    Relu().asAny(),
    MaxPool2D(windowSize: 2, stride: 2).asAny(), // 16x4x4
    Flatten().asAny(), // 256
    Dense(inputFeatures: 256, outputFeatures: 120).asAny(),
    Relu().asAny(),
    Dense(inputFeatures: 120, outputFeatures: 10).asAny(),
    Softmax().asAny()
)

let optimizer = Adam(parameters: model.trainableParameters, learningRate: 0.001)

// Single iteration of minibatch gradient descent
optimizer.zeroGradient()

let batch: Tensor<Float, CPU> = ... // shape: [batchSize, 28, 28]
let y_true: Tensor<Int32, CPU> = ... // shape: [batchSize]

let pred = model.forward(batch)
let loss = categoricalCrossEntropy(expected: y_true, actual: pred)

loss.backwards()
optimizer.step()
```

### Recurrent Networks

Example for MNIST classification

The LSTM scans the image from top to bottom and uses the final hidden state for classification.

```swift
let model = Sequential<Float, CPU>(
    LSTM(inputSize: 28, hiddenSize: 128).asAny(),
    Dense(inputFeatures: 128, outputFeatures: 10).asAny(),
    Softmax().asAny()
)

let optimizer = Adam(parameters: model.trainableParameters, learningRate: 0.001)

// Single iteration of minibatch gradient descent
optimizer.zeroGradient()

let batch: Tensor<Float, CPU> = ... // shape: [batchSize, 28, 28]
let y_true: Tensor<Int32, CPU> = ... // shape: [batchSize]

let x = batch.permuted(to: 1, 0, 2) // Swap first and second axis
let pred = model.forward(x)
let loss = categoricalCrossEntropy(expected: y_true, actual: pred)

loss.backwards()
optimizer.step()
```

### Generative Adversarial Networks

Example to generate random images similar to those in MNIST

```swift
let images: Tensor<Float, CPU> = ... // shape [numImages x 28 x 28]

let d1 = Dropout<Float, CPU>(rate: 0.5)
let d2 = Dropout<Float, CPU>(rate: 0.5)

let generator = Sequential<Float, CPU>(
    Dense(inputFeatures: 20, outputFeatures: 200).asAny(),
    Tanh().asAny(),
    d1.asAny(),
    Dense(inputFeatures: 200, outputFeatures: 800).asAny(),
    Tanh().asAny(),
    d2.asAny(),
    Dense(inputFeatures: 800, outputFeatures: 28 * 28).asAny(),
    Sigmoid().asAny(),
    Reshape(shape: 28, 28).asAny()
)

let discriminator = Sequential<Float, CPU>(
    Flatten().asAny(),
    Dense(inputFeatures: 28 * 28, outputFeatures: 400).asAny(),
    Tanh().asAny(),
    Dense(inputFeatures: 400, outputFeatures: 100).asAny(),
    Tanh().asAny(),
    Dense(inputFeatures: 100, outputFeatures: 1).asAny(),
    Sigmoid().asAny()
)

let network = Sequential(generator.asAny(), discriminator.asAny())


let optimGen = Adam(parameters: generator.trainableParameters, learningRate: 0.0003)
let optimDis = Adam(parameters: discriminator.trainableParameters, learningRate: 0.0003)

let batchSize = 32
let epochs = 10_000
let regularization: Float = 0.001

let genInputs = Tensor<Float, CPU>(repeating: 0, shape: batchSize, 20)

for epoch in 1 ... epochs {
    optimDis.zeroGradient()

    let real = Random.minibatch(from: images, count: batchSize)
    Random.fillNormal(genInputs)

    let realResult = discriminator.forward(real)
    let fakeResult = network.forward(genInputs)

    let dRegLoss = optimDis.parameters.map {l2loss($0, loss: regularization)}.reduce(0, +)
    let discriminatorLoss = -mean(log(realResult)) - mean(log(1 - fakeResult)) + dRegLoss

    discriminatorLoss.backwards()
    optimDis.step()

    var generatorLoss = Tensor<Float, CPU>(0)

    for _ in 0 ..< 4 {
        optimGen.zeroGradient()
        Random.fillNormal(genInputs)

        let genResult = network.forward(genInputs)

        let gRegLoss = optimGen.parameters.map {l2loss($0, loss: regularization)}.reduce(0, +)
        generatorLoss = -0.5 * mean(log(genResult)) + gRegLoss // heuristic non-saturating loss

        generatorLoss.backwards()
        optimGen.step()
    }

    if epoch % 100 == 0 {
        print(" [\(epoch)/\(epochs)] loss d: \(discriminatorLoss.item), g: \(generatorLoss.item)")
    }
}

Random.fillNormal(genInputs)
let genResult = generator.forward(genInputs)

for i in 0 ..< batchSize {
    let slice = genResult[i].T.unsqueeze(at: 0)
    guard let image = NSImage(slice), let imgData = image.tiffRepresentation else {
        continue
    }
    guard let rep = NSBitmapImageRep.init(data: imgData) else {
        continue
    }
    let png = rep.representation(using: .png, properties: [:])
    try? png?.write(to: URL(fileURLWithPath: "generated_\(i).png"))
}

```
