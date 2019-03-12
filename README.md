# DL4S

This package contains an implementation of reverse mode automatic differentiation, 
vectorized implementations of common matrix and vector operators and high level neural network operations.

This project is a work in progress. While all the shown examples work, there are important things missing, such as GPU acceleration and convolution operators.

## Overview
1. [Installation](#installation)
2. [Features](#features)
    1. [Layers](#layers)
    2. [Optimizers](#optimizers)
    3. [Losses](#losses)
    4. [Tensor Operations](#tensor-operations)
    5. [Engines](#engines)
3. [Examples](#examples)
    1. [Feed Forward Network](#feed-forward-networks)
    2. [Recurrent Network (LSTM)](#recurrent-networks)
    3. [Generative Adversarial Network](#generative-adversarial-networks)


## Installation
Add the dependency to your `Package.swift` file:

```swift
.package(url: "https://github.com/palle-k/DL4S.git", .branch("master"))
```

Then add `DL4S` as a dependency to your target:

```swift
.target(name: "MyPackage", dependencies: ["DL4S"])
```

## Features

### Layers
- [x] Dense/Linear/Fully Connected
- [x] LSTM
- [x] Gated Recurrent Unit (GRU)
- [x] Relu
- [x] Tanh
- [x] Sigmoid
- [x] Softmax
- [x] Embedding
- [x] Batchnorm
- [x] Lambda 
- [x] Sequential
- [x] Dropout
- [ ] Convolution
- [ ] Pooling

### Optimizers
- [x] SGD
- [x] Momentum
- [x] Adam

### Losses
- [x] Binary Cross-Entropy
- [x] Categorical Cross-Entropy
- [x] MSE
- [x] L1 & L2 regularization

### Tensor Operations
- [x] add
- [x] sub
- [x] mul 
- [x] div
- [x] neg
- [x] exp
- [x] log
- [x] sqrt
- [x] subscript
- [x] subscript range
- [x] transpose
- [x] axis permute
- [x] sum
- [x] max
- [x] reduce sum
- [ ] reduce max
- [ ] conv2d
- [ ] max pool
- [ ] avg pool


### Engines
- [x] CPU (Accelerate framework)
- [ ] GPU (Metal)


## Examples

### Arithmetic & Differentiation

```swift
let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
let prod = mmul(a.T, a)
let s = sum(prod)
let l = log(s)
print(l) // 5.1873856


// Backpropagate
l.backwards()

print(a.gradientDescription!)
/*
[[0.03351955, 0.03351955],
 [0.07821229, 0.07821229],
 [0.12290502, 0.12290502]]
*/
```

### Feed Forward Networks

Example for MNIST classification

```swift
let model = Sequential<Float, CPU>(
    Flatten().asAny() // Flatten batchSize x 28 x 28 image to batchSize x 784 vector
    Dense(inputFeatures: 784, outputFeatures: 500).asAny(),
    Relu().asAny(),
    Dense(inputFeatures: 500, outputFeatures: 300).asAny(),
    Relu().asAny(),
    Dense(inputFeatures: 300, outputFeatures: 10).asAny(),
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

        let gRegLoss = optimDis.parameters.map {l2loss($0, loss: regularization)}.reduce(0, +)
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
