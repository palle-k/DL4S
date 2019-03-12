# DL4S

This package contains an implementation of reverse mode automatic differentiation as well as vectorized implementations of common matrix and vector operations as well as high level neural network operations.

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



### Engines.
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
