# DL4S

This package contains an implementation of reverse mode autodifferentiation as well as vectorized implementations of common matrix and vector operations as well as high level neural network operations.

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
- [x] Add, Subtract, Multiply, Divide, Negate
- [x] exp, log, sqrt
- [x] Subscripting
- [x] Transpose, Axis permutation
- [x] Sum, Max


## Examples

```swift
let ((train_inputs, train_expected), (validation_inputs, validation_expected)) = MNistTest.images(from: "/Users/Palle/Downloads/")

let model = Sequential<Float>(
    Flatten().asAny(),
    Dense(inputFeatures: 28 * 28, outputFeatures: 500).asAny(),
    Tanh().asAny(),
    Dense(inputFeatures: 500, outputFeatures: 300).asAny(),
    Tanh().asAny(),
    Dense(inputFeatures: 300, outputFeatures: 10).asAny(),
    Softmax().asAny()
)

let epochs = 5_000
let batchSize = 128

let optimizer = Adam(parameters: model.parameters, learningRate: 0.001)

for epoch in 1 ... epochs {
    optimizer.zeroGradient()
    let (batch, expected) = Random.minibatch(from: train_inputs, labels: train_expected, count: batchSize)
    
    let y_pred = model.forward(batch)
    let y_true = expected
    
    let loss = categoricalCrossEntropy(expected: y_true, actual: y_pred)
    
    loss.backwards()
    optimizer.step()
    
    if epoch % 100 == 0 {
        let avgLoss = loss.item
        print("[\(epoch)/\(epochs)] loss: \(avgLoss)")
    }
}

var correctCount = 0

for i in 0 ..< ds_val.0.shape[0] {
    let x = validation_inputs[i].unsqueeze(at: 0)
    let pred = argmax(model.forward(x).squeeze())
    let actual = Int(validation_expected.item)

    if pred == actual {
        correctCount += 1
    }
}

let accuracy = Float(correctCount) / Float(ds_val.0.shape[0])
print("Accuracy: \(accuracy)")

try model.saveWeights(to: URL(fileURLWithPath: "mnist_params.json"))
```
