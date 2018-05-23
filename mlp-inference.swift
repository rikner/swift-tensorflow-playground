#!/usr/bin/env swift -O

import TensorFlow

struct MLPClassifier {
    var w1 = Tensor<Float>(shape: [2, 4], repeating: 0.1)
    var w2 = Tensor<Float>(shape: [4, 1], scalars: [0.4, -0.5, -0.5, 0.4])
    var b1 = Tensor<Float>([0.2, -0.3, -0.3, 0.2])
    var b2 = Tensor<Float>([[0.4]])
            
    let activationFunction = hyperbolicTangentFunction

    func prediction(for x: Tensor<Float>) -> Tensor<Float> {
        // The ⊗ operator performs matrix multiplication.
        let weightedSum1 = (x ⊗ w1 + b1)
        let o1 = activationFunction(weightedSum1)
        
        let weightedSum2 = (o1 ⊗ w2 + b2)
        let o2 = activationFunction(weightedSum2)
        
        return o2
    }
}

    
// logistic function as activation function
private func logisticFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return pow((1 + exp(-input)), Float(-1))
}

// hyperbolic tangent as activation function
private func hyperbolicTangentFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return tanh(input)
}

let input = Tensor<Float>([[0.2, 0.8]])
let classifier = MLPClassifier()
let prediction = classifier.prediction(for: input)
print(prediction)