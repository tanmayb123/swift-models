// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Datasets
import ImageClassificationModels
import TensorFlow
import x10_tensor

private func TF<T>(_ x: TensorFlow.Tensor<T>) -> x10_tensor.Tensor<T> {
  return x10_tensor.Tensor<T>(shape: x10_tensor.TensorShape(x.shape.dimensions), scalars: x.scalars)
}

let batchSize = 10

let dataset = CIFAR10(batchSize: batchSize)

// Use the network sized for CIFAR-10
var model = ResNet(classCount: 10, depth: .resNet56, downsamplingInFirstStage: false)

// the classic ImageNet optimizer setting diverges on CIFAR-10
// let optimizer = SGD(for: model, learningRate: 0.1, momentum: 0.9)
let optimizer = x10_tensor.SGD(for: model, learningRate: 0.001)

print("Starting training...")

for epoch in 1...10 {
    x10_tensor.Context.local.learningPhase = .training
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    for batch in dataset.training.sequenced() {
        let (images, labels) = (batch.first, batch.second)
        let x10Labels = TF(labels)
        let x10Images = TF(images)
        let (loss, gradients) = valueWithGradient(at: model) { model -> x10_tensor.Tensor<Float> in
            let logits = model(x10Images)
            return softmaxCrossEntropy(logits: logits, labels: x10Labels)
        }
        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        optimizer.update(&model, along: gradients)
        LazyTensorBarrier()
    }

    x10_tensor.Context.local.learningPhase = .inference
    var testLossSum: Float = 0
    var testBatchCount = 0
    var correctGuessCount = 0
    var totalGuessCount = 0
    for batch in dataset.test.sequenced() {
        let (images, labels) = (batch.first, batch.second)
        let x10Labels = TF(labels)
        let x10Images = TF(images)
        let logits = model(x10Images)
        testLossSum += softmaxCrossEntropy(logits: logits, labels: x10Labels).scalarized()
        testBatchCount += 1

        let correctPredictions = logits.argmax(squeezingAxis: 1) .== x10Labels
        correctGuessCount = correctGuessCount
            + Int(
                x10_tensor.Tensor<Int32>(correctPredictions).sum().scalarized())
        totalGuessCount = totalGuessCount + batchSize
        LazyTensorBarrier()
    }

    let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
    print(
        """
        [Epoch \(epoch)] \
        Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
        Loss: \(testLossSum / Float(testBatchCount))
        """
    )
}
