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

import TensorFlow

/// Returns `x`.
public func identity<T>(_ x: T) -> T { x }

/// Returns the elements of `0..<dataset.count`, in that order if `shuffled == false`,
/// and randomly shuffled otherwise.
// Note: dataset is inout because this is required by `Batcher`
public func defaultSample<C: Collection>(on dataset: inout C, shuffled: Bool) -> [Int] {
    return shuffled ? Array(0..<dataset.count).shuffled() : Array(0..<dataset.count)
}

/// Main struct to collate the samples from a dataset into batches
public struct Batcher<C: Collection> where C.Index == Int {
    /// The dataset to get the batches from.
    public var dataset: C
    /// The size of each batch.
    public var batchSize: Int
    /// Optionally set a limit to the number of threads used.
    public var threadsLimit: Int? = nil
    /// If `true`, shuffle the dataset at each iteration.
    public var shuffle: Bool = false
    /// If `true`, drop the last batch if it has less elements than batchSize.
    public var dropLast: Bool = false
    /// Hook to customize the way indices are sampled at each iteration.
    public let sampleIndices: (inout C, Bool) -> [Int]
    /// Hook to add padding to the samples before they are collated.
    public let padSamples: ([C.Element]) -> [C.Element]
    /// Hook to customize how the samples are collated.
    public let collateSamples: ([C.Element]) -> C.Element
    
    /// Returns the number of batches contained in the `Batcher`.
    public var count: Int {
        let nSamples = dataset.count
        return nSamples / batchSize + (nSamples % batchSize == 0 || dropLast ? 0 : 1)
    }
    
    public init(
        on dataset: C, 
        batchSize: Int, 
        threadsLimit: Int? = nil, 
        shuffle: Bool = false, 
        dropLast: Bool = false,
        sampleIndices: @escaping (inout C, Bool) -> [Int] = defaultSample,
        padSamples: @escaping ([C.Element]) -> [C.Element] = identity,
        collateSamples: @escaping ([C.Element]) -> C.Element
    ) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.threadsLimit = threadsLimit
        self.shuffle = shuffle
        self.dropLast = dropLast
        self.sampleIndices = sampleIndices
        self.padSamples = padSamples
        self.collateSamples = collateSamples
    }
    
    // To iterate through the batches
    public func sequenced() -> BatchIterator<C> {
        return BatchIterator(self)
    }
}

// Iterator through a Batcher
public struct BatchIterator<C: Collection>: IteratorProtocol, Sequence where C.Index == Int{
    /// Batcher to iterate through.
    var b: Batcher<C>
    /// Indices that will be used to go through the dataset of `b`.
    let indices: [Int]
    /// The length of the underlying dataset.
    let samplesCount: Int
    /// Where we are at in the dataset.
    var pos: Int = 0
    
    init(_ b: Batcher<C>) { 
        self.b = b
        indices = b.sampleIndices(&self.b.dataset, b.shuffle)
        samplesCount = b.dataset.count
        pos = 0
    }
    
    /// Returns the next batch
    public mutating func next() -> C.Element? {
        guard pos < samplesCount else { return nil }
        let end = Swift.min(pos + b.batchSize, samplesCount)
        if (end - pos) < b.batchSize && b.dropLast { return nil }
        // The idea is to have samples processed and collated on the CPU before moving to the host.
        // This part has not been optimized yet
        return withDevice(.cpu) { () -> C.Element in
            let n = b.threadsLimit == nil ? 1 : (end-pos) / b.threadsLimit
            let samples = Array(pos..<end).concurrentMap(minBatchSize: n) {
                b.dataset[indices[$0]]
            }
            pos = end
            return b.collateSamples(b.padSamples(samples))
        }
    }
}

/// Collate function when `S` conforms to Collatable
public func defaultCollate<S: Collatable>(_ batch: [S]) -> S {
    return S(collating: batch)
}

public extension Batcher where C.Element: Collatable {
    /// Add default to collateSamples when the dataset elements conform to Collatable
    init(
        on dataset: C, 
        batchSize: Int, 
        threadsLimit: Int? = nil, 
        shuffle: Bool = false, 
        dropLast: Bool = false,
        sampleIndices: @escaping (inout C, Bool) -> [Int] = defaultSample,
        padSamples: @escaping ([C.Element]) -> [C.Element] = identity,
        collateSamples: @escaping ([C.Element]) -> C.Element = defaultCollate
    ) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.threadsLimit = threadsLimit
        self.shuffle = shuffle
        self.dropLast = dropLast
        self.sampleIndices = sampleIndices
        self.padSamples = padSamples
        self.collateSamples = collateSamples
    }
}