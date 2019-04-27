//
//  Helpers.swift
//  DL4S
//
//  Created by Palle Klewitz on 14.03.19.
//

import Foundation
import DL4S


struct Helper<Element: RandomizableType, Device: DeviceType> {
    let encoder: Encoder<Element, Device>
    let decoder: Decoder<Element, Device>
    
    func encode(sequence: [Int32]) -> Tensor<Element, Device> {
        let t = Tensor<Int32, Device>(sequence)
        let h = encoder(t)
        return h
    }
    
    func decodeSequence(fromInitialState initialState: Tensor<Element, Device>, initialToken: Int32, endToken: Int32, maxLength: Int? = nil) -> Tensor<Element, Device> {
        //TODO: [Maybe] implement beam search with more than 1 beam.
        
        var hiddenState = initialState
        var token = initialToken
        var sequence: [Tensor<Element, Device>] = []
        for _ in maxLength.map({AnySequence(0 ..< $0)}) ?? AnySequence(0...) {
            let (n, h) = decoder.forward(input: Tensor(token), previousHidden: hiddenState)
            sequence.append(n) // n has a shape of [1, vocabularySize]
            hiddenState = h
            token = Int32(argmax(n))
            
            if token == endToken {
                break
            }
        }
        
        return stack(sequence) // [sequenceLength, vocabularySize]
    }
    
    func beamDecodeSequence(fromInitialState initialState: Tensor<Element, Device>, initialToken: Int32, endToken: Int32, beamCount: Int, maxLength: Int? = nil) -> [[Int32]] {
        precondition(beamCount >= 1, "Beam count must be at least 1.")
        
        var beams = Array(repeating: [initialToken], count: beamCount)
        var beamStates = Array(repeating: initialState, count: beamCount)
        var beamLikelihoods = [Element](repeating: 1.0, count: beamCount)
        
        for _ in maxLength.map({AnySequence(0 ..< $0)}) ?? AnySequence(0...) {
            var newHiddenStates: [Tensor<Element, Device>] = []
            var newLikelihoods: [[Element]] = []
            
            for i in 0 ..< beamCount {
                let input = beams[i].last!
                
                if input == endToken {
                    newLikelihoods.append([beamLikelihoods[i]])
                    newHiddenStates.append(beamStates[i])
                    continue
                }
                
                let prevHidden = beamStates[i]
                let (out, newHidden) = decoder.forward(input: Tensor([input]), previousHidden: prevHidden)
                
                let newProbas = Tensor(repeating: beamLikelihoods[i]) * out.view(as: -1)
                
                var probas: [Element] = []
                for i in 0 ..< newProbas.count {
                    probas.append(newProbas[i].item)
                }
                newLikelihoods.append(probas)
                newHiddenStates.append(newHidden)
            }
            
            var newBeams: [[Int32]] = []
            var newBeamStates = Array<Tensor<Element, Device>>()
            var newBeamLikelihoods = Array<Element>()
            
            let (bestIdxs, bestLikelihoods) = bestBeams(likelihoods: newLikelihoods)
            for i in 0 ..< beamCount {
                let (src, argmax) = bestIdxs[i]
                let max = bestLikelihoods[i]
                
                newBeams.append(beams[Int(src)] + [Int32(argmax)])
                newBeamStates.append(newHiddenStates[Int(src)])
                newBeamLikelihoods.append(max)
            }
            
            beams = newBeams
            beamLikelihoods = newBeamLikelihoods
            beamStates = newBeamStates
            
            if beams.allSatisfy({$0.last! == endToken}) {
                break
            }
        }
        
        return beams
    }
    
    private func bestBeams(likelihoods: [[Element]]) -> ([(Int, Int)], [Element]) {
        var bestLikelihoods: [Element] = []
        var bestIndices: [(Int, Int)] = []
        
        let beamCount = likelihoods.count
        
        for i in 0 ..< beamCount {
            for j in 0 ..< likelihoods[i].count {
                if bestLikelihoods.count < beamCount {
                    let idx = bestLikelihoods.insertOrdered(likelihoods[i][j])
                    bestIndices.insert((i, j), at: idx)
                } else if bestLikelihoods.first! < likelihoods[i][j] {
                    bestLikelihoods.removeFirst()
                    bestIndices.removeFirst()
                    let idx = bestLikelihoods.insertOrdered(likelihoods[i][j])
                    bestIndices.insert((i, j), at: idx)
                }
            }
        }
        
        return (bestIndices, bestLikelihoods)
    }
    
    func decodeSequence(fromInitialState initialState: Tensor<Element, Device>, forcedResult: [Int32]) -> Tensor<Element, Device> {
        let input = Tensor<Int32, Device>(forcedResult)
        
        return decoder.forwardFullSequence(input: input, initialHidden: initialState)
    }
    
    func decodingLoss(forExpectedSequence expectedSequence: [Int32], actualSequence: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let subsequence = Array(expectedSequence[0 ..< actualSequence.shape[0]])
        return categoricalCrossEntropy(expected: Tensor(subsequence), actual: actualSequence)
    }
    
    func sequence(from tensor: Tensor<Element, Device>) -> [Int32] {
        var s: [Int32] = []
        
        for i in 0 ..< tensor.shape[0] {
            s.append(Int32(argmax(tensor[i])))
        }
        
        return s
    }
    
    func translate(_ text: String, from sourceLanguage: Language, to destination: Language) -> [String] {
        let input = text
            .lowercased()
            .replacingOccurrences(of: #"([.,?!])"#, with: " $1", options: .regularExpression)
            .replacingOccurrences(of: #"[0-9]+"#, with: "<NUM>", options: .regularExpression)
        
        let inputIdxs = sourceLanguage.indexSequence(from: input)
        let encoded = encode(sequence: inputIdxs)
        let bestCandidates = beamDecodeSequence(fromInitialState: encoded, initialToken: Int32(Language.startOfSentence), endToken: Int32(Language.endOfSentence), beamCount: 4, maxLength: inputIdxs.count * 2 + 2)
        
        let sentences = bestCandidates.map(destination.formattedSentence(from:))
        return sentences
    }
}

fileprivate extension Array where Element: Comparable {
    mutating func insertOrdered(_ element: Element) -> Int {
        if let idx = self.firstIndex(where: {$0 > element}) {
            self.insert(element, at: idx)
            return idx
        } else {
            self.append(element)
            return self.count - 1
        }
    }
}
