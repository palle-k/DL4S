//
//  Language.swift
//  DL4S
//
//  Created by Palle Klewitz on 15.03.19.
//

import Foundation
import DL4S


struct Language {
    static let startOfSentence: Int = 0
    static let endOfSentence: Int = 1
    
    var wordToIndex: [String: Int]
    var indexToWord: [Int: String]
    
    init(from examples: [String]) {
        let words = examples
            .flatMap {
                $0.components(separatedBy: .whitespaces)
            }
        let uniqueWords = Set(words).sorted()
        
        let enumerated = [(Language.startOfSentence, "<SOS>"), (Language.endOfSentence, "<EOS>")] + uniqueWords.enumerated().map {($0 + 2, $1)}
        
        indexToWord = Dictionary(uniqueKeysWithValues: enumerated.map {($0, $1)})
        wordToIndex = Dictionary(uniqueKeysWithValues: enumerated.map {($1, $0)})
    }
    
    static func pair(from path: String) throws -> (Language, Language, [(String, String)]) {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let string = String(data: data, encoding: .utf8)!
        
        let cleaned = string
            .lowercased()
            .replacingOccurrences(of: #"([.,?!])"#, with: " $1", options: .regularExpression)
            .replacingOccurrences(of: #"[0-9]+"#, with: "<NUM>", options: .regularExpression)
        
        let lines = cleaned.components(separatedBy: "\n").filter {!$0.isEmpty}
        let pairs = lines
            .map {$0.components(separatedBy: "\t")}
            .map {($0[0], $0[1])}
        
        let l1 = Language(from: pairs.map {$0.0})
        let l2 = Language(from: pairs.map {$1})
        
        return (l1, l2, pairs)
    }
    
    func formattedSentence(from sequence: [Int32]) -> String {
        let words = sequence.filter {$0 >= 2}.map(Int.init).compactMap {indexToWord[$0]}
        
        var result: String = ""
        
        for w in words {
            if let f = w.first, Set(".,!?").contains(f) {
                result.append(w)
            } else if result.isEmpty {
                result.append(w)
            } else {
                result.append(" \(w)")
            }
        }
        
        return result
    }
    
    func indexSequence(from sentence: String) -> [Int32] {
        let words = sentence.components(separatedBy: .whitespaces)
        let indices = words.compactMap {wordToIndex[$0]}.map(Int32.init)
        return [Int32(Language.startOfSentence)] + indices + [Int32(Language.endOfSentence)]
    }
}
