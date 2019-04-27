import DL4S
import Foundation
import Commander


func validatePositive<N: Numeric & Comparable>(message: String) -> (N) throws -> N {
    return { num in
        if num >= N.zero {
            return num
        } else {
            throw NSError(domain: "NMTSwiftErrorDomain", code: 2, userInfo: [NSLocalizedDescriptionKey: message])
        }
    }
}

func validateRange<N: Numeric & Comparable>(_ range: ClosedRange<N>, message: String) -> (N) throws -> N {
    return { num in
        if range ~= num {
            return num
        } else {
            throw NSError(domain: "NMTSwiftErrorDomain", code: 3, userInfo: [NSLocalizedDescriptionKey: message])
        }
    }
}

func validatePathExists(isDirectory: Bool? = nil) -> (String) throws -> String {
    return { path in
        var isActualDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isActualDirectory), isDirectory == nil || isDirectory == isActualDirectory.boolValue {
            return path
        } else {
            throw NSError(domain: "NMTSwiftErrorDomain", code: 1, userInfo: [NSLocalizedDescriptionKey: "Path '\(path)' does not exist."])
        }
    }
}


let group = Group { g in
    g.command(
        "train",
        // Required args
        Argument<String>("dataset", description: "Path to tab separated dataset file", validator: validatePathExists(isDirectory: false)),
        Argument<String>("destination", description: "Directory to write the final trained model files to", validator: validatePathExists(isDirectory: true)),
        // Model parameters
        Option<Float>("lr", default: 0.001, flag: nil, description: "Learning rate", validator: validatePositive(message: "Learning rate must be positive")),
        Option<Int>("latent_size", default: 1024, flag: nil, description: "Size of latent sentence representation", validator: validatePositive(message: "Latent size must be positive")),
        Option<Float>("forcing_rate", default: 0.5, flag: nil, description: "Teacher forcing rate", validator: validateRange(0 ... 1, message: "Teacher forcing rate must be between 0 and 1")),
        // Training parameters
        Option<Int>("iterations", default: 10000, flag: "i", description: "Number of training iterations", validator: validatePositive(message: "Number of training iterations must be positive")),
        Option<String>("checkpoint_dir", default: "./", flag: nil, description: "Directory to write checkpoints to", validator: validatePathExists(isDirectory: true)),
        Option<Int>("checkpoint_frequency", default: 2500, flag: nil, description: "Number of iterations between checkpoints (set to zero if no checkpoints should be created)", validator: validatePositive(message: "Number of iterations between checkpoints must be positive (set to zero if no checkpoints should be created)")),
        Option<String>("log_dir", default: "./", flag: nil, description: "Directory to write logs to", validator: validatePathExists(isDirectory: true)),
        description: "Trains a Seq2Seq model"
    ) { datasetPath, destinationPath, learningRate, latentSize, teacherForcingRate, iterations, checkpointDir, checkpointFrequency, logDir in
        
        print("Loading dataset...", terminator: "")
        fflush(stdout)
        let (english, german, examples) = try Language.pair(from: datasetPath)
        print(" Done.")
        
        print("Creating model...", terminator: "")
        fflush(stdout)
        let encoder = Encoder<Float, CPU>(inputSize: english.indexToWord.count, hiddenSize: latentSize)
        let decoder = Decoder<Float, CPU>(inputSize: german.indexToWord.count, hiddenSize: latentSize)
        print(" Done.")
        
        let optimEnc = Adam(parameters: encoder.trainableParameters, learningRate: learningRate)
        let optimDec = Adam(parameters: decoder.trainableParameters, learningRate: learningRate)
        
        let epochs = iterations
        
        let helper = Helper(encoder: encoder, decoder: decoder)
        
        var progressBar = DL4S.ProgressBar<String>(totalUnitCount: epochs, formatUserInfo: {$0}, label: "training")
        let writer = try SummaryWriter(destination: URL(fileURLWithPath: logDir))
        
        for i in 1 ... epochs {
            optimEnc.zeroGradient()
            optimDec.zeroGradient()
            
            let (eng, ger) = examples.randomElement()!
            
            let engIdxs = english.indexSequence(from: eng)
            let gerIdxs = german.indexSequence(from: ger)
            
            let encoded = helper.encode(sequence: engIdxs)
            
            let maxLength = gerIdxs.count
            
            let decoded: Tensor<Float, CPU>
            
            if Float.random(in: 0 ... 1) <= teacherForcingRate {
                decoded = helper.decodeSequence(fromInitialState: encoded, forcedResult: gerIdxs)
            } else {
                decoded = helper.decodeSequence(fromInitialState: encoded, initialToken: Int32(Language.startOfSentence), endToken: Int32(Language.endOfSentence), maxLength: maxLength)
            }
            
            let loss = helper.decodingLoss(forExpectedSequence: gerIdxs, actualSequence: decoded)
            loss.backwards()
            
            optimEnc.step()
            optimDec.step()
            
            // Prevents stack overflow when releasing compute graph lol
            loss.detachAll()
            
            let decodedIdxs = helper.sequence(from: decoded)
            let pair = "\(english.formattedSentence(from: engIdxs)) -> \(german.formattedSentence(from: decodedIdxs))"
            
            progressBar.next(userInfo: "[loss: \(loss)] \(pair)")
            writer.write(loss.item, named: "loss", at: i)
            
            if checkpointFrequency > 0 && i.isMultiple(of: checkpointFrequency) && i != epochs {
                let checkpointDir = URL(fileURLWithPath: checkpointDir)
                try encoder.saveWeights(to: URL(fileURLWithPath: "encoder_\(i).json", relativeTo: checkpointDir))
                try decoder.saveWeights(to: URL(fileURLWithPath: "decoder_\(i).json", relativeTo: checkpointDir))
            }
        }
        progressBar.complete()
        
        let destinationURL = URL(fileURLWithPath: destinationPath)
        try encoder.saveWeights(to: URL(fileURLWithPath: "encoder.json", relativeTo: destinationURL))
        try decoder.saveWeights(to: URL(fileURLWithPath: "decoder.json", relativeTo: destinationURL))
    }
    
    g.command(
        "eval",
        Argument<String>("dataset", description: "Path to tab separated dataset file", validator: validatePathExists(isDirectory: false)),
        Argument<String>("encoder_location", description: "Path to the stored weights of the encoder", validator: validatePathExists(isDirectory: false)),
        Argument<String>("decoder_location", description: "Path to the stored weights of the decoder", validator: validatePathExists(isDirectory: false)),
        Option<Int>("latent_size", default: 1024, flag: nil, description: "Size of latent sentence representation", validator: validatePositive(message: "Latent size must be positive")),
        Option<Int>("beam_count", default: 4, flag: nil, description: "Number of beams to use for decoding", validator: validatePositive(message: "Number of beams must be positive")),
        description: "Evaluate a trained seq2seq model."
    ) { datasetDir, encoderPath, decoderPath, latentSize, beamCount in
        print("Loading dataset...", terminator: "")
        fflush(stdout)
        let (english, german, _) = try Language.pair(from: datasetDir)
        print(" Done.")
        
        print("Creating model...", terminator: "")
        fflush(stdout)
        let encoder = Encoder<Float, CPU>(inputSize: english.indexToWord.count, hiddenSize: latentSize)
        let decoder = Decoder<Float, CPU>(inputSize: german.indexToWord.count, hiddenSize: latentSize)
        print(" Done.")
        
        print("Loading trained model...", terminator: "")
        fflush(stdout)
        try encoder.loadWeights(from: URL(fileURLWithPath: encoderPath))
        try decoder.loadWeights(from: URL(fileURLWithPath: decoderPath))
        print(" Done.")
        
        let helper = Helper(encoder: encoder, decoder: decoder)
        
        print("> ", terminator: "")
        fflush(stdout)
        while let line = readLine() {
            let translated = helper.translate(line, from: english, to: german)
            
            print(translated.joined(separator: "\n"))
            
            print("> ", terminator: "")
            fflush(stdout)
        }
    }
}

group.run()
