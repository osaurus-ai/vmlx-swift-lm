import Foundation
import MLX
import MLXLLM
import MLXLMCommon

@main
struct Bench {
    static func main() async throws {
        let modelDir = URL(fileURLWithPath: "/Users/eric/models/Qwen3.5-35B-A3B-4bit")
        
        print("Loading model from \(modelDir.path)...")
        
        print("ModelFactory loading...")
        let context = try await MLXLMCommon.loadModel(from: modelDir, using: TestTokenizerLoader())
        
        print("Model object created and weights loaded.")
        eval(context.model)
        
        print("Weights evaluated. Benchmarking decode...")
        
        let prompt = MLXArray(Array(0..<20))[.newAxis, .ellipsis]
        let input = LMInput(text: .init(tokens: prompt))
        let maxTokens = 100
        
        var params = GenerateParameters(maxTokens: maxTokens)
        params.enableCompiledDecode = true
        params.compiledMaxCacheLength = 4096
        
        var iterator = try TokenIterator(input: input, model: context.model, parameters: params)
        
        let start = CFAbsoluteTimeGetCurrent()
        var count = 0
        while let _ = iterator.next() {
            count += 1
            if count == 1 {
                print("First token generated.")
            }
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let tps = Double(count) / elapsed
        
        print(String(format: "Speed: %.2f tok/s (%.3fs for %d tokens)", tps, elapsed, count))
    }
}

struct TestTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        return TestTokenizer()
    }
}