import CoreImage
import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import Testing

@Suite("VLM processor cache scope salt propagation", .serialized)
struct VLMProcessorCacheScopeSaltTests {
    private static func decode<T: Decodable>(_ json: String, as type: T.Type = T.self) throws -> T {
        try JSONDecoder.json5().decode(T.self, from: Data(json.utf8))
    }

    @Test("GlmOcr processor threads reasoning cache scope on text-only inputs")
    func glmOcrThreadsCacheScopeSalt() async throws {
        try await MLXMetalTestLock.withLock {
            let config: GlmOcrProcessorConfiguration = try Self.decode("""
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "merge_size": 2,
              "patch_size": 14,
              "temporal_patch_size": 1,
              "size": {"shortest_edge": 56, "longest_edge": 56}
            }
            """)
            let processor = GlmOcrProcessor(config, tokenizer: TestTokenizer())

            let input = try await processor.prepare(input: UserInput(
                prompt: "Read this.",
                additionalContext: ["enable_thinking": false]))

            #expect(input.cacheScopeSalt == "reasoning=off")
        }
    }

    @Test("Idefics3 processor threads reasoning cache scope on text-only inputs")
    func idefics3ThreadsCacheScopeSalt() async throws {
        try await MLXMetalTestLock.withLock {
            let config: Idefics3ProcessorConfiguration = try Self.decode("""
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "size": {"longest_edge": 384},
              "image_seq_len": 64
            }
            """)
            let processor = Idefics3Processor(config, tokenizer: TestTokenizer())

            let input = try processor.prepare(input: UserInput(
                prompt: "Describe.",
                additionalContext: ["enable_thinking": false]))

            #expect(input.cacheScopeSalt == "reasoning=off")
        }
    }

    @Test("Pixtral processor threads reasoning cache scope on text-only inputs")
    func pixtralThreadsCacheScopeSalt() async throws {
        try await MLXMetalTestLock.withLock {
            let config: PixtralProcessorConfiguration = try Self.decode("""
            {
              "image_processor": {
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "size": {"longest_edge": 56},
                "patch_size": 14
              },
              "image_token": "<image>",
              "patch_size": 14
            }
            """)
            let processor = PixtralProcessor(config, tokenizer: TestTokenizer())

            let input = try processor.prepare(input: UserInput(
                prompt: "Describe.",
                additionalContext: ["enable_thinking": false]))

            #expect(input.cacheScopeSalt == "reasoning=off")
        }
    }

    @Test("PaliGemma processor threads reasoning cache scope on image inputs")
    func paliGemmaThreadsCacheScopeSalt() async throws {
        try await MLXMetalTestLock.withLock {
            let config: PaliGemmaProcessorConfiguration = try Self.decode("""
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "size": {"width": 2, "height": 2},
              "image_seq_length": 1
            }
            """)
            let processor = PaliGemmaProcessor(config, tokenizer: TestTokenizer())
            let image = CIImage(color: .red)
                .cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2))

            let input = try processor.prepare(input: UserInput(
                prompt: "Describe.",
                images: [.ciImage(image)],
                additionalContext: ["enable_thinking": false]))

            #expect(input.cacheScopeSalt == "reasoning=off")
        }
    }

    @Test("unrelated VLM context does not fragment cache keys")
    func unrelatedContextDoesNotCreateCacheScopeSalt() async throws {
        try await MLXMetalTestLock.withLock {
            let glmConfig: GlmOcrProcessorConfiguration = try Self.decode("""
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "merge_size": 2,
              "patch_size": 14,
              "temporal_patch_size": 1,
              "size": {"shortest_edge": 56, "longest_edge": 56}
            }
            """)
            let ideficsConfig: Idefics3ProcessorConfiguration = try Self.decode("""
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "size": {"longest_edge": 384},
              "image_seq_len": 64
            }
            """)
            let pixtralConfig: PixtralProcessorConfiguration = try Self.decode("""
            {
              "image_processor": {
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "size": {"longest_edge": 56},
                "patch_size": 14
              },
              "image_token": "<image>",
              "patch_size": 14
            }
            """)
            let paliConfig: PaliGemmaProcessorConfiguration = try Self.decode("""
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "size": {"width": 2, "height": 2},
              "image_seq_length": 1
            }
            """)
            let image = CIImage(color: .red)
                .cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2))
            let context: [String: any Sendable] = ["display_label": "ignored"]

            let glm = try await GlmOcrProcessor(glmConfig, tokenizer: TestTokenizer())
                .prepare(input: UserInput(prompt: "Read this.", additionalContext: context))
            let idefics = try Idefics3Processor(ideficsConfig, tokenizer: TestTokenizer())
                .prepare(input: UserInput(prompt: "Describe.", additionalContext: context))
            let pixtral = try PixtralProcessor(pixtralConfig, tokenizer: TestTokenizer())
                .prepare(input: UserInput(prompt: "Describe.", additionalContext: context))
            let pali = try PaliGemmaProcessor(paliConfig, tokenizer: TestTokenizer())
                .prepare(input: UserInput(
                    prompt: "Describe.",
                    images: [.ciImage(image)],
                    additionalContext: context))

            #expect(glm.cacheScopeSalt == nil)
            #expect(idefics.cacheScopeSalt == nil)
            #expect(pixtral.cacheScopeSalt == nil)
            #expect(pali.cacheScopeSalt == nil)
        }
    }
}
