import MLX
import MLXLMCommon
@testable import MLXVLM
import CoreImage
import Testing

@Suite("VLM default template context", .serialized)
struct VLMDefaultContextProcessorTests {
    @Test("reasoning-capable VLM stamps do not get family-based thinking overrides")
    func reasoningCapableVLMStampsDoNotGetFamilyOverrides() {
        let context = VLMDefaultContextUserInputProcessor.defaultContext(
            capabilities: JangCapabilities(
                reasoningParser: "qwen3",
                toolParser: "zaya_xml",
                thinkInTemplate: false,
                supportsTools: true,
                supportsThinking: true,
                family: "zaya1_vl",
                modality: "vision",
                cacheType: "hybrid"))

        #expect(context == nil)
    }

    private struct EchoContextProcessor: UserInputProcessor {
        func prepare(input: UserInput) async throws -> LMInput {
            LMInput(
                tokens: MLXArray([1]),
                cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
        }
    }

    @Test("non-thinking VLM capability defaults seed enable_thinking=false")
    func nonThinkingDefaultSeedsReasoningOffSalt() async throws {
        try await MLXMetalTestLock.withLock {
            let processor = VLMDefaultContextUserInputProcessor(
                base: EchoContextProcessor(),
                defaultAdditionalContext: ["enable_thinking": false])

            let input = try await processor.prepare(input: UserInput(prompt: "Describe."))

            #expect(input.cacheScopeSalt == "reasoning=off")
        }
    }

    @Test("explicit VLM request context overrides safe non-thinking default")
    func requestContextOverridesDefault() async throws {
        try await MLXMetalTestLock.withLock {
            let processor = VLMDefaultContextUserInputProcessor(
                base: EchoContextProcessor(),
                defaultAdditionalContext: ["enable_thinking": false])

            let input = try await processor.prepare(input: UserInput(
                prompt: "Describe.",
                additionalContext: ["enable_thinking": true]))

            #expect(input.cacheScopeSalt == "reasoning=on")
        }
    }

    @Test("nil VLM defaults do not invent cache salt")
    func nilDefaultsPreservePlainInput() async throws {
        try await MLXMetalTestLock.withLock {
            let processor = VLMDefaultContextUserInputProcessor(
                base: EchoContextProcessor(),
                defaultAdditionalContext: nil)

            let input = try await processor.prepare(input: UserInput(prompt: "Describe."))

            #expect(input.cacheScopeSalt == nil)
        }
    }

    private struct PreserveInputShapeProcessor: UserInputProcessor {
        func prepare(input: UserInput) async throws -> LMInput {
            #expect(input.images.count == 1)
            #expect(input.videos.isEmpty)
            #expect(input.audios.isEmpty)
            #expect(input.tools?.count == 1)
            #expect(input.processing.resize == CGSize(width: 32, height: 48))
            #expect(input.additionalContext?["custom_flag"] as? String == "kept")
            #expect(input.additionalContext?["enable_thinking"] as? Bool == false)
            return LMInput(
                tokens: MLXArray([1]),
                cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
        }
    }

    @Test("default-context rewrite preserves VLM media, tools, and processing")
    func rewritePreservesMediaToolsAndProcessing() async throws {
        try await MLXMetalTestLock.withLock {
            let processor = VLMDefaultContextUserInputProcessor(
                base: PreserveInputShapeProcessor(),
                defaultAdditionalContext: ["enable_thinking": false])
            let image = CIImage(
                color: .red
            ).cropped(to: CGRect(x: 0, y: 0, width: 56, height: 56))
            let tool: ToolSpec = [
                "type": "function",
                "function": [
                    "name": "describe_image",
                    "description": "Describe the image",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]

            let input = try await processor.prepare(input: UserInput(
                prompt: .text("Describe."),
                images: [.ciImage(image)],
                processing: .init(resize: CGSize(width: 32, height: 48)),
                tools: [tool],
                additionalContext: ["custom_flag": "kept"]))

            #expect(input.cacheScopeSalt == "reasoning=off")
        }
    }
}
