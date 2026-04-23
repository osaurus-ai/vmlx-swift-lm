import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

@main
struct Macros: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        DownloaderMacro.self,
        TokenizerAdaptorMacro.self,
        TokenizerLoaderMacro.self,
        LoadContainerMacro.self,
        LoadContextMacro.self,
    ]
}

public struct DownloaderMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        let argument = node.arguments.first?.expression.description ?? "HubClient()"

        return
            """
            // make sure you:
            //
            // import HuggingFace
            //
            { (hubApi: HubClient) -> MLXLMCommon.Downloader in
                struct HubBridge: MLXLMCommon.Downloader {
                    private let upstream: HubClient

                    init(_ upstream: HubClient) {
                        self.upstream = upstream
                    }

                    public func download(
                        id: String,
                        revision: String?,
                        matching patterns: [String],
                        useLatest: Bool,
                        progressHandler: @Sendable @escaping (Progress) -> Void
                    ) async throws -> URL {                        
                        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else {
                            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
                        }
                        let revision = revision ?? "main"

                        return try await upstream.downloadSnapshot(
                            of: repoID,
                            revision: revision,
                            matching: patterns,
                            progressHandler: { @MainActor progress in
                                progressHandler(progress)
                            }
                        )
                    }                    
                }

                return HubBridge(hubApi)
            }(\(raw: argument))
            """
    }
}

public struct TokenizerAdaptorMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let argument = node.arguments.first?.expression else {
            throw MacroExpansionError.message("#adaptHuggingFaceTokenizer requires an argument")
        }

        return
            """
            // make sure you:
            //
            // import Tokenizers
            //
            { (huggingFaceTokenizer: Tokenizers.Tokenizer) -> MLXLMCommon.Tokenizer in
                struct TokenizerBridge: MLXLMCommon.Tokenizer {
                    private let upstream: any Tokenizers.Tokenizer

                    init(_ upstream: any Tokenizers.Tokenizer) {
                        self.upstream = upstream
                    }

                    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
                        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
                    }

                    // swift-transformers uses `decode(tokens:)` instead of `decode(tokenIds:)`.
                    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
                        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
                    }

                    func convertTokenToId(_ token: String) -> Int? {
                        upstream.convertTokenToId(token)
                    }

                    func convertIdToToken(_ id: Int) -> String? {
                        upstream.convertIdToToken(id)
                    }

                    var bosToken: String? { upstream.bosToken }
                    var eosToken: String? { upstream.eosToken }
                    var unknownToken: String? { upstream.unknownToken }

                    func applyChatTemplate(
                        messages: [[String: any Sendable]],
                        tools: [[String: any Sendable]]?,
                        additionalContext: [String: any Sendable]?
                    ) throws -> [Int] {
                        // Iter 50 escape hatch: `VMLX_CHAT_TEMPLATE_OVERRIDE=/path/to/template.jinja`
                        // bypasses the tokenizer's shipped chat template. Motivation: Gemma-4's
                        // native template trips a swift-jinja 1.3.0 interaction bug — all
                        // constructs parse individually (see Gemma4ChatTemplateProbeTests)
                        // but the full assembly fails. The override lets callers ship
                        // `Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja` (or any
                        // other compatible template) for models blocked by upstream gaps.
                        // Default behaviour (no env var) is unchanged.
                        let env = ProcessInfo.processInfo.environment
                        if let path = env["VMLX_CHAT_TEMPLATE_OVERRIDE"], !path.isEmpty,
                           let src = try? String(contentsOfFile: path, encoding: .utf8) {
                            do {
                                return try upstream.applyChatTemplate(
                                    messages: messages,
                                    chatTemplate: Tokenizers.ChatTemplateArgument.literal(src))
                            } catch Tokenizers.TokenizerError.missingChatTemplate {
                                throw MLXLMCommon.TokenizerError.missingChatTemplate
                            }
                        }
                        do {
                            return try upstream.applyChatTemplate(
                                messages: messages, tools: tools, additionalContext: additionalContext)
                        } catch Tokenizers.TokenizerError.missingChatTemplate {
                            throw MLXLMCommon.TokenizerError.missingChatTemplate
                        } catch {
                            // Upstream threw on a template the swift-jinja runtime
                            // can't evaluate (Gemma-4 `multiplicativeBinaryOperator`
                            // parse, Nemotron `not in` on ArrayValue tuples, …).
                            // Try built-in fallbacks, picking the family that
                            // matches the tokenizer's special-token vocabulary so
                            // the emitted prompt shape stays model-native.
                            // `VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE=1` opts out.
                            if (env["VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE"] ?? "0") == "1" {
                                throw error
                            }
                            // Family sniff. Gemma-4 is the only widely-used
                            // family whose bos_token is literally "<bos>";
                            // ChatML-family models (Nemotron-Cascade-2 + all
                            // Mistral/Qwen 3.x descendants) use "<s>" or no
                            // bos. That single check lets us pick the right
                            // fallback ordering without needing the model
                            // config parsed separately. `convertTokenToId`
                            // for added-special tokens isn't universally
                            // reliable across swift-transformers builds, so
                            // we keep bosToken as the primary signal and
                            // treat the `<|turn>` probe as a confirmatory
                            // tiebreaker only when bos is ambiguous.
                            let isGemmaFamily: Bool = {
                                if upstream.bosToken == "<bos>" { return true }
                                if upstream.convertTokenToId("<|turn>") != nil { return true }
                                return false
                            }()
                            let ordered: [(label: String, template: String)]
                            if isGemmaFamily {
                                ordered = MLXLMCommon.ChatTemplateFallbacks.orderedFallbacks
                            } else {
                                ordered = [
                                    ("NemotronMinimal", MLXLMCommon.ChatTemplateFallbacks.nemotronMinimal),
                                    ("Gemma4WithTools", MLXLMCommon.ChatTemplateFallbacks.gemma4WithTools),
                                    ("Gemma4Minimal",   MLXLMCommon.ChatTemplateFallbacks.gemma4Minimal),
                                ]
                            }
                            for (label, template) in ordered {
                                do {
                                    let ids = try upstream.applyChatTemplate(
                                        messages: messages,
                                        chatTemplate: Tokenizers.ChatTemplateArgument.literal(template))
                                    if (env["VMLX_CHAT_TEMPLATE_FALLBACK_LOG"] ?? "0") == "1" {
                                        FileHandle.standardError.write(
                                            "[vmlx] chat-template fallback engaged: \\(label) (original error: \\(error))\\n"
                                                .data(using: .utf8)!)
                                    }
                                    return ids
                                } catch {
                                    continue
                                }
                            }
                            // No fallback worked — surface the original upstream error.
                            throw error
                        }
                    }
                }

                return TokenizerBridge(huggingFaceTokenizer)
            }(\(argument))
            """
    }
}

public struct TokenizerLoaderMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        return
            """
            { () -> MLXLMCommon.TokenizerLoader in
                struct TransformersLoader: MLXLMCommon.TokenizerLoader {
                    public init() {}

                    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
                        // make sure you:
                        //
                        // import Tokenizers
                        //
                        let upstream = try await AutoTokenizer.from(modelFolder: directory)
                        return #adaptHuggingFaceTokenizer(upstream)
                    }
                }

                return TransformersLoader()
            }()
            """
    }
}

public struct LoadContainerMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let configuration = node.arguments.first?.expression else {
            throw MacroExpansionError.message(
                "#huggingFaceLoadModelContainer requires a configuration")
        }

        let progress =
            if let expr = node.arguments.first(where: { $0.label?.text == "progressHandler" })?
                .expression
            {
                expr.description
            } else {
                "{ _ in }"
            }

        return
            """
            loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: \(configuration),
                progressHandler: \(raw: progress))
            """
    }
}

public struct LoadContextMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let configuration = node.arguments.first?.expression else {
            throw MacroExpansionError.message("#huggingFaceLoadModel requires a configuration")
        }

        let progress =
            if let expr = node.arguments.first(where: { $0.label?.text == "progressHandler" })?
                .expression
            {
                expr.description
            } else {
                "{ _ in }"
            }

        return
            """
            loadModel(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: \(configuration),
                progressHandler: \(raw: progress))
            """
    }
}

enum MacroExpansionError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let text): return text
        }
    }
}
