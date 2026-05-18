//
//  OpenAIAPI.swift
//  osaurus
//
//  Created by Terence on 8/17/25.
//

import Foundation

// MARK: - OpenAI API Compatible Structures

/// OpenAI-compatible model object
public struct OpenAIModel: Codable, Sendable {
    public let id: String
    public var object: String = "model"
    public var created: Int = 0
    public var owned_by: String = "osaurus"
    public var permission: [ModelPermission]? = nil
    public var root: String? = nil
    public var parent: String? = nil
    public var name: String? = nil
    public var model: String? = nil
    public var modified_at: String? = nil
    public var size: Int? = nil
    public var digest: String? = nil
    public var details: ModelDetails? = nil

    /// Initialize from a model name (for local models)
    public init(modelName: String) {
        self.id = modelName
        self.object = "model"
        self.created = Int(Date().timeIntervalSince1970)
        self.owned_by = "osaurus"
        self.root = modelName
    }

    /// Full initializer
    public init(
        id: String,
        object: String = "model",
        created: Int = 0,
        owned_by: String = "osaurus",
        permission: [ModelPermission]? = nil,
        root: String? = nil,
        parent: String? = nil,
        name: String? = nil,
        model: String? = nil,
        modified_at: String? = nil,
        size: Int? = nil,
        digest: String? = nil,
        details: ModelDetails? = nil
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.owned_by = owned_by
        self.permission = permission
        self.root = root
        self.parent = parent
        self.name = name
        self.model = model
        self.modified_at = modified_at
        self.size = size
        self.digest = digest
        self.details = details
    }

    // Explicit Codable implementation to avoid ambiguity
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        object = try container.decodeIfPresent(String.self, forKey: .object) ?? "model"
        created = try container.decodeIfPresent(Int.self, forKey: .created) ?? 0
        owned_by = try container.decodeIfPresent(String.self, forKey: .owned_by) ?? "unknown"
        permission = try container.decodeIfPresent([ModelPermission].self, forKey: .permission)
        root = try container.decodeIfPresent(String.self, forKey: .root)
        parent = try container.decodeIfPresent(String.self, forKey: .parent)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        modified_at = try container.decodeIfPresent(String.self, forKey: .modified_at)
        // Some OpenAI-compatible servers expose provider-local metadata in
        // `size` as a fractional value. That field is informational for
        // Osaurus model discovery, so preserve the model row instead of
        // rejecting the whole `/models` response.
        size = try? container.decodeIfPresent(Int.self, forKey: .size)
        digest = try container.decodeIfPresent(String.self, forKey: .digest)
        details = try container.decodeIfPresent(ModelDetails.self, forKey: .details)
    }

    private enum CodingKeys: String, CodingKey {
        case id, object, created, owned_by, permission, root, parent
        case name, model, modified_at, size, digest, details
    }
}

/// Model permission object (OpenAI format)
public struct ModelPermission: Codable, Sendable {
    public var id: String?
    public var object: String?
    public var created: Int?
    public var allow_create_engine: Bool?
    public var allow_sampling: Bool?
    public var allow_logprobs: Bool?
    public var allow_search_indices: Bool?
    public var allow_view: Bool?
    public var allow_fine_tuning: Bool?
    public var organization: String?
    public var group: String?
    public var is_blocking: Bool?
}

public struct ModelDetails: Codable, Sendable {
    public let parent_model: String?
    public let format: String?
    public let family: String?
    public let families: [String]?
    public let parameter_size: String?
    public let quantization_level: String?
}

/// Response for /models endpoint
public struct ModelsResponse: Codable, Sendable {
    public var object: String = "list"
    public let data: [OpenAIModel]

    private enum CodingKeys: String, CodingKey {
        case object, data
    }

    /// Memberwise initializer
    public init(object: String = "list", data: [OpenAIModel]) {
        self.object = object
        self.data = data
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // Make object optional for providers like OpenRouter that don't include it
        self.object = try container.decodeIfPresent(String.self, forKey: .object) ?? "list"
        self.data = try container.decode([OpenAIModel].self, forKey: .data)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(object, forKey: .object)
        try container.encode(data, forKey: .data)
    }
}

public struct LocalAudioSamples: Sendable, Equatable {
    public let samples: [Float]
    public let sampleRate: Int
    public let preencodedAttachmentId: UUID?

    public init(
        samples: [Float],
        sampleRate: Int,
        preencodedAttachmentId: UUID? = nil
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.preencodedAttachmentId = preencodedAttachmentId
    }
}

// MARK: - Multimodal Content Parts

/// OpenAI-compatible content part for multimodal messages.
///
/// Supports four shapes:
///   - `text` / `input_text` — plain text
///   - `image_url` — `{url, detail?}`. URL may be `data:image/...;base64,...` or `https://...`
///   - `input_audio` — `{data: <base64>, format: "wav"|"mp3"|"flac"|...}`. Mirrors the
///     OpenAI Realtime / GPT-4o audio shape; valid WAV bytes decode directly to
///     `UserInput.Audio.samples(...)` for local MLX, while other containers fall
///     back to a temp file handed to vmlx as `UserInput.Audio.url(...)` so
///     `nemotronOmniLoadAudioFile` can use AVAudioConverter.
///   - `video_url` — `{url}`. Mirrors the convention adopted by LM Studio / Ollama
///     for video inputs since OpenAI hasn't published a canonical chat-completions
///     video shape. URL may be `data:video/...;base64,...` or `https://...`.
public enum MessageContentPart: Codable, Sendable {
    case text(String)
    case imageUrl(url: String, detail: String?)
    case audioInput(data: String, format: String)
    case videoUrl(url: String)

    private enum CodingKeys: String, CodingKey {
        case type
        case text
        case input_text
        case image_url
        case input_audio
        case video_url
    }

    private struct ImageUrlContent: Codable {
        public let url: String
        public let detail: String?
    }

    private struct InputAudioContent: Codable {
        public let data: String
        public let format: String
    }

    private struct VideoUrlContent: Codable {
        public let url: String
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "text":
            if let text = try? container.decode(String.self, forKey: .text) {
                self = .text(text)
            } else if let inputText = try? container.decode(String.self, forKey: .input_text) {
                self = .text(inputText)
            } else {
                self = .text("")
            }
        case "image_url":
            let imageUrl = try container.decode(ImageUrlContent.self, forKey: .image_url)
            self = .imageUrl(url: imageUrl.url, detail: imageUrl.detail)
        case "input_audio":
            let audio = try container.decode(InputAudioContent.self, forKey: .input_audio)
            self = .audioInput(data: audio.data, format: audio.format)
        case "video_url":
            let video = try container.decode(VideoUrlContent.self, forKey: .video_url)
            self = .videoUrl(url: video.url)
        default:
            // Fallback to text for unknown types
            if let text = try? container.decode(String.self, forKey: .text) {
                self = .text(text)
            } else {
                self = .text("")
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .imageUrl(let url, let detail):
            try container.encode("image_url", forKey: .type)
            try container.encode(ImageUrlContent(url: url, detail: detail), forKey: .image_url)
        case .audioInput(let data, let format):
            try container.encode("input_audio", forKey: .type)
            try container.encode(InputAudioContent(data: data, format: format), forKey: .input_audio)
        case .videoUrl(let url):
            try container.encode("video_url", forKey: .type)
            try container.encode(VideoUrlContent(url: url), forKey: .video_url)
        }
    }
}

/// Chat message in OpenAI format
public struct ChatMessage: Codable, Sendable {
    public let role: String
    public let content: String?
    /// Multimodal content parts (images, text) - populated when content is an array
    public let contentParts: [MessageContentPart]?
    /// In-process live voice samples aligned to audio input parts. This is
    /// deliberately not Codable: OpenAI-compatible JSON keeps the portable
    /// `input_audio` payload, while local MLX requests can bypass the
    /// WAV/base64/temp-file round trip.
    public let localAudioSamples: [LocalAudioSamples?]
    /// Present when assistant requests tool invocations
    public let tool_calls: [ToolCall]?
    /// Required for role=="tool" messages to associate with a prior tool call
    public let tool_call_id: String?
    /// Reasoning/thinking text from thinking-capable OpenAI-compat providers
    /// (DeepSeek thinking mode, Qwen, vLLM, …). Echoed back on follow-ups
    /// for providers that require it (issue #959); `RemoteProviderService`
    /// strips it on the wire for everyone else.
    public let reasoning_content: String?

    /// Extract image URLs from content parts (supports both data URLs and http URLs)
    public var imageUrls: [String] {
        guard let parts = contentParts else { return [] }
        return parts.compactMap { part in
            if case .imageUrl(let url, _) = part {
                return url
            }
            return nil
        }
    }

    /// Extract base64 image data from data URLs in content parts
    public var imageDataFromParts: [Data] {
        imageUrls.compactMap { url in
            // Parse data URL: data:image/png;base64,<base64data>
            guard url.hasPrefix("data:image/") else { return nil }
            guard let commaIndex = url.firstIndex(of: ",") else { return nil }
            let base64String = String(url[url.index(after: commaIndex)...])
            return Data(base64Encoded: base64String)
        }
    }

    /// Extract `(base64, format)` pairs from `input_audio` content parts.
    /// `format` is whatever the client sent (e.g. `"wav"`, `"mp3"`); valid
    /// WAV data can bypass temp-file materialization, and fallback containers
    /// pass the format through to the temp-file extension for AVAudioConverter.
    public var audioInputs: [(data: String, format: String)] {
        audioInputsWithLocalSamples.map { (data: $0.data, format: $0.format) }
    }

    public var audioInputsWithLocalSamples: [(data: String, format: String, localSamples: LocalAudioSamples?)] {
        guard let parts = contentParts else { return [] }
        var audioIndex = 0
        return parts.compactMap { part in
            if case .audioInput(let data, let format) = part {
                let local = audioIndex < localAudioSamples.count ? localAudioSamples[audioIndex] : nil
                audioIndex += 1
                return (data, format, local)
            }
            return nil
        }
    }

    /// Extract video URLs (data: or http(s):) from `video_url` content parts.
    public var videoUrls: [String] {
        guard let parts = contentParts else { return [] }
        return parts.compactMap { part in
            if case .videoUrl(let url) = part {
                return url
            }
            return nil
        }
    }
}

// Allow decoding OpenAI-style array-of-parts content while preserving string encoding
extension ChatMessage {
    private enum CodingKeys: String, CodingKey {
        case role
        case content
        case tool_calls
        case tool_call_id
        case reasoning_content
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.role = try container.decode(String.self, forKey: .role)
        self.tool_calls = try? container.decode([ToolCall].self, forKey: .tool_calls)
        self.tool_call_id = try? container.decode(String.self, forKey: .tool_call_id)
        self.reasoning_content = try? container.decode(String.self, forKey: .reasoning_content)
        self.localAudioSamples = []

        if let stringContent = try? container.decode(String.self, forKey: .content) {
            self.content = stringContent
            self.contentParts = nil
        } else if let parts = try? container.decode([MessageContentPart].self, forKey: .content) {
            // Store the parts for multimodal access
            self.contentParts = parts
            // Also extract text for backward compatibility
            let texts = parts.compactMap { part -> String? in
                if case .text(let text) = part { return text }
                return nil
            }
            // OpenAI-style array-of-parts text should be concatenated verbatim. Newlines should be
            // represented explicitly in the text segments themselves, not inserted by the decoder.
            self.content = texts.isEmpty ? nil : texts.joined()
        } else {
            self.content = nil
            self.contentParts = nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        // If we have content parts with any non-text media, encode as array;
        // otherwise as string. Round-trip preserves audio/video/image parts
        // so a request that came in with `input_audio` or `video_url` is
        // re-serialized in the same shape.
        if let parts = contentParts,
            parts.contains(where: {
                switch $0 {
                case .imageUrl, .audioInput, .videoUrl: return true
                case .text: return false
                }
            })
        {
            try container.encode(parts, forKey: .content)
        } else if let content = content {
            // Only encode content if it's not nil (OpenAI rejects null content)
            try container.encode(content, forKey: .content)
        }
        // Note: content is intentionally omitted when nil (e.g., assistant messages with tool_calls)
        try container.encodeIfPresent(tool_calls, forKey: .tool_calls)
        try container.encodeIfPresent(tool_call_id, forKey: .tool_call_id)
        // Stripped at the transport layer for providers that don't need it.
        try container.encodeIfPresent(reasoning_content, forKey: .reasoning_content)
    }
}

extension ChatMessage {
    public init(role: String, content: String) {
        self.role = role
        self.content = content
        self.contentParts = nil
        self.localAudioSamples = []
        self.tool_calls = nil
        self.tool_call_id = nil
        self.reasoning_content = nil
    }

    /// Initialize with optional tool calls, tool call id, and reasoning content.
    /// `reasoning_content` is echoed back to thinking-capable providers
    /// (e.g. DeepSeek) on multi-turn follow-ups.
    public init(
        role: String,
        content: String?,
        tool_calls: [ToolCall]?,
        tool_call_id: String?,
        reasoning_content: String? = nil
    ) {
        self.role = role
        self.content = content
        self.contentParts = nil
        self.localAudioSamples = []
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.reasoning_content = reasoning_content
    }

    /// Initialize with multimodal content (text and images)
    public init(role: String, text: String, imageData: [Data]) {
        self.role = role
        var parts: [MessageContentPart] = []

        // Add text part
        if !text.isEmpty {
            parts.append(.text(text))
        }

        // Add image parts as base64 data URLs
        for data in imageData {
            let base64 = data.base64EncodedString()
            let dataUrl = "data:image/png;base64,\(base64)"
            parts.append(.imageUrl(url: dataUrl, detail: nil))
        }

        self.contentParts = parts.isEmpty ? nil : parts
        self.content = text.isEmpty ? nil : text
        self.localAudioSamples = []
        self.tool_calls = nil
        self.tool_call_id = nil
        self.reasoning_content = nil
    }

    /// Multimodal init covering image + audio + video. Used by the
    /// chat composer when the loaded model's capabilities advertise the
    /// modality. Audio bytes encode as `input_audio` with explicit
    /// format hint; video bytes encode as `video_url` with
    /// `data:video/<container>` URL. All three flow into the
    /// OpenAI-compatible JSON shape that `mapOpenAIChatToMLX` lowers through
    /// `extractAudioSources` / `extractVideoSources`.
    public init(
        role: String,
        text: String,
        imageData: [Data],
        audios: [(data: Data, format: String)],
        localAudioSamples: [LocalAudioSamples?] = [],
        videos: [(data: Data, mimeSubtype: String)]
    ) {
        self.role = role
        var parts: [MessageContentPart] = []

        if !text.isEmpty {
            parts.append(.text(text))
        }

        for data in imageData {
            let base64 = data.base64EncodedString()
            parts.append(.imageUrl(url: "data:image/png;base64,\(base64)", detail: nil))
        }

        for (data, format) in audios {
            // OpenAI audio shape: bare base64 string + format hint.
            // The format string round-trips to vmlx's
            // `materializeMediaDataUrl` audio canonicalization (mp4 → m4a
            // for audio mime, NOT for video — audit fix locked in
            // `MaterializeMediaDataUrlMCDCTests`).
            parts.append(.audioInput(data: data.base64EncodedString(), format: format))
        }

        for (data, mimeSubtype) in videos {
            // Video data URL with the container subtype (`mp4` / `mov` /
            // `webm` / `quicktime`) so the materializer keeps the right
            // file extension (NOT downgraded to .m4a — see audit fix).
            let base64 = data.base64EncodedString()
            parts.append(
                .videoUrl(url: "data:video/\(mimeSubtype);base64,\(base64)")
            )
        }

        self.contentParts = parts.isEmpty ? nil : parts
        self.content = text.isEmpty ? nil : text
        self.localAudioSamples = localAudioSamples
        self.tool_calls = nil
        self.tool_call_id = nil
        self.reasoning_content = nil
    }
}

/// Chat completion request
public struct ChatCompletionRequest: Codable, Sendable {
    public let model: String
    public var messages: [ChatMessage]
    public let temperature: Float?
    public let max_tokens: Int?
    /// OpenAI newer alias for max_tokens; accepted on inbound requests alongside max_tokens.
    public var max_completion_tokens: Int? = nil
    public let stream: Bool?
    public let top_p: Float?
    public let frequency_penalty: Float?
    public let presence_penalty: Float?
    public let stop: [String]?
    public let n: Int?
    /// OpenAI tools/function-calling definitions
    public let tools: [Tool]?
    /// OpenAI tool_choice ("none" | "auto" | {"type":"function","function":{"name":...}})
    public let tool_choice: ToolChoiceOption?
    /// Optional session identifier for chat/history grouping. Not a KV cache key —
    /// vmlx-swift-lm's `CacheCoordinator` is content-addressed and discovers
    /// reusable prefixes autonomously.
    public var session_id: String? = nil
    /// Deterministic-sampling seed (OpenAI v1.x). When set, identical
    /// requests should yield identical completions on the same backend.
    public var seed: Int? = nil
    /// `{"type":"json_object"}` for OpenAI JSON mode. Other shapes
    /// (`text`, `json_schema`) are rejected at request validation.
    public var response_format: ResponseFormat? = nil
    /// `{"include_usage": true}` instructs the SSE producer to emit a
    /// final chunk carrying `usage` (prompt/completion/total tokens).
    public var stream_options: StreamOptions? = nil
    /// Model-specific options from the active ModelProfile (not serialized to JSON).
    public var modelOptions: [String: ModelOptionValue]? = nil
    /// Optional TTFT trace for diagnostic timing (not serialized to JSON).
    public var ttftTrace: TTFTTrace? = nil
    /// Per-request thinking toggle. Translated to `modelOptions["disableThinking"]`
    /// at request entry; absent preserves server defaults.
    public var enable_thinking: Bool? = nil
    /// OpenAI-compatible reasoning effort. Local Hy3 uses this as the native
    /// `reasoning_effort` chat-template kwarg; remote providers forward it
    /// natively where supported.
    public var reasoning_effort: String? = nil

    /// Resolved max tokens, preferring max_tokens then max_completion_tokens.
    public var resolvedMaxTokens: Int? { max_tokens ?? max_completion_tokens }

    private enum CodingKeys: String, CodingKey {
        case model, messages, temperature, max_tokens, max_completion_tokens, stream, top_p
        case frequency_penalty, presence_penalty, stop, n
        case tools, tool_choice, session_id
        case seed, response_format, stream_options
        case enable_thinking, reasoning_effort
    }

    public init(
        model: String,
        messages: [ChatMessage],
        temperature: Float? = nil,
        max_tokens: Int? = nil,
        max_completion_tokens: Int? = nil,
        stream: Bool? = nil,
        top_p: Float? = nil,
        frequency_penalty: Float? = nil,
        presence_penalty: Float? = nil,
        stop: [String]? = nil,
        n: Int? = nil,
        tools: [Tool]? = nil,
        tool_choice: ToolChoiceOption? = nil,
        session_id: String? = nil,
        seed: Int? = nil,
        response_format: ResponseFormat? = nil,
        stream_options: StreamOptions? = nil
    ) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.stream = stream
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.n = n
        self.tools = tools
        self.tool_choice = tool_choice
        self.session_id = session_id
        self.seed = seed
        self.response_format = response_format
        self.stream_options = stream_options
    }

    public func withModel(_ newModel: String) -> ChatCompletionRequest {
        var copy = ChatCompletionRequest(
            model: newModel,
            messages: messages,
            temperature: temperature,
            max_tokens: max_tokens,
            stream: stream,
            top_p: top_p,
            frequency_penalty: frequency_penalty,
            presence_penalty: presence_penalty,
            stop: stop,
            n: n,
            tools: tools,
            tool_choice: tool_choice,
            session_id: session_id,
            seed: seed,
            response_format: response_format,
            stream_options: stream_options
        )
        copy.modelOptions = modelOptions
        copy.ttftTrace = ttftTrace
        copy.enable_thinking = enable_thinking
        copy.reasoning_effort = reasoning_effort
        return copy
    }
}

/// OpenAI `response_format`. We only act on `json_object`; other kinds
/// (`text`, `json_schema`) flow through unchanged so the request
/// validator can accept or reject them with a clear, specific error.
public struct ResponseFormat: Codable, Sendable, Equatable {
    public let type: String
}

/// OpenAI `stream_options` shape. Today we only honor `include_usage`.
public struct StreamOptions: Codable, Sendable, Equatable {
    public let include_usage: Bool?
}

/// Chat completion choice
public struct ChatChoice: Codable, Sendable {
    public let index: Int
    public let message: ChatMessage
    public let finish_reason: String

    public init(index: Int, message: ChatMessage, finish_reason: String) {
        self.index = index
        self.message = message
        self.finish_reason = finish_reason
    }
}

/// Token usage information
public struct Usage: Codable, Sendable {
    public let prompt_tokens: Int
    public let completion_tokens: Int
    public let total_tokens: Int

    public init(prompt_tokens: Int, completion_tokens: Int, total_tokens: Int) {
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
    }
}

/// Chat completion response
public struct ChatCompletionResponse: Codable, Sendable {
    public let id: String
    public var object: String = "chat.completion"
    public let created: Int
    public let model: String
    public let choices: [ChatChoice]
    public let usage: Usage
    public var system_fingerprint: String? = nil
    /// Content hash of the system prompt + canonical tool schemas used for this request.
    /// Informational only — clients can use it to detect when the system
    /// prefix changed across requests. KV reuse itself is handled
    /// autonomously by vmlx's `CacheCoordinator` (content-addressed).
    public var prefix_hash: String? = nil

    public init(
        id: String,
        object: String = "chat.completion",
        created: Int,
        model: String,
        choices: [ChatChoice],
        usage: Usage,
        system_fingerprint: String? = nil,
        prefix_hash: String? = nil
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.system_fingerprint = system_fingerprint
        self.prefix_hash = prefix_hash
    }
}

// MARK: - Streaming Response Structures

/// Delta content for streaming
public struct DeltaContent: Codable, Sendable {
    public let role: String?
    public let content: String?
    public let refusal: String?
    /// Incremental tool_calls information (OpenAI-compatible)
    public let tool_calls: [DeltaToolCall]?
    /// Reasoning/thinking text streamed in a separate channel by OpenAI-compatible
    /// providers (DeepSeek, Qwen, Together, vLLM). Absent on providers that only
    /// emit content. The stream parser wraps these chunks with synthetic `<think>`
    /// tags so the rest of the pipeline can route them as reasoning.
    public let reasoning_content: String?

    public init(
        role: String? = nil,
        content: String? = nil,
        refusal: String? = nil,
        tool_calls: [DeltaToolCall]? = nil,
        reasoning_content: String? = nil
    ) {
        self.role = role
        self.content = content
        self.refusal = refusal
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
    }
}

/// Streaming choice
public struct StreamChoice: Codable, Sendable {
    public let index: Int
    public let delta: DeltaContent
    public let finish_reason: String?
}

/// Chat completion chunk for streaming
public struct ChatCompletionChunk: Codable, Sendable {
    public let id: String
    public var object: String = "chat.completion.chunk"
    public let created: Int
    public let model: String
    public let choices: [StreamChoice]
    public var system_fingerprint: String? = nil
    /// Included only in the first chunk; see `ChatCompletionResponse.prefix_hash`.
    public var prefix_hash: String? = nil
    /// Final usage chunk (OpenAI `stream_options.include_usage`). Populated
    /// only on the dedicated penultimate SSE chunk; nil on every other.
    public var usage: Usage? = nil
}

// MARK: - Error Response

/// OpenAI-compatible error response
public struct OpenAIError: Codable, Error, Sendable {
    public let error: ErrorDetail

    public struct ErrorDetail: Codable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String?
    }
}

// MARK: - Helper Extensions

extension ChatCompletionRequest {
    /// Convert OpenAI format messages to internal Message format
    public func toInternalMessages() -> [Message] {
        return messages.map { chatMessage in
            let role: MessageRole =
                switch chatMessage.role {
                case "system": .system
                case "user": .user
                case "assistant": .assistant
                default: .user
                }
            return Message(role: role, content: chatMessage.content ?? "")
        }
    }
}

extension OpenAIModel {
    /// Create an OpenAI model from an internal model name
    public init(from modelName: String) {
        self.id = modelName
        self.created = Int(Date().timeIntervalSince1970)
        self.root = modelName
    }
}

// MARK: - Tools: Request/Response Models

/// Tool definition (currently only type=="function")
public struct Tool: Codable, Sendable {
    public let type: String  // "function"
    public let function: ToolFunction

    public init(type: String, function: ToolFunction) {
        self.type = type
        self.function = function
    }
}

public struct ToolFunction: Codable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue?

    public init(name: String, description: String?, parameters: JSONValue?) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encodeIfPresent(description, forKey: .description)
        let params = parameters ?? .object(["type": .string("object"), "properties": .object([:])])
        try container.encode(params, forKey: .parameters)
    }

    private enum CodingKeys: String, CodingKey {
        case name, description, parameters
    }
}

/// tool_choice option
public enum ToolChoiceOption: Codable, Sendable {
    case auto
    case none
    case function(FunctionName)

    public struct FunctionName: Codable, Sendable {
        public let type: String
        public let function: Name
    }
    public struct Name: Codable, Sendable { public let name: String }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            switch str {
            case "auto": self = .auto
            case "none": self = .none
            default:
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription:
                        "Unsupported tool_choice string '\(str)'. Expected 'auto', 'none', or a typed function selector."
                )
            }
            return
        }
        let obj = try container.decode(FunctionName.self)
        self = .function(obj)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .auto:
            try container.encode("auto")
        case .none:
            try container.encode("none")
        case .function(let obj):
            try container.encode(obj)
        }
    }
}

/// Assistant tool call in responses
public struct ToolCall: Codable, Sendable {
    public let id: String
    public let type: String  // "function"
    public let function: ToolCallFunction
    /// Optional thought signature for Gemini thinking-mode models (e.g. Gemini 2.5)
    public let geminiThoughtSignature: String?

    public init(id: String, type: String, function: ToolCallFunction, geminiThoughtSignature: String? = nil) {
        self.id = id
        self.type = type
        self.function = function
        self.geminiThoughtSignature = geminiThoughtSignature
    }
}

public struct ToolCallFunction: Codable, Sendable {
    public let name: String
    /// Arguments serialized as JSON string per OpenAI spec
    public let arguments: String

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

// Streaming deltas for tool calls
public struct DeltaToolCall: Codable, Sendable {
    public let index: Int?
    public let id: String?
    public let type: String?
    public let function: DeltaToolCallFunction?
}

public struct DeltaToolCallFunction: Codable, Sendable {
    public let name: String?
    public let arguments: String?
}

// MARK: - Generic JSON value for tool parameters

/// Simple JSON value representation to carry arbitrary JSON schema/arguments
public enum JSONValue: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let n = try? container.decode(Double.self) {
            self = .number(n)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let arr = try? container.decode([JSONValue].self) {
            self = .array(arr)
        } else if let dict = try? container.decode([String: JSONValue].self) {
            self = .object(dict)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let b):
            try container.encode(b)
        case .number(let n):
            try container.encode(n)
        case .string(let s):
            try container.encode(s)
        case .array(let arr):
            try container.encode(arr)
        case .object(let obj):
            try container.encode(obj)
        }
    }
}

// MARK: - JSONValue Conversions

extension JSONValue {
    /// Convert JSONValue to Sendable-compatible value for Jinja chat templates.
    /// Null values are dropped from dictionaries because Jinja's `Value(any:)` cannot
    /// handle `NSNull` and throws a runtime error. JSON Schema treats a missing key
    /// the same as `null`, so this is semantically lossless for tool specs.
    public var sendableValue: any Sendable {
        switch self {
        case .null:
            return NSNull()
        case .bool(let b):
            return b
        case .number(let n):
            return n
        case .string(let s):
            return s
        case .array(let arr):
            return arr.map { $0.sendableValue }
        case .object(let obj):
            var dict: [String: any Sendable] = [:]
            for (k, v) in obj {
                if case .null = v { continue }
                dict[k] = v.sendableValue
            }
            return dict
        }
    }

    /// Convert JSONValue to Foundation JSON-compatible Any (for JSONSerialization).
    /// Unlike `sendableValue`, this preserves null as `NSNull` in dictionaries
    /// since `JSONSerialization` handles it correctly.
    public var anyValue: Any {
        switch self {
        case .null:
            return NSNull()
        case .bool(let b):
            return b
        case .number(let n):
            return n
        case .string(let s):
            return s
        case .array(let arr):
            return arr.map { $0.anyValue }
        case .object(let obj):
            var dict: [String: Any] = [:]
            for (k, v) in obj { dict[k] = v.anyValue }
            return dict
        }
    }
}

extension ToolFunction {
    /// Convert to MLXLMCommon.ToolSpec-compatible function dictionary
    fileprivate func toFunctionSpec() -> [String: any Sendable] {
        var fn: [String: any Sendable] = [
            "name": name
        ]
        if let description {
            fn["description"] = description
        }
        if let parameters {
            fn["parameters"] = parameters.sendableValue
        }
        return fn
    }
}

extension Tool {
    /// Convert to Tokenizers.ToolSpec (`[String: any Sendable]`) for MLX chat templates.
    ///
    /// The dictionary is round-tripped through canonical JSON
    /// (`JSONSerialization.WritingOptions.sortedKeys`) so the structure handed
    /// to the chat template — and the resulting `<tools>` block in the
    /// rendered prompt — is byte-stable across calls. Without this, key
    /// iteration order from a fresh dictionary literal can shift between
    /// requests and silently invalidate the MLX paged KV cache prefix.
    public func toTokenizerToolSpec() -> [String: any Sendable] {
        let raw: [String: any Sendable] = [
            "type": type,
            "function": function.toFunctionSpec(),
        ]
        return Self.canonicalize(raw) ?? raw
    }

    /// Canonical JSON bytes for hash/evidence paths that need to distinguish
    /// compact bootstrap schemas from full tool schemas. This mirrors the
    /// tokenizer-tool shape so prefix evidence tracks the bytes handed to the
    /// chat template, not just the callable names.
    public func canonicalHashPayload() -> Data {
        let spec = toTokenizerToolSpec()
        if JSONSerialization.isValidJSONObject(spec),
            let data = try? JSONSerialization.data(withJSONObject: spec, options: [.sortedKeys])
        {
            return data
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return (try? encoder.encode(self)) ?? Data("\(type)\0\(function.name)".utf8)
    }

    /// Round-trip a Sendable JSON value through `JSONSerialization` with
    /// `.sortedKeys` to canonicalise nested key ordering. Returns `nil` on
    /// the (extremely unlikely) serialisation failure so callers fall back
    /// to the raw dict rather than emit nothing.
    fileprivate static func canonicalize(_ value: [String: any Sendable]) -> [String: any Sendable]? {
        guard JSONSerialization.isValidJSONObject(value),
            let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
            let reparsed = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: any Sendable]
        else { return nil }
        return reparsed
    }
}
