# Tool Calling

## Overview

mlx-swift-lm supports function calling / tool use with multiple model-specific formats. Models can generate structured tool calls that applications parse and execute, returning results back to the model.

**Files:**
- `Libraries/MLXLMCommon/Tool/Tool.swift`
- `Libraries/MLXLMCommon/Tool/ToolCall.swift`
- `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`
- `Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift`
- `Libraries/MLXLMCommon/Tool/ToolParameter.swift`

## Quick Reference

| Type | Purpose |
|------|---------|
| `Tool<Input, Output>` | Define a callable tool with typed `handler` |
| `ToolCall` | Parsed tool call from model output; has `execute(with:)` method |
| `ToolCallFormat` | Enum of supported formats |
| `ToolCallProcessor` | Streaming tool call detection |
| `ToolParameter` | Parameter definition for schema |
| `ToolSpec` | JSON schema dictionary type |

> **Note:** The `execute(with:)` method belongs to `ToolCall`, not `Tool`. You pass the `Tool` instance to `toolCall.execute(with: tool)` for type-safe execution.

## Supported Formats

| Format | Models | Example Output |
|--------|--------|----------------|
| `.json` | Llama, Qwen, most models | `<tool_call>{"name":"f","arguments":{...}}</tool_call>` |
| `.lfm2` | LFM2 / LFM2.5 | `<\|tool_call_start\|>[f(arg='v')]<\|tool_call_end\|>` (pythonic) |
| `.xmlFunction` | Nemotron-H, Qwen3 Coder, Qwen3.5, **Qwen 3.6** | `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>` |
| `.glm4` | GLM4 / GLM5 / DeepSeek | `func<arg_key>k</arg_key><arg_value>v</arg_value>` |
| `.gemma` | Gemma 3 | `<start_function_call>call:name{key:<escape>v<escape>}<end_function_call>` |
| `.gemma4` | **Gemma 4** (different envelope than Gemma 3) | `<\|tool_call>call:name{key:<\|"\|>v<\|"\|>}<tool_call\|>` |
| `.kimiK2` | Kimi K2 | `functions.name:0<\|tool_call_argument_begin\|>{...}` |
| `.minimaxM2` | **MiniMax M2 / M2.5** | `<minimax:tool_call><invoke name="f"><parameter name="k">v</parameter></invoke></minimax:tool_call>` |
| `.mistral` | Mistral V11+ / Mistral Small 4 | `[TOOL_CALLS]f[ARGS]{"k":"v"}` |
| `.llama3` | Llama 3.x (inline JSON) | `<\|python_tag\|>{"name":"f","parameters":{...}}` |

### Interleaved reasoning (Qwen 3.6, MiniMax M2)

Both emit `<think>...</think>` blocks **between** tool calls, not just before. Example from Qwen 3.6:

```
<think>need weather</think>
<tool_call><function=get_weather><parameter=location>Paris</parameter></function></tool_call>
<think>now check time</think>
<tool_call><function=get_time><parameter=zone>UTC</parameter></function></tool_call>
Final answer.
```

`BatchEngine.generate` / `Evaluate.generate` pipeline `ReasoningParser` → `ToolCallProcessor` so both tool calls surface as authoritative `.toolCall(ToolCall)` events and the reasoning never leaks to `.chunk`. See `reasoning-parser.md` for the streaming contract.

## Defining Tools

### Basic Tool Definition

```swift
struct WeatherInput: Codable {
    let location: String
    let unit: String?
}

struct WeatherOutput: Codable {
    let temperature: Double
    let condition: String
}

let weatherTool = Tool<WeatherInput, WeatherOutput>(
    name: "get_weather",
    description: "Get current weather for a location",
    parameters: [
        .required("location", type: .string, description: "City name"),
        .optional("unit", type: .string, description: "celsius or fahrenheit")
    ]
) { input in
    // Fetch weather...
    return WeatherOutput(temperature: 72, condition: "sunny")
}
```

### ToolParameter Types

```swift
// Required parameters
ToolParameter.required("count", type: .int, description: "Number of items")
ToolParameter.required("price", type: .double, description: "Price in USD")
ToolParameter.required("enabled", type: .bool, description: "Enable feature")
ToolParameter.required("name", type: .string, description: "User name")

// Optional parameters
ToolParameter.optional("tags", type: .array(elementType: .string), description: "List of tags")
ToolParameter.optional("config", type: .object(properties: [...]), description: "Config object")
```

### Custom Schema

```swift
let tool = Tool<Input, Output>(
    schema: [
        "type": "function",
        "function": [
            "name": "search",
            "description": "Search the database",
            "parameters": [
                "type": "object",
                "properties": [
                    "query": ["type": "string", "description": "Search query"]
                ],
                "required": ["query"]
            ]
        ]
    ],
    handler: { input in
        // ...
    }
)
```

## Passing Tools to Model

```swift
// Include tools in UserInput
let userInput = UserInput(
    prompt: .text("What's the weather in Paris?"),
    tools: [weatherTool.schema, searchTool.schema]
)

// Prepare and generate
let lmInput = try await modelContainer.prepare(input: userInput)
let stream = try await modelContainer.generate(input: lmInput, parameters: params)
```

## Processing Tool Calls

### From Generation Stream

```swift
for await generation in stream {
    switch generation {
    case .chunk(let text):
        print(text, terminator: "")

    case .toolCall(let toolCall):
        // Model wants to call a tool
        print("Tool call: \(toolCall.function.name)")
        print("Arguments: \(toolCall.function.arguments)")

        // Execute the tool
        let result = try await toolCall.execute(with: weatherTool)
        // Send result back to model...

    case .info(let info):
        print("\nDone: \(info.tokensPerSecond) tok/s")
    }
}
```

### Executing Tool Calls

```swift
// Type-safe execution
let toolCall: ToolCall = ...
let result = try await toolCall.execute(with: weatherTool)
// result is WeatherOutput

// Manual execution
let args = toolCall.function.arguments
let location = args["location"]  // JSONValue
```

## ToolCallProcessor

For streaming detection of tool calls:

```swift
let processor = ToolCallProcessor(format: .json)

// Process each chunk
for chunk in generatedChunks {
    if let text = processor.processChunk(chunk) {
        // Regular text output
        print(text, terminator: "")
    }
}

// After generation, check for tool calls
for toolCall in processor.toolCalls {
    print("Detected tool call: \(toolCall.function.name)")
}
```

### Processor with Tool Schemas

```swift
let processor = ToolCallProcessor(
    format: .lfm2,
    tools: [weatherTool.schema, searchTool.schema]  // For type-aware parsing
)
```

## Format Auto-Detection

Both `LLMModelFactory._load` and `VLMModelFactory._load` stamp `ModelConfiguration.toolCallFormat` at load time with this priority:

1. Caller-supplied override on `configuration.toolCallFormat`.
2. **JANG `capabilities.tool_parser` stamp** via `ToolCallFormat.fromCapabilityName(_:)` — authoritative when `jang_config.json` is present.
3. `ToolCallFormat.infer(from: modelType, configData:)` model-type heuristic.

```swift
// Model-type heuristic (secondary signal allows Llama 3 detection)
ToolCallFormat.infer(from: "lfm2")      // -> .lfm2
ToolCallFormat.infer(from: "glm4_moe")  // -> .glm4
ToolCallFormat.infer(from: "gemma3")    // -> .gemma
ToolCallFormat.infer(from: "gemma4")    // -> .gemma4
ToolCallFormat.infer(from: "llama",
    configData: configJSON)              // -> .llama3 if vocab_size≥128000

// JANG capability stamps — short family names from the JANG converter
ToolCallFormat.fromCapabilityName("qwen")        // -> .xmlFunction
ToolCallFormat.fromCapabilityName("qwen3_6")     // -> .xmlFunction
ToolCallFormat.fromCapabilityName("minimax")     // -> .minimaxM2
ToolCallFormat.fromCapabilityName("gemma4")      // -> .gemma4
ToolCallFormat.fromCapabilityName("glm47")       // -> .glm4
ToolCallFormat.fromCapabilityName("nemotron")    // -> .xmlFunction
ToolCallFormat.fromCapabilityName("kimi_k2")     // -> .kimiK2
ToolCallFormat.fromCapabilityName("mistral4")    // -> .mistral
```

### Explicit Format in Configuration

```swift
let config = ModelConfiguration(
    id: "mlx-community/GLM-4-9B-0414-4bit",
    toolCallFormat: .glm4,
    reasoningParserName: "glm4"   // JANG stamp for ReasoningParser
)
```

## ToolCall Structure

```swift
public struct ToolCall: Hashable, Codable, Sendable {
    public struct Function: Hashable, Codable, Sendable {
        public let name: String
        public let arguments: [String: JSONValue]
    }

    public let function: Function
}

// JSONValue handles various JSON types
public enum JSONValue: Hashable, Codable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])
}
```

## Multi-Turn with Tool Results

```swift
var messages: [Chat.Message] = [
    .user("What's the weather in Paris?")
]

// Use streamDetails to receive tool calls (respond/streamResponse drops them)
var responseText = ""
var detectedToolCall: ToolCall?

for try await generation in session.streamDetails(to: "What's the weather?", images: [], videos: []) {
    switch generation {
    case .chunk(let text):
        responseText += text
    case .toolCall(let toolCall):
        detectedToolCall = toolCall
    case .info:
        break
    }
}

// If tool call detected, execute and add result
if let toolCall = detectedToolCall {
    let result = try await toolCall.execute(with: weatherTool)

    // Add assistant's tool call and result
    messages.append(.assistant(responseText))
    messages.append(.tool("""
        {"temperature": \(result.temperature), "condition": "\(result.condition)"}
        """))

    // Continue conversation with updated history
    let session = ChatSession(modelContainer, history: messages)
    let finalResponse = try await session.respond(to: "")
}
```

## Parser Protocol

Custom formats can implement `ToolCallParser`:

```swift
public protocol ToolCallParser: Sendable {
    var startTag: String? { get }  // nil for inline formats
    var endTag: String? { get }
    func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall?
    // Default impl splits on startTag + calls parse; override when a single
    // buffer can contain multiple calls (Pythonic / LFM2).
    func parseEOS(_ toolCallBuffer: String,
                  tools: [[String: any Sendable]]?) -> [ToolCall]
}
```

### ToolCallProcessor contract

`ToolCallProcessor.swift` is byte-identical with ml-explore/mlx-swift-lm `main`. The streaming state machine is: `normal → potentialToolCall → collectingToolCall`. For inline formats (no wrapper tags), chunks are buffered until either (a) the parser returns a `ToolCall` or (b) JSON braces balance with no match, at which point the buffer flushes as user text. This is the invariant osaurus's `StreamAccumulator` depends on.

## Osaurus integration

Osaurus consumes this module at four sites (see `Libraries/MLXLMCommon/BatchEngine/OSAURUS-API-SURFACE.md` for the full per-symbol map):

- `StreamAccumulator.swift` — holds `ToolCallProcessor(format:, tools:)`, calls `processChunk` per token + `processEOS` on stream end.
- `BatchEngineAdapter.swift` — reads `context.configuration.toolCallFormat` + accepts a `toolCallFormatOverride`.
- `JANGReasoningResolver.swift` — uses `ParserResolution.toolCall(capabilities:modelType:)` + caches per model.
- `StreamingDeltaProcessor.swift` — holds a `ReasoningParser` instance for `<think>` stripping.

For the authoritative migration path (drop app-layer tool-call parsing, consume `.toolCall(ToolCall)` events from `BatchEngine.generate` directly), see `Libraries/MLXLMCommon/BatchEngine/OSAURUS-INTEGRATION.md` §4.

## Error Handling

```swift
public enum ToolError: Error {
    case nameMismatch(toolName: String, functionName: String)
}

do {
    let result = try await toolCall.execute(with: myTool)
} catch ToolError.nameMismatch(let expected, let got) {
    print("Tool '\(got)' doesn't match expected '\(expected)'")
}
```

## Best Practices

### DO

```swift
// DO: Check tool call name before executing
guard toolCall.function.name == "get_weather" else {
    // Handle unknown tool
}

// DO: Handle missing optional arguments with pattern matching
let unit: String
if case .string(let value) = toolCall.function.arguments["unit"] {
    unit = value
} else {
    unit = "celsius"  // default
}

// DO: Use specific format for models that need it
let config = ModelConfiguration(
    id: "glm-model",
    toolCallFormat: .glm4  // Required for GLM4
)
```

### DON'T

```swift
// DON'T: Assume format from model family
// Some models in a family may use different formats

// DON'T: Ignore tool call errors
// Always handle potential parsing/execution failures
```

## Deprecated Patterns

No major deprecations in tool calling - this is a newer feature. However, ensure you're using the streaming `Generation.toolCall` pattern rather than post-hoc parsing of raw output.
