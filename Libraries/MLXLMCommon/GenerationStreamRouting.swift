// Copyright © 2026 Osaurus AI. All rights reserved.

enum GenerationTextChannel {
    case content
    case reasoning
}

func routeGenerationText(
    _ text: String,
    channel: GenerationTextChannel,
    through toolCallProcessor: ToolCallProcessor
) -> [Generation] {
    var events: [Generation] = []
    if let visible = toolCallProcessor.processChunk(text) {
        switch channel {
        case .content:
            events.append(.chunk(visible))
        case .reasoning:
            events.append(.reasoning(visible))
        }
    }
    events.append(contentsOf: drainToolCallEvents(from: toolCallProcessor))
    return events
}

func drainToolCallEvents(from toolCallProcessor: ToolCallProcessor) -> [Generation] {
    guard !toolCallProcessor.toolCalls.isEmpty else { return [] }
    let calls = toolCallProcessor.toolCalls
    toolCallProcessor.toolCalls.removeAll(keepingCapacity: true)
    return calls.map { .toolCall($0) }
}

func flushGenerationText(
    channel: GenerationTextChannel,
    through toolCallProcessor: ToolCallProcessor
) -> [Generation] {
    var events: [Generation] = []
    if let visible = toolCallProcessor.processEOS() {
        switch channel {
        case .content:
            events.append(.chunk(visible))
        case .reasoning:
            events.append(.reasoning(visible))
        }
    }
    events.append(contentsOf: drainToolCallEvents(from: toolCallProcessor))
    return events
}
