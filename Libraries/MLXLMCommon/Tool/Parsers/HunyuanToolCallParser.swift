// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation

/// Parser for Tencent Hunyuan / Hy3 tool-call blocks.
///
/// Wire format:
///
/// ```text
/// <tool_calls>
/// <tool_call>function_name<tool_sep>
/// <arg_key>key</arg_key><arg_value>value</arg_value>
/// </tool_call>
/// </tool_calls>
/// ```
///
/// A single `<tool_calls>` wrapper may contain multiple `<tool_call>` entries,
/// so ``parseEOS(_:tools:)`` returns every call in the block. ``parse(content:tools:)``
/// returns the first call for protocol compatibility.
public struct HunyuanToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<tool_calls>"
    public let endTag: String? = "</tool_calls>"

    private let toolCallStart = "<tool_call>"
    private let toolCallEnd = "</tool_call>"
    private let toolSeparator = "<tool_sep>"
    private let argKeyStart = "<arg_key>"
    private let argKeyEnd = "</arg_key>"
    private let argValueStart = "<arg_value>"
    private let argValueEnd = "</arg_value>"

    public init() {}

    public func isValidPartialContent(_ toolCallBuffer: String) -> Bool {
        guard let startTag else { return true }

        var text = toolCallBuffer
        if text.hasPrefix(startTag) {
            text.removeFirst(startTag.count)
        }
        if let endTag, let endRange = text.range(of: endTag) {
            text = String(text[..<endRange.lowerBound])
        }

        let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty else { return true }

        if body.count <= toolCallStart.count {
            return toolCallStart.hasPrefix(body)
        }
        return body.hasPrefix(toolCallStart)
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseCalls(content, tools: tools).first
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        parseCalls(toolCallBuffer, tools: tools)
    }

    private func parseCalls(_ content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        let block = stripOuterWrapper(content)
        var calls: [ToolCall] = []
        var searchRange = block.startIndex..<block.endIndex

        while let callStart = block.range(of: toolCallStart, range: searchRange),
              let callEnd = block.range(of: toolCallEnd, range: callStart.upperBound..<block.endIndex)
        {
            let rawCall = String(block[callStart.upperBound..<callEnd.lowerBound])
            if let call = parseSingleCall(rawCall, tools: tools) {
                calls.append(call)
            }
            searchRange = callEnd.upperBound..<block.endIndex
        }

        return calls
    }

    private func stripOuterWrapper(_ content: String) -> String {
        var text = content
        if let startTag, let startRange = text.range(of: startTag) {
            text = String(text[startRange.upperBound...])
        }
        if let endTag, let endRange = text.range(of: endTag, options: .backwards) {
            text = String(text[..<endRange.lowerBound])
        }
        return text
    }

    private func parseSingleCall(_ rawCall: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let separator = rawCall.range(of: toolSeparator) else { return nil }

        let functionName = String(rawCall[..<separator.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !functionName.isEmpty else { return nil }

        let argumentsText = String(rawCall[separator.upperBound...])
        var arguments: [String: any Sendable] = [:]
        var searchRange = argumentsText.startIndex..<argumentsText.endIndex

        while let keyStart = argumentsText.range(of: argKeyStart, range: searchRange),
              let keyEnd = argumentsText.range(
                of: argKeyEnd, range: keyStart.upperBound..<argumentsText.endIndex),
              let valueStart = argumentsText.range(
                of: argValueStart, range: keyEnd.upperBound..<argumentsText.endIndex),
              let valueEnd = argumentsText.range(
                of: argValueEnd, range: valueStart.upperBound..<argumentsText.endIndex)
        {
            let key = String(argumentsText[keyStart.upperBound..<keyEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let value = String(argumentsText[valueStart.upperBound..<valueEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if !key.isEmpty {
                arguments[key] = convertHunyuanArgument(
                    value, functionName: functionName, argumentName: key, tools: tools)
            }
            searchRange = valueEnd.upperBound..<argumentsText.endIndex
        }

        return ToolCall(function: .init(name: functionName, arguments: arguments))
    }

    private func convertHunyuanArgument(
        _ value: String,
        functionName: String,
        argumentName: String,
        tools: [[String: any Sendable]]?
    ) -> any Sendable {
        if let type = getParameterType(
            funcName: functionName, paramName: argumentName, tools: tools)
        {
            let schema = ["type": type] as [String: any Sendable]
            let types = extractTypesFromSchema(schema)
            let normalizedTypes = Set(types.map { $0.lowercased() })
            if normalizedTypes.contains("string") || normalizedTypes.contains("str")
                || normalizedTypes.contains("text"),
               let decoded = parseHunyuanJSONFragment(value) as? String
            {
                return decoded
            }
            return convertValueWithTypes(value, types: types)
        }
        return parseHunyuanJSONFragment(value) ?? value
    }

    private func parseHunyuanJSONFragment(_ value: String) -> (any Sendable)? {
        guard let data = value.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(JSONValue.self, from: data)
        else {
            return nil
        }
        return decoded.sendableValue
    }
}
