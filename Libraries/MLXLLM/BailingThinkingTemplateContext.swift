// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon

/// Bailing/Ling chat templates do not read the generic `enable_thinking`
/// Jinja kwarg. They use the system prompt contract from the upstream
/// template instead: a system message containing "detailed thinking on"
/// enables `<think>...</think>` output, while "detailed thinking off"
/// disables it. Keep the translation in the model package so hosts can keep
/// passing the standard `additionalContext["enable_thinking"]` knob.
enum BailingThinkingTemplateContext {
    private static let thinkingOnDirective = "detailed thinking on"
    private static let thinkingOffDirective = "detailed thinking off"

    static func applies(to modelType: String?) -> Bool {
        modelType?.lowercased().hasPrefix("bailing") == true
    }

    static func apply(
        to messages: [Message],
        modelType: String?,
        additionalContext: [String: any Sendable]?
    ) -> [Message] {
        guard applies(to: modelType),
              let enableThinking = additionalContext?["enable_thinking"] as? Bool
        else {
            return messages
        }

        let directive = enableThinking ? thinkingOnDirective : thinkingOffDirective
        var out = messages
        if let index = out.firstIndex(where: { ($0["role"] as? String) == "system" }),
           let content = out[index]["content"] as? String
        {
            var system = out[index]
            let cleaned = removeExistingDirectives(from: content)
            system["content"] = cleaned.isEmpty ? directive : "\(directive)\n\n\(cleaned)"
            out[index] = system
        } else {
            out.insert(["role": "system", "content": directive], at: 0)
        }
        return out
    }

    private static func removeExistingDirectives(from content: String) -> String {
        content
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { line in
                let normalized = String(line)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                return normalized != thinkingOnDirective
                    && normalized != thinkingOffDirective
            }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
