//
//  PromptPrefixHasher.swift
//  VMLXServer
//
//  Deterministic SHA-256 hash over the static prefix of a chat prompt
//  (system content + canonical tool payloads). Used by the model runtime
//  to key prefix caches across compatible requests.
//

import CryptoKit
import Foundation

public enum PromptPrefixHasher {
    public static func hash(systemContent: String, toolNames: [String]) -> String {
        let tools = toolNames.sorted().joined(separator: "\0")
        let combined = systemContent + "\0" + tools
        return digest(Data(combined.utf8))
    }

    public static func hash(systemContent: String, tools: [Tool]) -> String {
        var payload = Data(systemContent.utf8)
        payload.append(0)
        for tool in tools {
            payload.append(tool.canonicalHashPayload())
            payload.append(0)
        }
        return digest(payload)
    }

    private static func digest(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.prefix(16).map { String(format: "%02x", $0) }.joined()
    }
}
