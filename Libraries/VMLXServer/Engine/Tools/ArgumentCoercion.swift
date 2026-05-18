import Foundation

/// Shared coercion helpers for tool arguments. Local/quantized models frequently
/// serialize values with wrong JSON types (arrays as strings, numbers as strings, etc.).
/// These helpers normalize common mistakes so tool execution succeeds.
enum ArgumentCoercion {
    /// Coerce to `[String]`: actual array, JSON-encoded string (`"[\"a\"]"`),
    /// or bare string wrapped into a single-element array.
    public static func stringArray(_ value: Any?) -> [String]? {
        if let arr = value as? [String] { return arr }
        if let str = value as? String {
            if let data = str.data(using: .utf8),
                let parsed = try? JSONSerialization.jsonObject(with: data) as? [String]
            {
                return parsed
            }
            let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return [trimmed] }
        }
        return nil
    }

    /// Coerce to `Int`: native int, `NSNumber`, or string-encoded integer (`"30"`).
    public static func int(_ value: Any?) -> Int? {
        if let n = value as? Int { return n }
        if let n = (value as? NSNumber)?.intValue { return n }
        if let s = value as? String, let n = Int(s) { return n }
        return nil
    }

    /// Coerce to `Bool`: native bool, string variants (`"true"`, `"1"`, `"yes"`), or `NSNumber`.
    public static func bool(_ value: Any?) -> Bool? {
        if let b = value as? Bool { return b }
        if let s = value as? String {
            switch s.lowercased() {
            case "true", "1", "yes": return true
            case "false", "0", "no": return false
            default: return nil
            }
        }
        if let n = value as? NSNumber { return n.boolValue }
        return nil
    }
}
