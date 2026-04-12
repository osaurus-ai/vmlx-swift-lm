import Foundation

let path = "/Users/eric/models/Qwen3.5-35B-A3B-4bit/config.json"
let data = try! Data(contentsOf: URL(fileURLWithPath: path))
let dict = try! JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]

print("linear_num_key_heads = \(dict["linear_num_key_heads"] ?? "nil")")
print("linear_num_value_heads = \(dict["linear_num_value_heads"] ?? "nil")")
