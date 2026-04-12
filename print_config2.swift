import Foundation

let path = "/Users/eric/models/Qwen3.5-35B-A3B-4bit/config.json"
let data = try! Data(contentsOf: URL(fileURLWithPath: path))
print(String(data: data, encoding: .utf8)!.prefix(1000))
