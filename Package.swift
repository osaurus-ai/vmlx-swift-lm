// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
    name: "vmlx-swift-lm",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "MLXLLM",
            targets: ["MLXLLM"]),
        .library(
            name: "MLXVLM",
            targets: ["MLXVLM"]),
        .library(
            name: "MLXLMCommon",
            targets: ["MLXLMCommon"]),
        .library(
            name: "MLXEmbedders",
            targets: ["MLXEmbedders"]),
        .library(
            name: "MLXHuggingFace",
            targets: ["MLXHuggingFace"]),
        .library(
            name: "BenchmarkHelpers",
            targets: ["BenchmarkHelpers"]),
        .library(
            name: "IntegrationTestHelpers",
            targets: ["IntegrationTestHelpers"]),
    ],
    dependencies: [
        // Phase 0.5 (2026-04-21) switched to the `-distributed-phase0`
        // branch which enables mlx-c's distributed bridge + the ring
        // TCP backend. The base `osaurus-0.31.3` branch excluded those
        // sources ("do not build distributed support (yet)"), which
        // left our Swift distributed bindings linking only against
        // weak-alias fallback stubs. The distributed-phase0 branch adds
        // them back. See `Libraries/MLXLMCommon/Distributed/DISTRIBUTED-DESIGN.md`
        // and commit 8de8295 on osaurus-ai/mlx-swift for the exact patch.
        .package(url: "https://github.com/osaurus-ai/mlx-swift", branch: "osaurus-0.31.3-distributed-phase0"),
        .package(url: "https://github.com/swiftlang/swift-syntax.git", from: "600.0.0-latest"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.21"),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXVLM",
            exclude: [
                "README.md"
            ]
        ),
        // Single-rank fallback for the mlx-core distributed C API.
        //
        // Upstream mlx-swift deliberately excludes mlx's distributed sources
        // from its Cmlx build (`do not build distributed support (yet)` —
        // Package.swift line 193 in osaurus-ai/mlx-swift @ osaurus-0.31.3).
        // That leaves the `mlx_distributed_*` symbols declared in public
        // Cmlx headers but not linked. This C target supplies weak-alias
        // stubs so `MLXDistributed.swift` links today on a single host.
        //
        // When upstream mlx-swift is patched to include the distributed
        // sources + at least one backend (Phase 0.5 in
        // `Libraries/MLXLMCommon/Distributed/DISTRIBUTED-DESIGN.md`), the
        // real symbols take precedence over these weak stubs automatically
        // — no Swift-side change required.
        .target(
            name: "MLXDistributedCFallback",
            path: "Libraries/MLXLMCommon/Distributed/CFallback",
            publicHeadersPath: "include"
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                "MLXDistributedCFallback",
            ],
            path: "Libraries/MLXLMCommon",
            exclude: [
                "README.md",
                "Distributed/CFallback",
            ]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .target(name: "MLXLMCommon"),
            ],
            path: "Libraries/MLXEmbedders",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "BenchmarkHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/BenchmarkHelpers"
        ),
        .target(
            name: "IntegrationTestHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/IntegrationTestHelpers",
            exclude: ["README.md"]
        ),
        .executableTarget(
            name: "CompileBench",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "CompileBench"
        ),
        .executableTarget(
            name: "RunBench",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXHuggingFace",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "RunBench"
        ),
        .testTarget(
            name: "MLXLMTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "MLXHuggingFace",
            ],
            path: "Tests/MLXLMTests",
            exclude: [
                "README.md"
            ],
            resources: [.process("Resources/1080p_30.mov"), .process("Resources/audio_only.mov")]
        ),
        .macro(
            name: "MLXHuggingFaceMacros",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
            path: "Libraries/MLXHuggingFaceMacros"
        ),
        .target(
            name: "MLXHuggingFace",
            dependencies: [
                "MLXHuggingFaceMacros",
                "MLXLMCommon",
            ],
            path: "Libraries/MLXHuggingFace"
        ),
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
