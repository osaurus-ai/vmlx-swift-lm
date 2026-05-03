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
        .library(
            name: "MLXDistributedCore",
            targets: ["MLXDistributedCore"]),
        .library(
            name: "MLXDistributedTransport",
            targets: ["MLXDistributedTransport"]),
        .library(
            name: "MLXDistributedJACCL",
            targets: ["MLXDistributedJACCL"]),
    ],
    dependencies: [
        // Bumped 2026-05-02: a21d2af = backport of ml-explore/mlx#3462
        // (retain bound buffers under MTLResourceHazardTrackingModeUntracked
        // + commandBufferWithUnretainedReferences). Fixes Invalid Resource
        // crashes at TurboQuant decode B≥16 on M5 Max. Submodule layered:
        //   7086ba37 backport: Retain bound buffers (#3462)
        //   96aa27a5 trace(metal): mx::malloc tracer
        //   f58e52da trace(custom_kernel): clear_library trace
        //   fa3a9616 fix(metal): keep MTLComputePipelineState alive
        // Env knob MLX_METAL_RETAIN_BOUND_BUFFERS=0 reverts to old behaviour.
        .package(url: "https://github.com/osaurus-ai/mlx-swift", revision: "a21d2afd96912f992a03f5cae3216b065f97929a"),
        .package(url: "https://github.com/swiftlang/swift-syntax.git", from: "600.0.0-latest"),
        // swift-jinja: pinned to osaurus-ai fork at 58d21aa which
        // contains a 1-line parser fix lifting the for-loop iterable
        // from parseFilter() (factor + |filter only) to parseOr()
        // (full binary + comparison + logical hierarchy, excluding
        // ternary). Required for Mistral-Medium-3.5's chat template
        // (`{%- for message in loop_messages + [{...}] %}` was
        // rejected with "Expected '%}' after for loop.. Got plus
        // instead"). All 756 swift-jinja tests pass on the fork.
        // Upstream: https://github.com/osaurus-ai/swift-jinja
        // SwiftPM transitive resolution honors this override because
        // it appears before swift-transformers (which depends on
        // swift-jinja).
        .package(url: "https://github.com/osaurus-ai/swift-jinja", revision: "58d21aa5b69fdd9eb7e23ce2c3730f47db8e0c9d"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
        // SwiftNIO stack — used by MLXDistributedTransport for TLS-backed
        // pipeline-parallel inference (Phase 2). swift-nio is already
        // resolved transitively via swift-transformers; we pin the floor
        // to a recent version known to compile under Swift 6.1 strict
        // concurrency.
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.65.0"),
        .package(url: "https://github.com/apple/swift-nio-ssl.git", from: "2.27.0"),
        .package(url: "https://github.com/apple/swift-nio-http2.git", from: "1.34.0"),
        // swift-certificates for self-signed cert generation in
        // MLXDistributedTransport (TLS-PP transport). swift-asn1 is
        // transitive; we declare swift-crypto explicitly so we can
        // consume its Crypto product directly.
        .package(url: "https://github.com/apple/swift-certificates.git", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-crypto.git", from: "4.0.0"),
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
                "MLXLLM",  // NemotronHOmni wrapper imports MLXLLM (NemotronHModel)
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXVLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLMCommon",
            exclude: [
                "README.md"
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
            name: "MLXDistributedCore",
            dependencies: [],
            path: "Libraries/MLXDistributedCore",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXDistributedTransport",
            dependencies: [
                "MLXDistributedCore",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOSSL", package: "swift-nio-ssl"),
                .product(name: "NIOHTTP2", package: "swift-nio-http2"),
                .product(name: "X509", package: "swift-certificates"),
                .product(name: "Crypto", package: "swift-crypto"),
            ],
            path: "Libraries/MLXDistributedTransport",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXDistributedJACCL",
            dependencies: [
                "MLXDistributedCore",
                // Depend on MLX so its dependency on Cmlx pulls in the
                // mlx_distributed_* C symbols at link time. We reach
                // those symbols via @_silgen_name in JACCL.swift instead
                // of importing Cmlx, since mlx-swift doesn't export Cmlx
                // as a library product.
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/MLXDistributedJACCL",
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
            path: "RunBench",
            exclude: ["coherency-matrix.sh", "test_slice.swift.bak"]
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
        .testTarget(
            name: "MLXDistributedCoreTests",
            dependencies: ["MLXDistributedCore"],
            path: "Tests/MLXDistributedCoreTests"
        ),
        .testTarget(
            name: "MLXDistributedTransportTests",
            dependencies: ["MLXDistributedTransport", "MLXDistributedCore"],
            path: "Tests/MLXDistributedTransportTests"
        ),
        .testTarget(
            name: "MLXDistributedJACCLTests",
            dependencies: ["MLXDistributedJACCL"],
            path: "Tests/MLXDistributedJACCLTests"
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
