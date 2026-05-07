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
        .library(
            name: "MLXDistributedTP",
            targets: ["MLXDistributedTP"]),
    ],
    dependencies: [
        // 2026-05-04 reverted to 0a56f90 (was a21d2af). The advance to
        // a21d2af pinned `osaurus-ai/mlx@7086ba37`, an INCOMPLETE backport
        // of upstream ml-explore/mlx#3462: `mlx/backend/metal/eval.cpp:62`
        // calls `encoder.take_retained_buffers()` but the corresponding
        // `auto& encoder = metal::get_command_encoder(s);` declaration was
        // not carried over. The package no longer builds at HEAD with the
        // a21d2af pin; a corrected version of the backport exists at
        // `osaurus-ai/mlx@e577ca02` (refs/heads/backport/3462-retain-bound-buffers)
        // but no mlx-swift branch advances the submodule pointer there yet.
        //
        // 0a56f90 is the last green pin (submodule mlx@96aa27a5,
        // mx::malloc tracer + Bug-1 fix layered on upstream
        // `ce45c525`). Reverting drops the perf-oriented buffer-retain
        // optimization but restores correctness. Re-introduce when the
        // mlx-swift backport branch points at `e577ca02` or later.
        .package(url: "https://github.com/osaurus-ai/mlx-swift", revision: "0a56f9041d56b4b8161f67a6cbd540ae66efc9fd"),
        .package(url: "https://github.com/swiftlang/swift-syntax.git", from: "600.0.0-latest"),
        // Osaurus-owned Jinja chain. `osaurus-ai/Jinja` carries the
        // HuggingFace swift-jinja 2.3.5 code on the `osaurus/hf-2.3.5`
        // branch/tag, and `osaurus-ai/swift-transformers` is the 1.3.0
        // transformer package with its Jinja dependency pointed at that
        // osaurus repo. Keeping both edges in the osaurus org avoids SwiftPM's
        // duplicate package-identity warning while retaining the 2.x Jinja API.
        .package(url: "https://github.com/osaurus-ai/Jinja.git", from: "2.0.0"),
        .package(url: "https://github.com/osaurus-ai/swift-transformers", revision: "b4a094b34b997167549c7f45bde16c80f18ed5a8"),
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
        // consume its Crypto product directly. The version range is
        // intentionally permissive (3.x-4.x) because the only APIs
        // touched (`SHA256.hash`, `P256.Signing.PrivateKey()`) are
        // stable across major versions, and host apps that pin
        // `apple/containerization` (still on swift-crypto 3.x as of
        // 0.32.0) need the 3.x branch to resolve cleanly.
        .package(url: "https://github.com/apple/swift-certificates.git", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-crypto.git", "3.0.0"..<"5.0.0"),
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
            name: "CmlxDistributedShim",
            dependencies: [],
            path: "Libraries/CmlxDistributedShim",
            publicHeadersPath: "include",
            cSettings: [
                // mlx-c headers needed by the shim.
                .headerSearchPath("../../.build/checkouts/mlx-swift/Source/Cmlx/mlx-c"),
            ],
            // The two MlxC*.cpp files are vendored verbatim from the mlx-swift
            // checkout (`Source/Cmlx/mlx-c/mlx/c/distributed.cpp` and
            // `distributed_group.cpp`). The mlx-swift Package.swift excludes
            // these from the Cmlx target, so we compile them ourselves to
            // satisfy the `_mlx_distributed_*` symbols our C shim references.
            // Re-vendor these files when bumping the mlx-swift pin if the
            // upstream C ABI changes.
            cxxSettings: [
                .headerSearchPath("../../.build/checkouts/mlx-swift/Source/Cmlx/mlx-c"),
                .headerSearchPath("../../.build/checkouts/mlx-swift/Source/Cmlx/mlx"),
                .headerSearchPath("../../.build/checkouts/mlx-swift/Source/Cmlx/json/single_include/nlohmann"),
                .headerSearchPath("../../.build/checkouts/mlx-swift/Source/Cmlx/fmt/include"),
            ]
        ),
        .target(
            name: "CmlxGraphShim",
            dependencies: [],
            path: "Libraries/CmlxGraphShim",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("../../.build/checkouts/mlx-swift/Source/Cmlx/mlx-c"),
            ],
            cxxSettings: [
                .headerSearchPath("../../.build/checkouts/mlx-swift/Source/Cmlx/mlx-c"),
                .headerSearchPath("../../.build/checkouts/mlx-swift/Source/Cmlx/mlx"),
                .headerSearchPath("../../.build/checkouts/mlx-swift/Source/Cmlx/fmt/include"),
            ]
        ),
        .target(
            name: "MLXDistributedTP",
            dependencies: [
                "MLXDistributedCore",
                "MLXDistributedJACCL",
                "CmlxDistributedShim",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Libraries/MLXDistributedTP",
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
            name: "TPRankWorker",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXHuggingFace",
                "MLXDistributedTP",
                "MLXDistributedJACCL",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Tools/TPRankWorker"
        ),
        .executableTarget(
            name: "RunBench",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXHuggingFace",
                "CmlxGraphShim",
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
                .product(name: "Jinja", package: "jinja"),
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
    ],
    // C++20 needed by the vendored mlx-c distributed C ABI shim files in
    // CmlxDistributedShim (they include `mlx/distributed/ops.h` which uses
    // `std::is_same_v` / `std::is_convertible_v` and other C++17/20-only
    // type traits). All other C++ targets compile fine at C++20 too.
    cxxLanguageStandard: .gnucxx20
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}

if Context.environment["VMLINUX_ENABLE_DISTRIBUTED_TESTS"] == "1" {
    package.targets.append(contentsOf: [
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
        .testTarget(
            name: "MLXDistributedTPTests",
            dependencies: ["MLXDistributedTP", "MLXDistributedCore"],
            path: "Tests/MLXDistributedTPTests"
        ),
    ])
}
