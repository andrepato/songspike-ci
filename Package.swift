// swift-tools-version: 6.2

import Foundation
import PackageDescription

/// The ACE-Step engine dependency is opt-in via the environment, following
/// the Perotta MLX pattern (PEROTTA_WITH_MLX): without SONGSPIKE_WITH_ACE=1
/// the manifest never resolves the engine package, so the shell builds and
/// the app runs with the engine adapter in its not-integrated state.
/// Verified 2026-11-16 against the cloned engine manifest: the product
/// `ACEStepCore` exists in package `mlx-ace-step-swift` (tools 6.2 —
/// hence this manifest's tools version), and ACEStepCore depends only on
/// mlx-swift products (MLX / MLXNN / MLXFast / MLXRandom).
let withACE = ProcessInfo.processInfo.environment["SONGSPIKE_WITH_ACE"] != nil

var packageDependencies: [Package.Dependency] = []
var appDependencies: [Target.Dependency] = []

if withACE {
    // Local vendored copy (vendor/mlx-ace-step-swift, = upstream v0.3.0
    // @ fc9e4be) instead of the remote package: upstream's manifest
    // declares macOS only, and SwiftPM then floors its targets at an
    // implicit iOS 15.0 — below mlx-swift's iOS 17 minimum, which fails
    // build planning for any iOS consumer. The vendored manifest adds
    // .iOS(.v17); no engine source is modified. (Found at the first
    // engine-linked build, 2026-11-16.)
    packageDependencies.append(
        .package(path: "vendor/mlx-ace-step-swift"))
    appDependencies.append(
        // Product name verified in the engine's Package.swift (2026-11-16).
        .product(name: "ACEStepCore", package: "mlx-ace-step-swift"))
}

let package = Package(
    name: "SongSpike",
    platforms: [
        // The engine manifest declares only .macOS(.v26); it states no
        // iOS floor (ENGINE-NOTES.md #6). MLX Swift's general floor is
        // iOS 17; André's test devices run iOS 26/27, so this is not
        // load-bearing for the spike.
        .iOS("17.0"),
    ],
    products: [
        // xtool builds apps from a library product (same layout as
        // Perotta/101.ai): no .executable product; the @main lives in
        // SongSpikeApp.swift inside the library target.
        .library(name: "SongSpikeApp", targets: ["SongSpikeApp"]),
    ],
    dependencies: packageDependencies,
    targets: [
        .target(
            name: "SongSpikeApp",
            dependencies: appDependencies,
            path: "Sources/SongSpikeApp"
        ),
    ]
)
