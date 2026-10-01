// swift-tools-version:6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

// Intel oneAPI MKL and IPP exist for x86_64 Linux only. On other hosts, the system library target
// is left out so that a build without the trait does not look for pkg-config files, and the trait
// defines MKL_UNSUPPORTED_PLATFORM instead of MKL_ENABLE. MKLPlatformCheck.swift turns that flag
// into a readable compile-time error.
#if os(Linux) && arch(x86_64)
// The MKL include path and link line come from the pkg-config file that oneAPI MKL ships.
// The threaded variant (GNU OpenMP) is used on purpose: training is often not data-parallel,
// so each operation benefits from the parallel kernels.
let mklTargets: [Target] = [
    .systemLibrary(
        name: "CMKL",
        pkgConfig: "mkl-dynamic-lp64-gomp",
        providers: [
            .apt(["intel-oneapi-mkl-devel", "intel-oneapi-ipp-devel"]),
        ],
    ),
]
let mklDependencies: [Target.Dependency] = [
    .target(name: "CMKL", condition: .when(traits: ["MKL"])),
]
let mklSwiftSettings: [SwiftSetting] = [
    .define("MKL_ENABLE", .when(traits: ["MKL"])),
]
#else
let mklTargets: [Target] = []
let mklDependencies: [Target.Dependency] = []
let mklSwiftSettings: [SwiftSetting] = [
    .define("MKL_UNSUPPORTED_PLATFORM", .when(traits: ["MKL"])),
]
#endif

let package = Package(
    name: "DL4S",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .tvOS(.v18),
        .watchOS(.v11),
    ],
    products: [
        .library(
            name: "DL4S",
            targets: ["DL4S"],
        ),
    ],
    traits: [
        .trait(
            name: "MKL",
            description: "Accelerates the CPU backend with Intel oneAPI MKL and IPP. Requires x86_64 Linux and the oneAPI environment (source the setvars.sh script of oneAPI before you build).",
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.5.0"),
        .package(url: "https://github.com/SimplyDanny/SwiftLintPlugins", from: "0.65.1"),
        .package(url: "https://github.com/nicklockwood/SwiftFormat", from: "0.63.0"),
        .package(url: "https://github.com/swiftlang/swift-syntax", "601.0.0" ..< "605.0.0"),
    ],
    targets: mklTargets + [
        .macro(
            name: "DL4SMacros",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
        ),
        .target(
            name: "DL4S",
            dependencies: mklDependencies + ["DL4SMacros"],
            // The Metal kernels are copied as source and compiled at run time, so the build needs no Metal compiler.
            resources: [
                .copy("Engine/GPU/Shaders"),
            ],
            swiftSettings: mklSwiftSettings,
        ),
        .testTarget(
            name: "DL4SMacrosTests",
            dependencies: [
                "DL4SMacros",
                .product(name: "SwiftSyntaxMacrosGenericTestSupport", package: "swift-syntax"),
            ],
        ),
        .testTarget(
            name: "DL4STests",
            dependencies: ["DL4S"],
            resources: [
                .copy("t10k-images.idx3-ubyte"),
                .copy("t10k-labels.idx1-ubyte"),
                .copy("train-images.idx3-ubyte"),
                .copy("train-labels.idx1-ubyte"),
            ],
            swiftSettings: mklSwiftSettings,
        ),
    ],
    swiftLanguageModes: [.v6],
)
