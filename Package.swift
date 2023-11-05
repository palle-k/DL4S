// swift-tools-version:5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "DL4S",
    platforms: [
        .macOS(.v10_15),
        .iOS(.v13),
        .tvOS(.v13),
        .watchOS(.v6)
    ],
    products: [
        .library(
            name: "DL4S",
            targets: ["DL4S", "MKL"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "MKL",
            dependencies: []),
        .target(
            name: "DL4S",
            dependencies: ["MKL"]
        ),
        .testTarget(
            name: "DL4STests",
            dependencies: ["DL4S"],
            resources: [
                .copy("t10k-images.idx3-ubyte"),
                .copy("t10k-labels.idx1-ubyte"),
                .copy("train-images.idx3-ubyte"),
                .copy("train-labels.idx1-ubyte")
            ]
        )
    ]
)
