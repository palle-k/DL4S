// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NMTSwift",
    platforms: [
        .macOS(.v10_12), .iOS(.v11),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/palle-k/DL4S.git", .branch("develop")),
        // .package(url: "https://github.com/apple/swift-package-manager.git", from: "0.1.0")
        .package(url: "https://github.com/kylef/Commander.git", from: "0.8.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "NMTSwift",
            dependencies: ["DL4S", "Commander"]),
        .testTarget(
            name: "NMTSwiftTests",
            dependencies: ["NMTSwift", "DL4S", "Commander"]),
    ]
)
