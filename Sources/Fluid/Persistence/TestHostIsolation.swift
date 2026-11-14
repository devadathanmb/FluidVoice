import Foundation

/// The XCTest host launches the full app. Release and Debug builds share
/// `~/Library/Application Support/FluidVoice`, so without this a test run would open,
/// lock, and write the person's real history, search index, audio, and corrections.
nonisolated enum TestHostIsolation {
    static let isXCTestHost: Bool = {
        #if DEBUG
        ProcessInfo.processInfo.environment.keys.contains(where: { $0.hasPrefix("XCTest") })
            || NSClassFromString("XCTestCase") != nil
        #else
        false
        #endif
    }()

    /// A per-process stand-in for Application Support in the XCTest host; nil otherwise.
    static let applicationSupportOverride: URL? = isXCTestHost
        ? FileManager.default.temporaryDirectory
        .appendingPathComponent("FluidVoice-XCTest-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        : nil

    static var applicationSupportDirectory: URL {
        applicationSupportOverride
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    }
}
