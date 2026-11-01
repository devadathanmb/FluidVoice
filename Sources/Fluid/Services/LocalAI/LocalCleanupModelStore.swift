import CryptoKit
import Foundation

/// Downloads and verifies local cleanup checkpoints on disk.
nonisolated enum LocalCleanupModelStore {
    static let readyMarker = ".fluidvoice-ready"

    static var rootDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        return base
            .appendingPathComponent("FluidVoice", isDirectory: true)
            .appendingPathComponent("LocalCleanup", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    static func directory(for model: LocalCleanupModel) -> URL {
        self.rootDirectory.appendingPathComponent(model.directoryName, isDirectory: true)
    }

    static func isInstalled(_ model: LocalCleanupModel) -> Bool {
        let marker = self.directory(for: model).appendingPathComponent(self.readyMarker)
        guard let revision = try? String(contentsOf: marker, encoding: .utf8) else { return false }
        return revision.trimmingCharacters(in: .whitespacesAndNewlines) == model.revision
    }

    /// Download any missing files and verify them. Only hash-pinned files are reused
    /// from disk, and only after their hash matches again, so a revision bump that
    /// keeps a file's size can never leave stale bytes behind. Small unhashed files
    /// are always fetched fresh.
    static func install(
        _ model: LocalCleanupModel,
        progress: (@Sendable (_ written: Int64, _ expected: Int64) async -> Void)?
    ) async throws -> URL {
        let directory = self.directory(for: model)
        if self.isInstalled(model) {
            return directory
        }

        let fileManager = FileManager.default
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        try? fileManager.removeItem(at: directory.appendingPathComponent(self.readyMarker))

        let expected = model.totalByteCount
        var completed: Int64 = 0
        for file in model.files {
            try Task.checkCancellation()
            let destination = directory.appendingPathComponent(file.name)
            if self.isVerified(destination, as: file) {
                completed += file.byteCount
                await progress?(completed, expected)
                continue
            }
            guard let url = model.remoteURL(for: file) else { throw LocalCleanupError.invalidURL(file.name) }
            let base = completed
            let temporary = try await FileDownloader.download(url) { written in
                await progress?(base + written, expected)
            }
            defer { try? fileManager.removeItem(at: temporary) }

            guard self.fileSize(temporary) == file.byteCount else {
                throw LocalCleanupError.sizeMismatch(file.name)
            }
            if file.sha256 != nil, !self.isVerified(temporary, as: file) {
                throw LocalCleanupError.checksumMismatch(file.name)
            }
            try? fileManager.removeItem(at: destination)
            try fileManager.moveItem(at: temporary, to: destination)
            completed += file.byteCount
            await progress?(completed, expected)
        }

        try model.revision.write(
            to: directory.appendingPathComponent(self.readyMarker), atomically: true, encoding: .utf8
        )
        return directory
    }

    static func remove(_ model: LocalCleanupModel) throws {
        let directory = self.directory(for: model)
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
    }

    /// Model directories other than `model`, such as checkpoints from older catalogs.
    static func otherInstalledDirectories(keeping model: LocalCleanupModel) -> [URL] {
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: self.rootDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return entries.filter { $0.hasDirectoryPath && $0.lastPathComponent != model.directoryName }
    }

    // MARK: - Helpers

    /// True when `url` has `file`'s size and pinned hash. Unpinned files never verify.
    static func isVerified(_ url: URL, as file: LocalCleanupModel.File) -> Bool {
        guard let expected = file.sha256, self.fileSize(url) == file.byteCount else { return false }
        return (try? self.sha256(of: url)) == expected
    }

    static func fileSize(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

nonisolated enum LocalCleanupError: LocalizedError {
    case invalidURL(String)
    case sizeMismatch(String)
    case checksumMismatch(String)
    case modelNotInstalled(String)
    case unknownModel(String)
    case httpStatus(Int)

    var errorDescription: String? {
        switch self {
        case let .invalidURL(file): "Invalid download URL for \(file)."
        case let .sizeMismatch(file): "Downloaded \(file) has the wrong size."
        case let .checksumMismatch(file): "Downloaded \(file) failed its checksum."
        case let .modelNotInstalled(id): "Local model \(id) is not downloaded."
        case let .unknownModel(id): "Unknown local model \(id)."
        case let .httpStatus(code): "Model download failed with HTTP \(code)."
        }
    }
}

/// URLSession download with byte progress, delivered in order to an async handler.
private final nonisolated class FileDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let progress: AsyncStream<Int64>.Continuation
    private var lastReported = Date.distantPast
    private let lock = NSLock()

    private init(progress: AsyncStream<Int64>.Continuation) {
        self.progress = progress
    }

    static func download(_ url: URL, onProgress: @escaping @Sendable (Int64) async -> Void) async throws -> URL {
        // One consumer drains a newest-only stream, so updates never arrive out of
        // order and the last one lands before this download returns.
        let (stream, continuation) = AsyncStream.makeStream(of: Int64.self, bufferingPolicy: .bufferingNewest(1))
        let reporter = Task {
            for await written in stream {
                await onProgress(written)
            }
        }
        defer {
            continuation.finish()
            reporter.cancel()
        }

        let delegate = FileDownloader(progress: continuation)
        let (location, response) = try await URLSession.shared.download(from: url, delegate: delegate)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            try? FileManager.default.removeItem(at: location)
            throw LocalCleanupError.httpStatus(http.statusCode)
        }
        // The system deletes `location` when this returns; move it somewhere we own.
        let owned = FileManager.default.temporaryDirectory.appendingPathComponent("fv-model-\(UUID().uuidString)")
        try FileManager.default.moveItem(at: location, to: owned)
        continuation.finish()
        await reporter.value
        return owned
    }

    func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didFinishDownloadingTo _: URL) {}

    func urlSession(
        _: URLSession, downloadTask _: URLSessionDownloadTask, didWriteData _: Int64,
        totalBytesWritten: Int64, totalBytesExpectedToWrite _: Int64
    ) {
        let now = Date()
        let shouldReport: Bool = self.lock.withLock {
            guard now.timeIntervalSince(self.lastReported) > 0.25 else { return false }
            self.lastReported = now
            return true
        }
        if shouldReport {
            self.progress.yield(totalBytesWritten)
        }
    }
}
