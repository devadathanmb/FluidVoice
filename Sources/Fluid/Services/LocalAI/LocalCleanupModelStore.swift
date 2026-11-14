import CryptoKit
import Foundation
import MLX

/// Downloads, verifies, and prepares local cleanup checkpoints on disk.
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

    /// Where the official FluidVoice.app keeps the Fluid Intelligence checkpoints it downloaded.
    static var fluidVoiceAppModelsDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        return base
            .appendingPathComponent("FluidIntelligence", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    /// True when the model can be installed: always for downloads, and for a
    /// FluidVoice.app checkpoint only while the app's copy has every pinned file
    /// at its pinned size. Hashes are checked during install.
    static func isSourceAvailable(_ model: LocalCleanupModel) -> Bool {
        guard case .fluidVoiceApp = model.source else { return true }
        guard let source = model.sourceURL else { return false }
        return model.files.allSatisfy { self.fileSize(source.appendingPathComponent($0.name)) == $0.byteCount }
    }

    static func directory(for model: LocalCleanupModel) -> URL {
        self.rootDirectory.appendingPathComponent(model.directoryName, isDirectory: true)
    }

    static func isInstalled(_ model: LocalCleanupModel) -> Bool {
        let marker = self.directory(for: model).appendingPathComponent(self.readyMarker)
        guard let revision = try? String(contentsOf: marker, encoding: .utf8) else { return false }
        return revision.trimmingCharacters(in: .whitespacesAndNewlines) == model.revision
    }

    typealias Progress = @Sendable (_ written: Int64, _ expected: Int64) async -> Void

    /// Fetch and verify the model's files, then convert the checkpoint if needed.
    static func install(_ model: LocalCleanupModel, progress: Progress?) async throws -> URL {
        let directory = self.directory(for: model)
        if self.isInstalled(model) {
            return directory
        }

        let fileManager = FileManager.default
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        try? fileManager.removeItem(at: directory.appendingPathComponent(self.readyMarker))

        switch model.source {
        case .huggingFace:
            try await self.download(model, into: directory, progress: progress)
        case .fluidVoiceApp:
            try await self.importFromFluidVoiceApp(model, into: directory, progress: progress)
        }

        switch model.conversion {
        case .none, .decodeLookupTables:
            break
        case .splitFusedLinearAttention:
            try self.splitFusedLinearAttention(in: directory)
        }
        try model.revision.write(
            to: directory.appendingPathComponent(self.readyMarker), atomically: true, encoding: .utf8
        )
        return directory
    }

    /// Download any missing files and verify them. Only hash-pinned files are reused
    /// from disk, and only after their hash matches again, so a revision bump that
    /// keeps a file's size can never leave stale bytes behind, and a weights file
    /// split by an interrupted install is fetched again. Small unhashed files are
    /// always fetched fresh.
    private static func download(_ model: LocalCleanupModel, into directory: URL, progress: Progress?) async throws {
        let fileManager = FileManager.default
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
    }

    /// Verify FluidVoice.app's copy against the pinned manifest, then build an MLX
    /// checkpoint from it. Progress covers hashing the source, then decoding.
    private static func importFromFluidVoiceApp(
        _ model: LocalCleanupModel,
        into directory: URL,
        progress: Progress?
    ) async throws {
        guard let source = model.sourceURL, self.isSourceAvailable(model) else {
            throw LocalCleanupError.sourceUnavailable(model.displayName)
        }
        let fileManager = FileManager.default
        let weightsBytes = model.files.first { $0.name == "model.safetensors" }?.byteCount ?? 0
        let expected = model.totalByteCount + weightsBytes
        var completed: Int64 = 0
        for file in model.files {
            try Task.checkCancellation()
            let original = source.appendingPathComponent(file.name)
            guard file.sha256 != nil, self.isVerified(original, as: file) else {
                throw LocalCleanupError.checksumMismatch(file.name)
            }
            completed += file.byteCount
            await progress?(completed, expected)
        }

        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
            let destination = directory.appendingPathComponent(name)
            try? fileManager.removeItem(at: destination)
            try fileManager.copyItem(at: source.appendingPathComponent(name), to: destination)
        }

        guard model.conversion == .decodeLookupTables else {
            throw LocalCleanupError.unsupportedCheckpoint("FluidVoice.app checkpoints need lookup-table decoding")
        }
        let base = completed
        let quantization = try await LocalCleanupLookupTableDecoder.convert(
            weights: source.appendingPathComponent("model.safetensors"),
            to: directory.appendingPathComponent("model.safetensors")
        ) { fraction in
            await progress?(base + Int64(Double(weightsBytes) * fraction), expected)
        }
        try LocalCleanupLookupTableDecoder.writeConfig(
            from: source.appendingPathComponent("config.json"),
            to: directory.appendingPathComponent("config.json"),
            quantization: quantization
        )
        await progress?(expected, expected)
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

    // MARK: - Fused projection split

    /// FluidDecode stores `linear_attn.in_proj` rows as [qkv | z | b | a]. Affine
    /// quantization groups run along the input dimension, so row slices are exact.
    static func splitFusedLinearAttention(in directory: URL) throws {
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        let configURL = directory.appendingPathComponent("config.json")
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]
        let text = config["text_config"] as? [String: Any] ?? config
        guard let keyHeads = text["linear_num_key_heads"] as? Int,
              let valueHeads = text["linear_num_value_heads"] as? Int,
              let keyDim = text["linear_key_head_dim"] as? Int,
              let valueDim = text["linear_value_head_dim"] as? Int
        else { throw LocalCleanupError.unsupportedCheckpoint("missing linear attention dimensions") }

        let sizes: [(String, Int)] = [
            ("qkv", 2 * keyHeads * keyDim + valueHeads * valueDim),
            ("z", valueHeads * valueDim),
            ("b", valueHeads),
            ("a", valueHeads),
        ]
        let totalRows = sizes.reduce(0) { $0 + $1.1 }

        let weights = try loadArrays(url: weightsURL)
        let marker = "linear_attn.in_proj."
        guard weights.keys.contains(where: { $0.contains(marker) }) else { return }

        var converted: [String: MLXArray] = [:]
        for (key, value) in weights {
            guard let range = key.range(of: marker) else {
                converted[key] = value
                continue
            }
            guard value.dim(0) == totalRows else {
                throw LocalCleanupError.unsupportedCheckpoint("\(key) has \(value.dim(0)) rows, expected \(totalRows)")
            }
            let prefix = key[..<range.lowerBound]
            let suffix = key[range.upperBound...]
            var offset = 0
            for (name, rows) in sizes {
                converted["\(prefix)linear_attn.in_proj_\(name).\(suffix)"] = value[offset..<offset + rows]
                offset += rows
            }
        }
        let temporary = directory.appendingPathComponent("model.split.safetensors")
        try save(arrays: converted, metadata: ["format": "mlx"], url: temporary)
        _ = try FileManager.default.replaceItemAt(weightsURL, withItemAt: temporary)
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
    case unsupportedCheckpoint(String)
    case sourceUnavailable(String)
    case modelNotInstalled(String)
    case unknownModel(String)
    case httpStatus(Int)

    var errorDescription: String? {
        switch self {
        case let .invalidURL(file): "Invalid download URL for \(file)."
        case let .sizeMismatch(file): "Downloaded \(file) has the wrong size."
        case let .checksumMismatch(file): "Downloaded \(file) failed its checksum."
        case let .unsupportedCheckpoint(reason): "Unsupported model checkpoint: \(reason)."
        case let .sourceUnavailable(name): "\(name) is not downloaded in FluidVoice.app. Download it there first."
        case let .modelNotInstalled(id): "Local model \(id) is not downloaded."
        case let .unknownModel(id): "Unknown local model \(id)."
        case let .httpStatus(code): "Model download failed with HTTP \(code)."
        }
    }
}

/// URLSession download with byte progress, delivered in order to an async handler.
///
/// Uses a session-level delegate and a classic download task: the async
/// `URLSession.download(from:delegate:)` never delivers `didWriteData` to its task
/// delegate, so progress froze at the small files while the weights downloaded.
private final nonisolated class FileDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let progress: AsyncStream<Int64>.Continuation
    private let lock = NSLock()
    private var lastReported = Date.distantPast
    private var continuation: CheckedContinuation<URL, Error>?
    private var result: Result<URL, Error>?

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
        let delegate = FileDownloader(progress: continuation)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        // A session retains its delegate until invalidated.
        defer { session.finishTasksAndInvalidate() }
        let task = session.downloadTask(with: url)

        do {
            let file = try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (resume: CheckedContinuation<URL, Error>) in
                    delegate.lock.withLock { delegate.continuation = resume }
                    task.resume()
                }
            } onCancel: {
                task.cancel()
            }
            continuation.finish()
            await reporter.value
            return file
        } catch {
            continuation.finish()
            reporter.cancel()
            throw error
        }
    }

    func urlSession(_: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let outcome: Result<URL, Error>
        if let http = downloadTask.response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            outcome = .failure(LocalCleanupError.httpStatus(http.statusCode))
        } else {
            // The system deletes `location` when this returns; move it somewhere we own.
            let owned = FileManager.default.temporaryDirectory.appendingPathComponent("fv-model-\(UUID().uuidString)")
            do {
                try FileManager.default.moveItem(at: location, to: owned)
                outcome = .success(owned)
            } catch {
                outcome = .failure(error)
            }
        }
        self.lock.withLock { self.result = outcome }
    }

    func urlSession(_: URLSession, task _: URLSessionTask, didCompleteWithError error: Error?) {
        let (resume, outcome): (CheckedContinuation<URL, Error>?, Result<URL, Error>?) = self.lock.withLock {
            defer { self.continuation = nil }
            return (self.continuation, self.result)
        }
        if let error {
            if case let .success(file) = outcome {
                try? FileManager.default.removeItem(at: file)
            }
            resume?.resume(throwing: error)
        } else {
            resume?.resume(with: outcome ?? .failure(URLError(.unknown)))
        }
    }

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
