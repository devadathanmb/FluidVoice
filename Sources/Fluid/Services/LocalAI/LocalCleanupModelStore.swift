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

    static func directory(for model: LocalCleanupModel) -> URL {
        self.rootDirectory.appendingPathComponent(model.directoryName, isDirectory: true)
    }

    static func isInstalled(_ model: LocalCleanupModel) -> Bool {
        let marker = self.directory(for: model).appendingPathComponent(self.readyMarker)
        guard let revision = try? String(contentsOf: marker, encoding: .utf8) else { return false }
        return revision.trimmingCharacters(in: .whitespacesAndNewlines) == model.revision
    }

    /// Download any missing files, verify them, and convert the checkpoint if needed.
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
            // A split checkpoint no longer matches the remote hash; the ready marker
            // is missing here, so always re-fetch weights to restart from a clean state.
            let reusable = file.name != "model.safetensors" || !model.splitsFusedLinearAttention
            if reusable, self.fileSize(destination) == file.byteCount {
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
            if let sha256 = file.sha256, try self.sha256(of: temporary) != sha256 {
                throw LocalCleanupError.checksumMismatch(file.name)
            }
            try? fileManager.removeItem(at: destination)
            try fileManager.moveItem(at: temporary, to: destination)
            completed += file.byteCount
            await progress?(completed, expected)
        }

        if model.splitsFusedLinearAttention {
            try self.splitFusedLinearAttention(in: directory)
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
    case modelNotInstalled(String)
    case unknownModel(String)
    case httpStatus(Int)

    var errorDescription: String? {
        switch self {
        case let .invalidURL(file): "Invalid download URL for \(file)."
        case let .sizeMismatch(file): "Downloaded \(file) has the wrong size."
        case let .checksumMismatch(file): "Downloaded \(file) failed its checksum."
        case let .unsupportedCheckpoint(reason): "Unsupported model checkpoint: \(reason)."
        case let .modelNotInstalled(id): "Local model \(id) is not downloaded."
        case let .unknownModel(id): "Unknown local model \(id)."
        case let .httpStatus(code): "Model download failed with HTTP \(code)."
        }
    }
}

/// URLSession download with byte progress, delivered to an async handler.
private final nonisolated class FileDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (Int64) async -> Void
    private var lastReported = Date.distantPast
    private let lock = NSLock()

    private init(onProgress: @escaping @Sendable (Int64) async -> Void) {
        self.onProgress = onProgress
    }

    static func download(_ url: URL, onProgress: @escaping @Sendable (Int64) async -> Void) async throws -> URL {
        let delegate = FileDownloader(onProgress: onProgress)
        let (location, response) = try await URLSession.shared.download(from: url, delegate: delegate)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            try? FileManager.default.removeItem(at: location)
            throw LocalCleanupError.httpStatus(http.statusCode)
        }
        // The system deletes `location` when this returns; move it somewhere we own.
        let owned = FileManager.default.temporaryDirectory.appendingPathComponent("fv-model-\(UUID().uuidString)")
        try FileManager.default.moveItem(at: location, to: owned)
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
        guard shouldReport else { return }
        let handler = self.onProgress
        Task { await handler(totalBytesWritten) }
    }
}
