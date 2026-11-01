import Foundation

/// On-device dictation cleanup models served by the in-process MLX runtime.
///
/// Each entry pins a Hugging Face revision so downloads are reproducible. Only
/// the files the text runtime needs are fetched.
nonisolated struct LocalCleanupModel: Sendable, Hashable, Identifiable {
    struct File: Sendable, Hashable {
        let name: String
        let byteCount: Int64
        /// Required for large weight files; small JSON files are size-checked only.
        let sha256: String?
    }

    let id: String
    let displayName: String
    let detail: String
    let parameterCount: String
    let recommendedMemoryGB: Int
    let repository: String
    let revision: String
    /// Subdirectory inside the repository that holds the MLX checkpoint.
    let subdirectory: String?
    let files: [File]
    /// FluidDecode checkpoints store the Qwen3.5 linear-attention input projection
    /// fused as one tensor; stock MLX expects four. Split once after download.
    let splitsFusedLinearAttention: Bool

    var directoryName: String {
        self.id
    }

    var totalByteCount: Int64 {
        self.files.reduce(0) { $0 + $1.byteCount }
    }

    var weightsSHA256: String? {
        self.files.first { $0.name == "model.safetensors" }?.sha256
    }

    func remoteURL(for file: File) -> URL? {
        let path = [self.subdirectory, file.name].compactMap(\.self).joined(separator: "/")
        return URL(string: "https://huggingface.co/\(self.repository)/resolve/\(self.revision)/\(path)")
    }
}

nonisolated enum LocalCleanupModelCatalog {
    static let fluid1Mini = LocalCleanupModel(
        id: "local-fluid-1-mini",
        displayName: "Fluid-1 Mini (local)",
        detail: "Fast cleanup. 2B, about 2 GB in memory.",
        parameterCount: "2B 6-bit",
        recommendedMemoryGB: 8,
        repository: "altic-dev/fluid-1-mini",
        revision: "af4b6e10a3631c346f77ca8c1018b59743433aa5",
        subdirectory: "mlx-6bit",
        files: [
            .init(name: "config.json", byteCount: 2948, sha256: nil),
            .init(name: "chat_template.jinja", byteCount: 7755, sha256: nil),
            .init(name: "tokenizer_config.json", byteCount: 9186, sha256: nil),
            .init(
                name: "tokenizer.json",
                byteCount: 19_989_343,
                sha256: "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
            ),
            .init(
                name: "model.safetensors",
                byteCount: 1_579_141_868,
                sha256: "19a5fa2e866d96a4090dc16ce0f8a30a9901d33850a0871be16cfa5e8d008f14"
            ),
        ],
        splitsFusedLinearAttention: true
    )

    static let qwen35_4B = LocalCleanupModel(
        id: "local-qwen3.5-4b",
        displayName: "Qwen3.5 4B (local)",
        detail: "Knows more technical terms but rewrites long dictation more. 4B, about 3.5 GB in memory.",
        parameterCount: "4B 4-bit",
        recommendedMemoryGB: 16,
        repository: "mlx-community/Qwen3.5-4B-MLX-4bit",
        revision: "32f3e8ecf65426fc3306969496342d504bfa13f3",
        subdirectory: nil,
        files: [
            .init(name: "config.json", byteCount: 3366, sha256: nil),
            .init(name: "chat_template.jinja", byteCount: 7756, sha256: nil),
            .init(name: "tokenizer_config.json", byteCount: 1139, sha256: nil),
            .init(
                name: "tokenizer.json",
                byteCount: 19_989_343,
                sha256: "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
            ),
            .init(
                name: "model.safetensors",
                byteCount: 3_034_300_695,
                sha256: "5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db"
            ),
        ],
        splitsFusedLinearAttention: false
    )

    static let all: [LocalCleanupModel] = [fluid1Mini, qwen35_4B]
    static let defaultModel = fluid1Mini

    static func model(id: String) -> LocalCleanupModel? {
        let normalized = id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return self.all.first { $0.id == normalized }
    }
}
