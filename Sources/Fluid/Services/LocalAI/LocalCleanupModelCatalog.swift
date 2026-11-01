import Foundation

/// On-device dictation cleanup models served by the in-process MLX runtime.
///
/// Each entry pins a Hugging Face revision of a standard MLX checkpoint, so
/// downloads are reproducible and need no on-device conversion. Only the files
/// the text runtime needs are fetched.
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
    let files: [File]

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
        URL(string: "https://huggingface.co/\(self.repository)/resolve/\(self.revision)/\(file.name)")
    }
}

nonisolated enum LocalCleanupModelCatalog {
    /// Utter's Qwen3-1.7B dictation cleanup fine-tune (Apache-2.0), converted from
    /// its Q6_K GGUF to MLX 6-bit with `tools/local-ai-eval/convert_gguf.py`.
    static let utter = LocalCleanupModel(
        id: "local-utter-qwen3-1.7b",
        displayName: "Utter Cleanup 1.7B (local)",
        detail: "Dictation cleanup tuned to keep your words. 1.7B, about 1.5 GB in memory.",
        parameterCount: "1.7B 6-bit",
        recommendedMemoryGB: 8,
        repository: "devadathanmb/utter-dictation-cleanup-qwen3-1.7b-mlx-6bit",
        revision: "88131144a300d5bd172866ace7166e5b9b35e258",
        files: [
            .init(name: "config.json", byteCount: 989, sha256: nil),
            .init(name: "chat_template.jinja", byteCount: 4116, sha256: nil),
            .init(name: "tokenizer_config.json", byteCount: 729, sha256: nil),
            .init(
                name: "tokenizer.json",
                byteCount: 11_422_650,
                sha256: "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"
            ),
            .init(
                name: "model.safetensors",
                byteCount: 1_398_193_307,
                sha256: "50e241bd4c97b24f7836958e1e9746ddfab6acc59197e197ceb74866f28b9208"
            ),
        ]
    )

    static let all: [LocalCleanupModel] = [utter]
    static let defaultModel = utter

    static func model(id: String) -> LocalCleanupModel? {
        let normalized = id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return self.all.first { $0.id == normalized }
    }
}
