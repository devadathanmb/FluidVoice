import Foundation

/// On-device dictation cleanup models served by the in-process MLX runtime.
///
/// Each entry pins a revision and every file's size, so installs are reproducible:
/// a Hugging Face commit, or a release manifest of the official FluidVoice.app.
nonisolated struct LocalCleanupModel: Sendable, Hashable, Identifiable {
    struct File: Sendable, Hashable {
        let name: String
        let byteCount: Int64
        /// Required for large weight files; small JSON files are size-checked only.
        let sha256: String?
    }

    enum Source: Sendable, Hashable {
        /// Downloaded from a Hugging Face repository, optionally from a subdirectory.
        case huggingFace(repository: String, subdirectory: String?)
        /// Read from a checkpoint the official FluidVoice.app already downloaded.
        /// The fork never downloads these from ALTIC's servers itself.
        case fluidVoiceApp(directoryName: String)
    }

    /// One-time rewrite of a FluidDecode checkpoint into the layout stock MLX loads.
    enum Conversion: Sendable, Hashable {
        case none
        /// The Qwen3.5 linear-attention input projection is stored fused as one tensor.
        case splitFusedLinearAttention
        /// Layers are stored as lookup-table indices; decode them to 8-bit affine MLX.
        case decodeLookupTables
    }

    let id: String
    let displayName: String
    let detail: String
    let parameterCount: String
    let recommendedMemoryGB: Int
    let source: Source
    let revision: String
    let files: [File]
    let conversion: Conversion

    var directoryName: String {
        self.id
    }

    var totalByteCount: Int64 {
        self.files.reduce(0) { $0 + $1.byteCount }
    }

    var weightsSHA256: String? {
        self.files.first { $0.name == "model.safetensors" }?.sha256
    }

    /// Where the pinned files come from: a Hugging Face download or a local file.
    var sourceURL: URL? {
        switch self.source {
        case let .huggingFace(repository, _):
            URL(string: "https://huggingface.co/\(repository)/tree/\(self.revision)")
        case let .fluidVoiceApp(directoryName):
            LocalCleanupModelStore.fluidVoiceAppModelsDirectory.appendingPathComponent(directoryName, isDirectory: true)
        }
    }

    func remoteURL(for file: File) -> URL? {
        guard case let .huggingFace(repository, subdirectory) = self.source else { return nil }
        let path = [subdirectory, file.name].compactMap(\.self).joined(separator: "/")
        return URL(string: "https://huggingface.co/\(repository)/resolve/\(self.revision)/\(path)")
    }
}

nonisolated enum LocalCleanupModelCatalog {
    /// ALTIC's Fluid 1.1 Splash (Qwen3.5-2B), release v3 as FluidVoice.app installs it.
    /// Its 2- to 6-bit lookup-table layers are decoded once into 8-bit MLX weights;
    /// the speculative drafter in the same file is dropped.
    static let fluid11Splash = LocalCleanupModel(
        id: "local-fluid-1.1-splash",
        displayName: "Fluid 1.1 Splash",
        detail: "Best cleanup. Uses the copy FluidVoice.app downloaded. 2B, about 1.7 GB in memory.",
        parameterCount: "2B 8-bit",
        recommendedMemoryGB: 8,
        source: .fluidVoiceApp(directoryName: "fluid-1.1-splash-3-46e150afe5cc"),
        revision: "46e150afe5cc6ea5c4e2d7f426b028a3d87e5c6e",
        files: [
            .init(
                name: "config.json",
                byteCount: 2980,
                sha256: "521fa6144efddd2c2b5526b473b6e5284e7f20c0f220d92c88b6d0fe9d8e6cc4"
            ),
            .init(
                name: "chat_template.jinja",
                byteCount: 7755,
                sha256: "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
            ),
            .init(
                name: "tokenizer_config.json",
                byteCount: 1135,
                sha256: "b2bed5e033438f09f22b0ce9522115b4807d9bbcc3b82bf23372cb74d93ed081"
            ),
            .init(
                name: "tokenizer.json",
                byteCount: 7_000_763,
                sha256: "a327311621c5780b2261e24005fc7b7adf271300594d8e502761cdeb944b0d6c"
            ),
            .init(
                name: "model.safetensors",
                byteCount: 768_562_000,
                sha256: "46e150afe5cc6ea5c4e2d7f426b028a3d87e5c6e28310d2ec4bb514187182e5e"
            ),
        ],
        conversion: .decodeLookupTables
    )

    /// ALTIC's Fluid-1 Mini (Qwen3.5-2B), published for personal non-commercial use.
    static let fluid1Mini = LocalCleanupModel(
        id: "local-fluid-1-mini",
        displayName: "Fluid-1 Mini",
        detail: "Fast cleanup. 2B, about 2 GB in memory.",
        parameterCount: "2B 6-bit",
        recommendedMemoryGB: 8,
        source: .huggingFace(repository: "altic-dev/fluid-1-mini", subdirectory: "mlx-6bit"),
        revision: "af4b6e10a3631c346f77ca8c1018b59743433aa5",
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
        conversion: .splitFusedLinearAttention
    )

    static let all: [LocalCleanupModel] = [fluid11Splash, fluid1Mini]

    /// Splash once it is imported, or while FluidVoice.app has its checkpoint to
    /// import from; otherwise Mini. The imported copy no longer needs the app's.
    static var defaultModel: LocalCleanupModel {
        let splash = self.fluid11Splash
        let usable = LocalCleanupModelStore.isInstalled(splash) || LocalCleanupModelStore.isSourceAvailable(splash)
        return usable ? splash : self.fluid1Mini
    }

    static func model(id: String) -> LocalCleanupModel? {
        let normalized = id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return self.all.first { $0.id == normalized }
    }
}
