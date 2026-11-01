import Foundation

/// Installs the fork's on-device cleanup engine behind the private AI provider seam.
///
/// `PrivateAIProviderBootstrap` calls `install()` when `PRIVATE_AI_PROVIDER` is
/// defined, which routes Settings, downloads, prewarm, dictation, and Rewrite
/// through `LocalCleanupRuntime`.
nonisolated enum PrivateAIProviderBridge {
    static func install() {
        PrivateAIProviderRegistry.feature = LocalCleanupFeature()
        PrivateAIProviderRegistry.integration = LocalCleanupIntegration()
    }
}

nonisolated struct LocalCleanupFeature: PrivateAIProviderFeatureProviding {
    let isAvailable = true
    let providerID = "local-cleanup"
    let providerName = "Local AI"
    let promptSelectionID = "__PRIVATE_AI_PROVIDER__"
    var defaultModelID: String {
        LocalCleanupModelCatalog.defaultModel.id
    }

    let selectedModelDefaultsKey = "LocalCleanupSelectedModelID"
    let localModelPathDefaultsKey = "LocalCleanupLocalModelPath"
    let prefixCacheDefaultsKey = "LocalCleanupPrefixKVCacheEnabled"
    let boostDefaultsKey = "LocalCleanupBoostEnabled"
    let modelDirectoryName = "LocalCleanup"

    func modelIDs() -> [String] {
        LocalCleanupModelCatalog.all.map(\.id)
    }

    func model(id: String) -> PrivateAIRegisteredModel? {
        LocalCleanupModelCatalog.model(id: id).map(Self.registered)
    }

    func canonicalModelID(for value: String) -> String? {
        LocalCleanupModelCatalog.model(id: value)?.id
    }

    func isKnownModelID(_ value: String) -> Bool {
        LocalCleanupModelCatalog.model(id: value) != nil
    }

    func localModelURL(for model: PrivateAIRegisteredModel, directoryURL _: URL) -> URL {
        guard let local = LocalCleanupModelCatalog.model(id: model.id) else {
            return LocalCleanupModelStore.rootDirectory.appendingPathComponent(model.artifact.filename)
        }
        return LocalCleanupModelStore.directory(for: local)
    }

    static func registered(_ model: LocalCleanupModel) -> PrivateAIRegisteredModel {
        PrivateAIRegisteredModel(
            displayName: model.displayName,
            detail: model.detail,
            isEnabled: true,
            parameterCount: model.parameterCount,
            recommendedMemoryGB: model.recommendedMemoryGB,
            artifact: PrivateAIModelArtifact(
                identifier: model.id,
                filename: model.directoryName,
                downloadURL: URL(string: "https://huggingface.co/\(model.repository)/tree/\(model.revision)"),
                sha256: model.weightsSHA256,
                byteCount: model.totalByteCount,
                version: String(model.revision.prefix(7))
            )
        )
    }
}

nonisolated struct LocalCleanupIntegration: PrivateAIIntegrationProviding {
    private var runtime: LocalCleanupRuntime {
        .shared
    }

    var configuredModelID: String {
        let stored = UserDefaults.standard.string(forKey: LocalCleanupFeature().selectedModelDefaultsKey) ?? ""
        return LocalCleanupModelCatalog.model(id: stored)?.id ?? LocalCleanupModelCatalog.defaultModel.id
    }

    var selectedModel: PrivateAIRegisteredModel {
        LocalCleanupFeature.registered(self.localModel(id: self.configuredModelID) ?? LocalCleanupModelCatalog.defaultModel)
    }

    var configuredLocalModelPath: String? {
        self.localModelPath(for: self.selectedModel)
    }

    var modelDirectoryURL: URL {
        LocalCleanupModelStore.rootDirectory
    }

    var isLocalRuntimeConfigured: Bool {
        self.localModel(id: self.configuredModelID).map(LocalCleanupModelStore.isInstalled) ?? false
    }

    func expectedLocalModelURL(for model: PrivateAIRegisteredModel) -> URL {
        LocalCleanupFeature().localModelURL(for: model, directoryURL: self.modelDirectoryURL)
    }

    func localModelPath(for model: PrivateAIRegisteredModel) -> String? {
        guard let local = self.localModel(id: model.id), LocalCleanupModelStore.isInstalled(local) else { return nil }
        return LocalCleanupModelStore.directory(for: local).path
    }

    func isModelInstalled(_ model: PrivateAIRegisteredModel) -> Bool {
        self.localModel(id: model.id).map(LocalCleanupModelStore.isInstalled) ?? false
    }

    func installedModelURLs(for model: PrivateAIRegisteredModel) -> [URL] {
        guard let local = self.localModel(id: model.id), LocalCleanupModelStore.isInstalled(local) else { return [] }
        return [LocalCleanupModelStore.directory(for: local)]
    }

    /// Lets the app reclaim checkpoints from models that left the catalog.
    func inactiveInstalledModelURLs(keeping model: PrivateAIRegisteredModel) -> [URL] {
        guard let local = self.localModel(id: model.id) else { return [] }
        return LocalCleanupModelStore.otherInstalledDirectories(keeping: local)
    }

    func prepareModel(
        _ model: PrivateAIRegisteredModel,
        progressHandler: PrivateAIModelDownloadProgressHandler?
    ) async throws -> URL {
        let local = try self.requireModel(model.id)
        return try await LocalCleanupModelStore.install(local) { written, expected in
            await progressHandler?(
                PrivateAIModelDownloadProgress(bytesWritten: 0, totalBytesWritten: written, totalBytesExpected: expected)
            )
        }
    }

    func shouldHandleDictation(model: String) -> Bool {
        LocalCleanupFeature().matches(model: model)
    }

    func status(for _: PrivateAIIntegrationService.RuntimeConfiguration) async -> PrivateAIStatus {
        let id = self.configuredModelID
        if await self.runtime.currentModelID == id {
            return PrivateAIStatus(state: .ready, message: nil)
        }
        if await self.runtime.loadingModelID == id {
            return PrivateAIStatus(state: .loading, message: nil)
        }
        guard let local = self.localModel(id: id), LocalCleanupModelStore.isInstalled(local) else {
            return PrivateAIStatus(state: .missingModel, message: "Download the model to use local cleanup.")
        }
        return PrivateAIStatus(state: .configured, message: nil)
    }

    func loadedModelState() async -> PrivateAIIntegrationService.LoadedModelState? {
        guard let id = await self.runtime.currentModelID else { return nil }
        return .init(modelID: id, state: .ready, message: nil)
    }

    func residencySnapshot() async throws -> MeetingResidentModel? {
        guard let id = await self.runtime.currentModelID else { return nil }
        return MeetingResidentModel(id: id, configuration: "local-cleanup")
    }

    func restoreResidency(_ model: MeetingResidentModel) async throws {
        try await self.runtime.load(self.requireModel(model.id))
    }

    func loadModel(_ model: PrivateAIRegisteredModel) async throws -> PrivateAIStatus {
        try await self.runtime.load(self.requireModel(model.id))
        await PrivateAIIntegrationService.postRuntimeDidChange()
        return PrivateAIStatus(state: .ready, message: nil)
    }

    /// Loading proves the weights and tokenizer work; one short cleanup proves generation does.
    func verifyModel(_ model: PrivateAIRegisteredModel) async throws -> PrivateAIStatus {
        try await self.runtime.load(self.requireModel(model.id))
        _ = try await self.runtime.cleanUp("um hello world", vocabulary: [], maxOutputTokens: 16, onText: nil)
        await PrivateAIIntegrationService.postRuntimeDidChange()
        return PrivateAIStatus(state: .ready, message: nil)
    }

    func prewarmDictation() async {
        guard let local = self.localModel(id: self.configuredModelID), LocalCleanupModelStore.isInstalled(local) else { return }
        do {
            try await self.runtime.load(local)
            try await self.runtime.prewarm(vocabulary: Self.vocabulary())
            await PrivateAIIntegrationService.postRuntimeDidChange()
        } catch {
            Self.log("Local cleanup prewarm failed: \(error.localizedDescription)")
        }
    }

    func unloadCachedRuntime(reason: String) async {
        await self.runtime.unload()
        Self.log("Local cleanup model unloaded (\(reason))")
    }

    nonisolated func enhanceDictation(
        _ inputText: String,
        runtime: PrivateAIIntegrationService.RuntimeConfiguration,
        context: PrivateAIIntegrationService.AppContext,
        maxOutputTokens: Int
    ) async throws -> PrivateAIIntegrationService.EnhancementResult {
        try await self.enhanceDictation(
            inputText, runtime: runtime, context: context, maxOutputTokens: maxOutputTokens, streamHandler: nil
        )
    }

    nonisolated func enhanceDictation(
        _ inputText: String,
        runtime _: PrivateAIIntegrationService.RuntimeConfiguration,
        context _: PrivateAIIntegrationService.AppContext,
        maxOutputTokens: Int,
        streamHandler: PrivateAIStreamHandler?
    ) async throws -> PrivateAIIntegrationService.EnhancementResult {
        let local = try self.requireModel(self.configuredModelID)
        try await self.runtime.load(local)
        let result = try await self.runtime.cleanUp(
            inputText,
            vocabulary: Self.vocabulary(),
            maxOutputTokens: maxOutputTokens,
            onText: streamHandler
        )
        let output: String
        switch LocalCleanupPrompt.check(output: result.text, transcript: inputText) {
        case let .accept(text):
            output = text
        case let .reject(reason):
            // Never deliver an answer or rewrite in place of what was said.
            Self.log("Local cleanup output rejected (\(reason)); using raw transcript")
            output = inputText
        }
        Self.log(
            "Local cleanup model=\(local.id) prompt=\(result.promptTokens) gen=\(result.generatedTokens) "
                + "prefill=\(result.prefillMilliseconds)ms total=\(result.totalMilliseconds)ms "
                + String(format: "%.0f tok/s", result.tokensPerSecond)
        )
        return PrivateAIIntegrationService.EnhancementResult(
            outputText: output,
            backendKind: "mlx-local",
            latencyMilliseconds: result.totalMilliseconds,
            tokensPerSecond: result.tokensPerSecond
        )
    }

    func rewrite(
        _ inputText: String,
        systemPrompt: String,
        runtime _: PrivateAIIntegrationService.RuntimeConfiguration,
        context _: PrivateAIIntegrationService.AppContext
    ) async throws -> PrivateAIIntegrationService.EnhancementResult {
        let local = try self.requireModel(self.configuredModelID)
        try await self.runtime.load(local)
        let budget = max(256, LocalCleanupPrompt.words(inputText).count * 3)
        let result = try await self.runtime.complete(
            messages: LocalCleanupPrompt.rewriteMessages(text: inputText, instructions: systemPrompt),
            maxOutputTokens: budget,
            onText: nil
        )
        return PrivateAIIntegrationService.EnhancementResult(
            outputText: LocalCleanupPrompt.strip(result.text),
            backendKind: "mlx-local",
            latencyMilliseconds: result.totalMilliseconds,
            tokensPerSecond: result.tokensPerSecond
        )
    }

    // MARK: - Helpers

    private func localModel(id: String) -> LocalCleanupModel? {
        LocalCleanupModelCatalog.model(id: id)
    }

    private func requireModel(_ id: String) throws -> LocalCleanupModel {
        guard let model = self.localModel(id: id) else { throw LocalCleanupError.unknownModel(id) }
        return model
    }

    /// The user's custom dictionary doubles as the cleanup vocabulary.
    private static func vocabulary() async -> [LocalCleanupPrompt.VocabularyTerm] {
        await MainActor.run { LocalCleanupPrompt.vocabulary(from: SettingsStore.shared.customDictionaryEntries) }
    }

    private static func log(_ message: String) {
        Task { @MainActor in DebugLogger.shared.info(message, source: "LocalCleanup") }
    }
}
