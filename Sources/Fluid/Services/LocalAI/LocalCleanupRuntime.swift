import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

/// In-process MLX runtime for local dictation cleanup.
///
/// Holds one model at a time. The stable part of the prompt (instructions,
/// vocabulary) is prefilled once and its cache is copied per request,
/// so each dictation only pays for its own tokens.
actor LocalCleanupRuntime {
    static let shared = LocalCleanupRuntime()

    struct Result: Sendable {
        let text: String
        let promptTokens: Int
        let generatedTokens: Int
        let prefillMilliseconds: Int
        let totalMilliseconds: Int
        let tokensPerSecond: Double
    }

    private struct PrefixCache {
        /// The prefix depends only on the vocabulary; the instructions are constant.
        let vocabulary: [LocalCleanupPrompt.VocabularyTerm]
        let tokens: [Int]
        let cache: [KVCache]
    }

    /// Built by a detached load task and handed to the actor once; never shared after.
    private struct LoadedModel: @unchecked Sendable {
        let id: String
        let context: ModelContext
        let stopTokenIDs: Set<Int>
    }

    private struct PendingLoad {
        let id: String
        let generation: Int
        let task: Task<LoadedModel, Error>
    }

    private var loaded: LoadedModel?
    private var pendingLoad: PendingLoad?
    /// Bumped by `unload()` so a load that finishes afterwards is discarded instead
    /// of making the model resident again behind the caller's back.
    private var generation = 0
    private var prefix: PrefixCache?

    /// Bound MLX's reusable buffer pool so an idle model does not grow resident
    /// memory past its weights; dictation runs alongside other dev tools.
    private static let bufferCacheLimit = 256 << 20

    var currentModelID: String? {
        self.loaded?.id
    }

    var loadingModelID: String? {
        self.pendingLoad?.id
    }

    /// Load `model`, or join a load of it that is already running. The actor is
    /// reentrant across the weight load, so concurrent prewarm and dictation
    /// requests must share one load instead of reading the weights twice.
    func load(_ model: LocalCleanupModel) async throws {
        if self.loaded?.id == model.id {
            return
        }
        if let pending = self.pendingLoad, pending.id == model.id {
            try await self.finishLoad(pending)
            return
        }
        guard LocalCleanupModelStore.isInstalled(model) else {
            throw LocalCleanupError.modelNotInstalled(model.id)
        }
        self.unload()
        Memory.cacheLimit = Self.bufferCacheLimit

        let directory = LocalCleanupModelStore.directory(for: model)
        let pending = PendingLoad(
            id: model.id,
            generation: self.generation,
            task: Task.detached(priority: .userInitiated) {
                try await Self.loadModel(id: model.id, from: directory)
            }
        )
        self.pendingLoad = pending
        try await self.finishLoad(pending)
    }

    func unload() {
        self.generation += 1
        self.pendingLoad?.task.cancel()
        self.pendingLoad = nil
        self.loaded = nil
        self.prefix = nil
        Memory.clearCache()
    }

    /// Prefill the instruction prefix and compile decode kernels ahead of the
    /// first dictation, which otherwise pays over a second of Metal warmup.
    func prewarm(vocabulary: [LocalCleanupPrompt.VocabularyTerm]) throws {
        guard let loaded else { return }
        let prefix = try self.prefixCache(loaded.context, vocabulary: vocabulary)
        let tokens = try self.cleanupTokens(loaded.context, transcript: "ok", vocabulary: vocabulary)
        _ = try self.generate(loaded, tokens: tokens, prefix: prefix, maxTokens: 2, onText: nil)
    }

    /// `onText` receives each newly decoded piece, not the accumulated text.
    func cleanUp(
        _ transcript: String,
        vocabulary: [LocalCleanupPrompt.VocabularyTerm],
        maxOutputTokens: Int,
        onText: (@Sendable (String) -> Void)?
    ) throws -> Result {
        guard let loaded else { throw LocalCleanupError.modelNotInstalled("none loaded") }
        let prefix = try self.prefixCache(loaded.context, vocabulary: vocabulary)
        let tokens = try self.cleanupTokens(loaded.context, transcript: transcript, vocabulary: vocabulary)
        return try self.generate(
            loaded,
            tokens: tokens,
            prefix: prefix,
            maxTokens: maxOutputTokens,
            onText: onText
        )
    }

    func complete(
        messages: [[String: any Sendable]],
        maxOutputTokens: Int,
        onText: (@Sendable (String) -> Void)?
    ) throws -> Result {
        guard let loaded else { throw LocalCleanupError.modelNotInstalled("none loaded") }
        return try self.generate(
            loaded,
            tokens: self.render(loaded.context, messages),
            prefix: nil,
            maxTokens: maxOutputTokens,
            onText: onText
        )
    }

    // MARK: - Internals

    private func finishLoad(_ pending: PendingLoad) async throws {
        let model: LoadedModel
        do {
            model = try await pending.task.value
        } catch {
            if self.pendingLoad?.generation == pending.generation, self.pendingLoad?.id == pending.id {
                self.pendingLoad = nil
            }
            throw error
        }
        // `unload()` or a load of another model ran while the weights were loading.
        guard self.generation == pending.generation else { throw CancellationError() }
        if self.loaded?.id != model.id {
            self.loaded = model
            self.prefix = nil
        }
        self.pendingLoad = nil
    }

    private static func loadModel(id: String, from directory: URL) async throws -> LoadedModel {
        let context = try await MLXLMCommon.loadModel(from: directory, using: TransformersTokenizerLoader())
        var stops = Set<Int>()
        for token in ["<|im_end|>", "<|endoftext|>", "<end_of_turn>"] {
            if let id = context.tokenizer.convertTokenToId(token) {
                stops.insert(id)
            }
        }
        if let eos = context.tokenizer.eosTokenId {
            stops.insert(eos)
        }
        stops.formUnion(context.configuration.eosTokenIds)
        return LoadedModel(id: id, context: context, stopTokenIDs: stops)
    }

    private func render(_ context: ModelContext, _ messages: [[String: any Sendable]]) throws -> [Int] {
        try context.tokenizer.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: ["enable_thinking": false]
        )
    }

    private func cleanupTokens(
        _ context: ModelContext,
        transcript: String,
        vocabulary: [LocalCleanupPrompt.VocabularyTerm]
    ) throws -> [Int] {
        try self.render(context, LocalCleanupPrompt.messages(transcript: transcript, vocabulary: vocabulary))
    }

    /// The prefix is whatever two renders with different transcripts share,
    /// minus one token so a boundary merge can never land inside the cache.
    /// Rebuilt only when the vocabulary changes.
    private func prefixCache(
        _ context: ModelContext,
        vocabulary: [LocalCleanupPrompt.VocabularyTerm]
    ) throws -> PrefixCache? {
        if let prefix, prefix.vocabulary == vocabulary {
            return prefix
        }
        self.prefix = nil
        let a = try self.cleanupTokens(context, transcript: "AAAA", vocabulary: vocabulary)
        let b = try self.cleanupTokens(context, transcript: "BBBB", vocabulary: vocabulary)
        var shared = 0
        while shared < min(a.count, b.count), a[shared] == b[shared] {
            shared += 1
        }
        let tokens = Array(a.prefix(max(0, shared - 1)))
        guard tokens.count >= 16 else { return nil }

        let cache = try context.model.newCache(parameters: nil)
        let input = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
        _ = context.model(input, cache: cache)
        eval(cache)
        let built = PrefixCache(vocabulary: vocabulary, tokens: tokens, cache: cache)
        self.prefix = built
        return built
    }

    private func generate(
        _ loaded: LoadedModel,
        tokens: [Int],
        prefix: PrefixCache?,
        maxTokens: Int,
        onText: (@Sendable (String) -> Void)?
    ) throws -> Result {
        let start = Date()
        var cache: [KVCache] = []
        var feed = tokens
        if let prefix, tokens.count > prefix.tokens.count, tokens.starts(with: prefix.tokens) {
            cache = prefix.cache.map { $0.copy() }
            feed = Array(tokens.dropFirst(prefix.tokens.count))
        }

        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(feed.map(Int32.init))),
            model: loaded.context.model,
            cache: cache.isEmpty ? nil : cache,
            processor: nil,
            sampler: ArgMaxSampler(),
            maxTokens: max(1, maxTokens)
        )
        let prefillDone = Date()

        var detokenizer = onText.map { _ in NaiveStreamingDetokenizer(tokenizer: loaded.context.tokenizer) }
        var generatedTokens: [Int] = []
        while let token = iterator.next() {
            if loaded.stopTokenIDs.contains(token) {
                break
            }
            try Task.checkCancellation()
            generatedTokens.append(token)
            if let onText {
                detokenizer?.append(token: token)
                if let piece = detokenizer?.next(), !piece.isEmpty {
                    onText(piece)
                }
            }
        }
        let end = Date()
        let decodeSeconds = end.timeIntervalSince(prefillDone)
        // Decode once at the end so the result never depends on streaming segmentation.
        let text = loaded.context.tokenizer.decode(tokenIds: generatedTokens, skipSpecialTokens: true)
        return Result(
            text: text,
            promptTokens: tokens.count,
            generatedTokens: generatedTokens.count,
            prefillMilliseconds: Int(prefillDone.timeIntervalSince(start) * 1000),
            totalMilliseconds: Int(end.timeIntervalSince(start) * 1000),
            tokensPerSecond: decodeSeconds > 0 ? Double(generatedTokens.count) / decodeSeconds : 0
        )
    }
}

/// Loads `tokenizer.json` with swift-transformers and adapts it to MLXLMCommon.
private nonisolated struct TransformersTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return TransformersTokenizerBridge(upstream: upstream)
    }
}

private nonisolated struct TransformersTokenizerBridge: MLXLMCommon.Tokenizer {
    let upstream: any Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        self.upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        self.upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        self.upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        self.upstream.convertIdToToken(id)
    }

    var bosToken: String? {
        self.upstream.bosToken
    }

    var eosToken: String? {
        self.upstream.eosToken
    }

    var unknownToken: String? {
        self.upstream.unknownToken
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?, // swiftlint:disable:this discouraged_optional_collection
        additionalContext: [String: any Sendable]? // swiftlint:disable:this discouraged_optional_collection
    ) throws -> [Int] {
        do {
            return try self.upstream.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext
            )
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
