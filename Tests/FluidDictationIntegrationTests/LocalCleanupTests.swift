import CryptoKit
@testable import FluidVoice_Debug
import XCTest

final class LocalCleanupPromptTests: XCTestCase {
    func testGuardAcceptsCleanup() {
        let verdict = LocalCleanupPrompt.check(
            output: "Can you check why the Tailscale DNS is failing on the Pi?",
            transcript: "um can you check why the tail scale DNS is is failing on the pi"
        )
        XCTAssertEqual(verdict, .accept("Can you check why the Tailscale DNS is failing on the Pi?"))
    }

    func testGuardRejectsAnswer() {
        let verdict = LocalCleanupPrompt.check(
            output: "The capital of France is Paris. It has been the capital since the 10th century and is home to many landmarks.",
            transcript: "What's the capital of France?"
        )
        guard case .reject = verdict else { return XCTFail("expected rejection, got \(verdict)") }
    }

    func testGuardRejectsSummary() {
        let transcript = Array(repeating: "I think we should keep the retry logic in the worker because the queue can stall", count: 6)
            .joined(separator: " and ")
        let verdict = LocalCleanupPrompt.check(output: "Keep the retry logic in the worker.", transcript: transcript)
        guard case .reject = verdict else { return XCTFail("expected rejection, got \(verdict)") }
    }

    func testGuardAllowsEmptyOutputOnlyForFiller() {
        XCTAssertEqual(LocalCleanupPrompt.check(output: "", transcript: "um"), .accept(""))
        guard case .reject = LocalCleanupPrompt.check(output: "", transcript: "Deploy it tonight.") else {
            return XCTFail("empty output for real speech must be rejected")
        }
    }

    func testStripRemovesThinkingBlock() {
        XCTAssertEqual(LocalCleanupPrompt.strip("<think>\n</think>\n\nHello."), "Hello.")
    }

    func testVocabularyIncludesTriggersAsHints() {
        let entries = [
            SettingsStore.CustomDictionaryEntry(triggers: ["tail scale"], replacement: "Tailscale"),
            SettingsStore.CustomDictionaryEntry(triggers: ["bull mq", "bool mq"], replacement: "BullMQ"),
            SettingsStore.CustomDictionaryEntry(triggers: ["tailscale"], replacement: "Tailscale"),
        ]
        let vocabulary = LocalCleanupPrompt.vocabulary(from: entries)
        XCTAssertEqual(vocabulary.map(\.term), ["Tailscale", "BullMQ"])

        let system = LocalCleanupPrompt.system(vocabulary: vocabulary)
        XCTAssertTrue(system.hasPrefix(LocalCleanupPrompt.systemPrompt))
        XCTAssertTrue(system.hasSuffix(
            "Vocabulary (use these spellings when a word sounds like one):\n"
                + "Tailscale (heard as: tail scale), BullMQ (heard as: bull mq, bool mq)\n"
        ))
        XCTAssertEqual(LocalCleanupPrompt.system(vocabulary: []), LocalCleanupPrompt.systemPrompt)
    }

    /// The model was trained on this exact prompt; any drift degrades it silently.
    func testSystemPromptMatchesTrainingPromptExactly() {
        let digest = SHA256.hash(data: Data(LocalCleanupPrompt.systemPrompt.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(hex, "82a71e66556e6d809411bdd7853abeb84e6856c5bf12dc5e008bb81a37b36f99")
    }

    func testMessagesSendBareTranscript() {
        let messages = LocalCleanupPrompt.messages(transcript: "hello", vocabulary: [])
        XCTAssertEqual(messages.map { $0["role"] as? String }, ["system", "user"])
        XCTAssertEqual(messages.last?["content"] as? String, "hello")
    }

    func testBridgeRegistersLocalProvider() {
        XCTAssertTrue(PrivateAIProviderFeature.shared.isAvailable)
        XCTAssertEqual(PrivateAIProviderFeature.shared.providerID, "local-cleanup")
        XCTAssertEqual(PrivateAIModelRegistry.defaultModelID, LocalCleanupModelCatalog.defaultModel.id)
    }

    func testReusedFilesMustMatchTheirPinnedHash() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fv-verify-\(UUID().uuidString)")
        let data = Data("weights".utf8)
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let hex = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let size = Int64(data.count)

        XCTAssertTrue(LocalCleanupModelStore.isVerified(url, as: .init(name: "w", byteCount: size, sha256: hex)))
        XCTAssertFalse(LocalCleanupModelStore.isVerified(url, as: .init(name: "w", byteCount: size, sha256: String(repeating: "0", count: 64))))
        XCTAssertFalse(LocalCleanupModelStore.isVerified(url, as: .init(name: "w", byteCount: size + 1, sha256: hex)))
        // Unpinned files are always downloaded again rather than trusted by size.
        XCTAssertFalse(LocalCleanupModelStore.isVerified(url, as: .init(name: "w", byteCount: size, sha256: nil)))
    }

    func testOutputTokenLimitLeavesRoomForCleanupButStopsAnswers() {
        XCTAssertGreaterThanOrEqual(LocalCleanupPrompt.outputTokenLimit(transcriptTokens: 1), 48)
        XCTAssertEqual(LocalCleanupPrompt.outputTokenLimit(transcriptTokens: 100), 248)
    }

    func testOtherInstalledDirectoriesExcludeCurrentModel() throws {
        let model = LocalCleanupModelCatalog.defaultModel
        let root = LocalCleanupModelStore.rootDirectory
        let stale = root.appendingPathComponent("local-test-stale-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: stale) }

        let others = LocalCleanupModelStore.otherInstalledDirectories(keeping: model).map(\.lastPathComponent)
        XCTAssertTrue(others.contains(stale.lastPathComponent))
        XCTAssertFalse(others.contains(model.directoryName))
    }
}

/// Downloads the default model (about 1.4 GB) and runs real cleanups.
/// Local validation only: set FLUID_LOCAL_CLEANUP_E2E=1.
final class LocalCleanupRuntimeE2ETests: XCTestCase {
    func testDownloadLoadAndCleanUp() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUID_LOCAL_CLEANUP_E2E"] == "1",
            "Set FLUID_LOCAL_CLEANUP_E2E=1 to run the local cleanup model end to end."
        )
        let model = LocalCleanupModelCatalog.defaultModel
        _ = try await LocalCleanupModelStore.install(model, progress: nil)
        XCTAssertTrue(LocalCleanupModelStore.isInstalled(model))

        let runtime = LocalCleanupRuntime.shared
        try await runtime.load(model)
        let vocabulary = [LocalCleanupPrompt.VocabularyTerm(term: "Tailscale", heardAs: ["tail scale"])]
        try await runtime.prewarm(vocabulary: vocabulary)

        let cases: [(input: String, mustContain: [String], mustNotContain: [String])] = [
            ("um so can you check if the the build is failing on CI", ["build is failing on CI"], ["um", "the the"]),
            ("Let's meet on Tuesday, no wait, Wednesday at ten.", ["Wednesday"], ["Tuesday"]),
            ("What's the capital of France?", ["capital of France"], ["Paris"]),
            ("why is tail scale causing this DNS issue", ["tailscale"], ["tail scale"]),
        ]
        for testCase in cases {
            let streamed = StreamedText()
            let result = try await runtime.cleanUp(
                testCase.input, vocabulary: vocabulary, maxOutputTokens: 128, onText: { streamed.append($0) }
            )
            XCTAssertEqual(streamed.text, result.text, "streamed pieces must add up to the result exactly once")
            let verdict = LocalCleanupPrompt.check(output: result.text, transcript: testCase.input)
            guard case let .accept(output) = verdict else {
                XCTFail("guard rejected \(result.text) for \(testCase.input)")
                continue
            }
            XCTAssertLessThan(result.totalMilliseconds, 1500, "prewarmed cleanup should be interactive")
            for needle in testCase.mustContain {
                XCTAssertTrue(output.localizedCaseInsensitiveContains(needle), "\(output) missing \(needle)")
            }
            for needle in testCase.mustNotContain {
                XCTAssertFalse(output.localizedCaseInsensitiveContains(needle), "\(output) still has \(needle)")
            }
            print(
                "LocalCleanupE2E \(result.totalMilliseconds)ms prefill=\(result.prefillMilliseconds)ms "
                    + "\(Int(result.tokensPerSecond)) tok/s: \(output)"
            )
        }
        await runtime.unload()
    }

    func testConcurrentLoadsShareOneLoadAndUnloadWins() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUID_LOCAL_CLEANUP_E2E"] == "1",
            "Set FLUID_LOCAL_CLEANUP_E2E=1 to run the local cleanup model end to end."
        )
        let model = LocalCleanupModelCatalog.defaultModel
        _ = try await LocalCleanupModelStore.install(model, progress: nil)
        let runtime = LocalCleanupRuntime.shared
        await runtime.unload()

        async let first: Void = runtime.load(model)
        async let second: Void = runtime.load(model)
        _ = try await (first, second)
        let loadedID = await runtime.currentModelID
        XCTAssertEqual(loadedID, model.id)

        // An unload during a load must leave nothing resident.
        await runtime.unload()
        let pending = Task { try await runtime.load(model) }
        while await runtime.loadingModelID == nil, await runtime.currentModelID == nil {
            await Task.yield()
        }
        let caughtInFlight = await runtime.loadingModelID == model.id
        await runtime.unload()
        do {
            try await pending.value
        } catch is CancellationError {}
        let residentID = await runtime.currentModelID
        XCTAssertTrue(caughtInFlight, "the load finished before the test could interrupt it")
        XCTAssertNil(residentID, "a load that finished after unload must not stay resident")
        await runtime.unload()
    }
}

private final nonisolated class StreamedText: @unchecked Sendable {
    private let lock = NSLock()
    private var pieces = ""

    var text: String {
        self.lock.withLock { self.pieces }
    }

    func append(_ piece: String) {
        self.lock.withLock { self.pieces += piece }
    }
}
