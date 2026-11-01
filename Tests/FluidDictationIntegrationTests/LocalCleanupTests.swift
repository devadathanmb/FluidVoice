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

    func testStripRemovesEchoedTagsAndThinking() {
        XCTAssertEqual(LocalCleanupPrompt.strip("<think>\n</think>\n<transcript>\nHello.\n</transcript>"), "Hello.")
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
        XCTAssertTrue(system.contains("Tailscale (may be heard as: tail scale)"))
        XCTAssertTrue(system.contains("BullMQ (may be heard as: bull mq, bool mq)"))
        XCTAssertEqual(LocalCleanupPrompt.system(vocabulary: []), LocalCleanupPrompt.systemPrompt)
    }

    func testMessagesEndWithWrappedTranscript() {
        let messages = LocalCleanupPrompt.messages(transcript: "hello", vocabulary: [])
        XCTAssertEqual(messages.first?["role"] as? String, "system")
        XCTAssertEqual(messages.last?["content"] as? String, "<transcript>\nhello\n</transcript>")
        XCTAssertEqual(messages.count, 2 + LocalCleanupPrompt.examples.count * 2)
    }

    func testBridgeRegistersLocalProvider() {
        XCTAssertTrue(PrivateAIProviderFeature.shared.isAvailable)
        XCTAssertEqual(PrivateAIProviderFeature.shared.providerID, "local-cleanup")
        XCTAssertEqual(PrivateAIModelRegistry.defaultModelID, LocalCleanupModelCatalog.defaultModel.id)
        XCTAssertNotNil(PrivateAIModelRegistry.model(id: LocalCleanupModelCatalog.qwen35_4B.id))
    }
}

/// Downloads the default model (about 1.6 GB) and runs real cleanups.
/// Local validation only: set FLUID_LOCAL_CLEANUP_E2E=1.
final class LocalCleanupRuntimeE2ETests: XCTestCase {
    func testDownloadSplitLoadAndCleanUp() async throws {
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
            let result = try await runtime.cleanUp(testCase.input, vocabulary: vocabulary, maxOutputTokens: 128, onText: nil)
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
            print("LocalCleanupE2E \(result.totalMilliseconds)ms \(Int(result.tokensPerSecond)) tok/s: \(output)")
        }
        await runtime.unload()
    }
}
