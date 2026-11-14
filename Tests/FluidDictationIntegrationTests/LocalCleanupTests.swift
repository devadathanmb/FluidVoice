import CryptoKit
@testable import FluidVoice_Debug
import MLX
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
            "\n\nVocabulary (use these spellings when a word sounds like one):\n"
                + "Tailscale (heard as: tail scale), BullMQ (heard as: bull mq, bool mq)"
        ))
        XCTAssertEqual(LocalCleanupPrompt.system(vocabulary: []), LocalCleanupPrompt.systemPrompt)
    }

    /// The model was trained on this exact prompt; any drift degrades it silently.
    func testSystemPromptMatchesTrainingPromptExactly() {
        let digest = SHA256.hash(data: Data(LocalCleanupPrompt.systemPrompt.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(hex, "a5367cc4a61c000fae37f9c7c572412080d101cc02262f08a7289b81fe162d74")
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

final class LocalCleanupLookupTableDecoderTests: XCTestCase {
    /// Packs indices the way FluidDecode does: one LSB-first bitstream per row.
    private static func pack(_ rows: [[Int]], bits: Int) -> [UInt32] {
        let words = rows[0].count * bits / 32
        var packed = [UInt32](repeating: 0, count: rows.count * words)
        for (row, indices) in rows.enumerated() {
            for (column, index) in indices.enumerated() {
                for bit in 0..<bits where index >> bit & 1 == 1 {
                    let position = column * bits + bit
                    packed[row * words + position / 32] |= 1 << UInt32(position % 32)
                }
            }
        }
        return packed
    }

    private func assertRoundTrip(bits: Int, rows: Int, columns: Int, rowsPerTable: Int) throws {
        let entries = 1 << bits
        let indices = (0..<rows).map { row in (0..<columns).map { column in (row * 7 + column * 5 + column / 3) % entries } }
        let tables = (0..<(rows / rowsPerTable)).map { group in (0..<entries).map { Float(group * 100 + $0) } }
        let packed = MLXArray(Self.pack(indices, bits: bits), [rows, columns * bits / 32])
        let table = MLXArray(tables.flatMap(\.self), [tables.count, entries])

        let decoded = try LocalCleanupLookupTableDecoder.decode(packed: packed, table: table, rows: 0..<rows)
        let expected = (0..<rows).flatMap { row in indices[row].map { tables[row / rowsPerTable][$0] } }
        XCTAssertEqual(decoded.shape, [rows, columns])
        XCTAssertEqual(decoded.asArray(Float.self), expected)

        // A chunk aligned to table groups decodes the same rows.
        guard rowsPerTable < rows else { return }
        let tail = try LocalCleanupLookupTableDecoder.decode(packed: packed, table: table, rows: rowsPerTable..<rows)
        XCTAssertEqual(tail.asArray(Float.self), Array(expected[(rowsPerTable * columns)...]))
    }

    func testDecodesThreeBitIndicesThatStraddleWords() throws {
        try self.assertRoundTrip(bits: 3, rows: 4, columns: 64, rowsPerTable: 2)
    }

    func testDecodesTwoFourAndSixBitIndices() throws {
        try self.assertRoundTrip(bits: 2, rows: 4, columns: 32, rowsPerTable: 2)
        try self.assertRoundTrip(bits: 4, rows: 4, columns: 16, rowsPerTable: 4)
        try self.assertRoundTrip(bits: 6, rows: 2, columns: 32, rowsPerTable: 1)
    }

    func testRejectsChunksThatSplitATableGroup() throws {
        let packed = MLXArray([UInt32](repeating: 0, count: 4), [4, 1])
        let table = MLXArray([Float](repeating: 0, count: 32), [2, 16])
        XCTAssertThrowsError(try LocalCleanupLookupTableDecoder.decode(packed: packed, table: table, rows: 1..<4))
        XCTAssertThrowsError(try LocalCleanupLookupTableDecoder.decode(packed: packed, table: table, rows: 4..<4))
    }

    func testConfigReplacesLookupTableQuantization() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("fv-lut-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("in.json")
        let destination = directory.appendingPathComponent("out.json")
        try Data(#"{"model_type":"qwen3_5","quantization":{"mode":"lut","bits":{"3":1}},"quantization_config":{"mode":"lut"}}"#.utf8)
            .write(to: source)

        try LocalCleanupLookupTableDecoder.writeConfig(
            from: source, to: destination, quantization: .init(groupSize: 64, bits: 8)
        )
        let config = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: destination)) as? [String: Any])
        XCTAssertEqual(config["model_type"] as? String, "qwen3_5")
        for key in ["quantization", "quantization_config"] {
            let block = try XCTUnwrap(config[key] as? [String: Int])
            XCTAssertEqual(block, ["group_size": 64, "bits": 8])
        }
    }

    func testSplashIsReadFromFluidVoiceAppWithEveryFilePinned() {
        let splash = LocalCleanupModelCatalog.fluid11Splash
        guard case .fluidVoiceApp = splash.source else { return XCTFail("Splash must come from FluidVoice.app") }
        XCTAssertNil(splash.remoteURL(for: splash.files[0]), "the fork never downloads Splash itself")
        XCTAssertTrue(splash.files.allSatisfy { $0.sha256?.count == 64 })
        XCTAssertEqual(splash.conversion, .decodeLookupTables)
    }
}

/// Downloads the default model (about 1.6 GB) and runs real cleanups.
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
            // ASRService applies dictionary triggers before cleanup; cleanup must keep the term.
            ("why is Tailscale causing this DNS issue", ["Tailscale"], ["tail scale"]),
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

    /// The weights must report byte progress while downloading, not only per file.
    func testWeightsDownloadReportsProgressAndCancels() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUID_LOCAL_CLEANUP_E2E"] == "1",
            "Set FLUID_LOCAL_CLEANUP_E2E=1 to run the local cleanup model end to end."
        )
        let weights = try XCTUnwrap(LocalCleanupModelCatalog.fluid1Mini.files.first { $0.name == "model.safetensors" })
        let probe = Self.probeModel(files: [weights])
        defer { try? LocalCleanupModelStore.remove(probe) }

        let (progress, report) = AsyncStream.makeStream(of: Int64.self)
        let install = Task {
            try await LocalCleanupModelStore.install(probe) { written, _ in report.yield(written) }
        }
        let firstBytes = Task {
            for await written in progress where written > 0 {
                return written
            }
            return Int64(0)
        }
        let timeout = Task {
            try await Task.sleep(for: .seconds(30))
            report.finish()
        }
        let written = await firstBytes.value
        timeout.cancel()
        install.cancel()
        XCTAssertGreaterThan(written, 0, "no byte progress within 30 s")
        XCTAssertLessThan(written, weights.byteCount)
        do {
            _ = try await install.value
            XCTFail("a cancelled download must not install")
        } catch {}
        XCTAssertFalse(LocalCleanupModelStore.isInstalled(probe))
    }

    /// A full download moves the file into place, verifies its hash, and marks the model ready.
    func testSmallPinnedFileDownloadInstalls() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUID_LOCAL_CLEANUP_E2E"] == "1",
            "Set FLUID_LOCAL_CLEANUP_E2E=1 to run the local cleanup model end to end."
        )
        let tokenizer = try XCTUnwrap(LocalCleanupModelCatalog.fluid1Mini.files.first { $0.name == "tokenizer.json" })
        let probe = Self.probeModel(files: [tokenizer])
        defer { try? LocalCleanupModelStore.remove(probe) }

        let directory = try await LocalCleanupModelStore.install(probe, progress: nil)
        XCTAssertTrue(LocalCleanupModelStore.isInstalled(probe))
        XCTAssertTrue(LocalCleanupModelStore.isVerified(directory.appendingPathComponent(tokenizer.name), as: tokenizer))
    }

    private static func probeModel(files: [LocalCleanupModel.File]) -> LocalCleanupModel {
        let base = LocalCleanupModelCatalog.fluid1Mini
        return LocalCleanupModel(
            id: "local-test-download-\(UUID().uuidString.lowercased())",
            displayName: "probe",
            detail: "",
            parameterCount: "",
            recommendedMemoryGB: 0,
            source: base.source,
            revision: base.revision,
            files: files,
            conversion: .none
        )
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
