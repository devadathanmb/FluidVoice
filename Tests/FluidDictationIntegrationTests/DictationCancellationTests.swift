import AppKit
import Carbon
@testable import FluidVoice_Debug
import Foundation
import XCTest

@MainActor
private final class DelayedAIReply {
    var continuation: CheckedContinuation<String, Never>?
    var finished = false

    func wait(entered: XCTestExpectation) async -> String {
        let value = await withCheckedContinuation { continuation in
            self.continuation = continuation
            entered.fulfill()
        }
        self.finished = true
        return value
    }

    func release(_ text: String = "late AI result") {
        self.continuation?.resume(returning: text)
        self.continuation = nil
    }
}

private nonisolated class PendingDictationURLProtocol: URLProtocol, @unchecked Sendable {
    private struct Hooks {
        let stopped: XCTestExpectation
    }

    private static let registryLock = NSLock()
    private nonisolated(unsafe) static var hooks: [String: Hooks] = [:]
    private let lock = NSLock()
    private var stopExpectation: XCTestExpectation?

    static func register(host: String, stopped: XCTestExpectation) {
        self.registryLock.withLock { self.hooks[host] = Hooks(stopped: stopped) }
    }

    static func unregister(host: String) {
        _ = self.registryLock.withLock { self.hooks.removeValue(forKey: host) }
    }

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host?.hasSuffix(".dictation-cancel.test") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let url = self.request.url,
              let hooks = Self.registryLock.withLock({ Self.hooks[url.host ?? ""] }),
              let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: ["Content-Type": "text/event-stream"])
        else { return }
        self.lock.withLock { self.stopExpectation = hooks.stopped }
        self.client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let json = url.path.contains("/responses")
            ? #"{"type":"response.output_text.delta","delta":"first token"}"#
            : #"{"choices":[{"delta":{"content":"First streamed token followed by enough text to exceed the thinking-tag lookahead buffer."}}]}"#
        self.client?.urlProtocol(self, didLoad: Data("data: \(json)\n\n".utf8))
        // Keep the response open until URLSession cancels its data task.
    }

    override func stopLoading() {
        let stopped = self.lock.withLock {
            let expectation = self.stopExpectation
            self.stopExpectation = nil
            return expectation
        }
        stopped?.fulfill()
    }
}

@MainActor
final class DictationCancellationTests: XCTestCase {
    private func assertStreamingTransportCancels(basePath: String) async {
        let host = UUID().uuidString.lowercased() + ".dictation-cancel.test"
        let tokenReceived = expectation(description: "Provider streamed its first token")
        let transportStopped = expectation(description: "URLSession cancelled the provider transport")
        let callerCancelled = expectation(description: "Foreground AI waiter cancelled")
        PendingDictationURLProtocol.register(host: host, stopped: transportStopped)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [PendingDictationURLProtocol.self]
        let session = URLSession(configuration: configuration)
        defer {
            session.invalidateAndCancel()
            PendingDictationURLProtocol.unregister(host: host)
        }
        let client = LLMClient(session: session)
        var config = LLMClient.Config(
            messages: [["role": "user", "content": "fixture"]],
            model: "fixture",
            baseURL: "https://\(host)\(basePath)",
            apiKey: "",
            streaming: true
        )
        config.maxRetries = 1
        config.timeoutSeconds = 10
        config.onContentChunk = { _ in tokenReceived.fulfill() }
        let request = CancellableDictationRequest()
        let capturedConfig = config
        let caller = Task { @MainActor in
            do {
                _ = try await request.value { try await client.call(capturedConfig).content }
                XCTFail("Cancelled streaming output must not be delivered")
            } catch {
                XCTAssertTrue(error is CancellationError)
                callerCancelled.fulfill()
            }
        }
        await fulfillment(of: [tokenReceived], timeout: 3)
        request.cancel()
        await fulfillment(of: [callerCancelled, transportStopped], timeout: 2)
        await caller.value
    }

    func testCancellingChatStreamStopsTheURLSessionDataTask() async {
        await self.assertStreamingTransportCancels(basePath: "/v1")
    }

    func testCancellingResponsesStreamStopsTheURLSessionDataTask() async {
        await self.assertStreamingTransportCancels(basePath: "/v1/responses")
    }

    func testCancellationReturnsBeforeAnUncooperativeProviderFinishes() async {
        let request = CancellableDictationRequest()
        let backend = DelayedAIReply()
        let entered = expectation(description: "AI request started")
        let cancelled = expectation(description: "UI waiter cancelled")
        var delivered = false
        let caller = Task { @MainActor in
            do {
                _ = try await request.value { await backend.wait(entered: entered) }
                delivered = true
            } catch {
                XCTAssertTrue(error is CancellationError)
                cancelled.fulfill()
            }
        }
        await fulfillment(of: [entered], timeout: 2)
        request.cancel()
        request.cancel()
        await fulfillment(of: [cancelled], timeout: 2)
        XCTAssertFalse(backend.finished)
        XCTAssertFalse(delivered)
        backend.release()
        await caller.value
        for _ in 0..<10 {
            await Task.yield()
        }
        XCTAssertTrue(backend.finished)
        XCTAssertFalse(delivered)
    }

    func testCancellingTheCallerCancelsTheRequestWithoutWaitingForTheProvider() async {
        let request = CancellableDictationRequest()
        let backend = DelayedAIReply()
        let entered = expectation(description: "AI request started")
        let cancelled = expectation(description: "Caller cancellation propagated")
        let caller = Task { @MainActor in
            do {
                _ = try await request.value { await backend.wait(entered: entered) }
                XCTFail("Cancelled AI result must not be delivered")
            } catch {
                XCTAssertTrue(error is CancellationError)
                cancelled.fulfill()
            }
        }
        await fulfillment(of: [entered], timeout: 2)
        caller.cancel()
        await fulfillment(of: [cancelled], timeout: 2)
        XCTAssertTrue(request.isCancelled)
        XCTAssertFalse(backend.finished)
        backend.release()
        await caller.value
    }

    func testCancellationBeforeStartDoesNotCallTheProvider() async {
        let request = CancellableDictationRequest()
        request.cancel()
        var calledProvider = false
        do {
            _ = try await request.value { calledProvider = true; return "unexpected" }
            XCTFail("A cancelled request must throw")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertFalse(calledProvider)
    }

    func testNormalCompletionStillDeliversTheProviderResult() async throws {
        let request = CancellableDictationRequest()
        let value = try await request.value { "completed AI result" }
        XCTAssertEqual(value, "completed AI result")
        XCTAssertFalse(request.isCancelled)
        request.cancel()
        XCTAssertFalse(request.isCancelled)
    }

    func testLateCompletionFromACancelledRequestCannotFinishANewerRequest() async {
        let old = CancellableDictationRequest()
        let new = CancellableDictationRequest()
        let oldBackend = DelayedAIReply()
        let newBackend = DelayedAIReply()
        let oldEntered = expectation(description: "Old AI request entered")
        let newEntered = expectation(description: "New AI request entered")
        let oldCancelled = expectation(description: "Old caller released")
        let newCompleted = expectation(description: "New request completed")
        var newOutput: String?
        let oldCaller = Task { @MainActor in
            do {
                _ = try await old.value { await oldBackend.wait(entered: oldEntered) }
            } catch {
                oldCancelled.fulfill()
            }
        }
        await fulfillment(of: [oldEntered], timeout: 2)
        old.cancel()
        await fulfillment(of: [oldCancelled], timeout: 2)
        let newCaller = Task { @MainActor in
            newOutput = try? await new.value { await newBackend.wait(entered: newEntered) }
            newCompleted.fulfill()
        }
        await fulfillment(of: [newEntered], timeout: 2)
        oldBackend.release("obsolete answer")
        for _ in 0..<10 {
            await Task.yield()
        }
        XCTAssertNil(newOutput)
        XCTAssertFalse(new.isCancelled)
        newBackend.release("new answer")
        await fulfillment(of: [newCompleted], timeout: 2)
        await oldCaller.value
        await newCaller.value
        XCTAssertEqual(newOutput, "new answer")
    }

    func testInvalidatedStreamPreviewRejectsLateTokensAndQueuedUpdates() async {
        var published: [String] = []
        let preview = DictationAIStreamPreviewBuffer(minimumUpdateInterval: 0, initialUpdateDelay: 0) {
            published.append($0)
        }
        preview.append("queued old answer")
        preview.invalidate()
        preview.append(" late token")
        preview.flush()
        for _ in 0..<10 {
            await Task.yield()
        }
        XCTAssertTrue(published.isEmpty)
    }

    func testEscapeReleasesTheHotkeyStopLockBeforeTheOldAIReplyFinishes() async throws {
        let asr = ASRService()
        defer { asr.isRunning = false }
        let request = CancellableDictationRequest()
        let backend = DelayedAIReply()
        let entered = expectation(description: "Post-processing is waiting for AI")
        let stopFinished = expectation(description: "Cancelled stop pipeline finished")
        let restarted = expectation(description: "New recording started")
        var starts = 0
        let shortcut = HotkeyShortcut(keyCode: 49, modifierFlags: [.control, .option])
        let manager = GlobalHotkeyManager(
            asrService: asr,
            primaryShortcuts: [shortcut],
            promptModeShortcut: shortcut,
            commandModeShortcut: nil,
            rewriteModeShortcut: shortcut,
            promptModeShortcutEnabled: false,
            commandModeShortcutEnabled: false,
            rewriteModeShortcutEnabled: false,
            dictationModeCallback: {
                starts += 1
                asr.isRunning = true
                if starts == 2 {
                    restarted.fulfill()
                }
            },
            stopAndProcessCallback: { _ in
                asr.isRunning = false
                do {
                    _ = try await request.value { await backend.wait(entered: entered) }
                } catch {
                    XCTAssertTrue(error is CancellationError)
                }
                stopFinished.fulfill()
            },
            isDictateRecordingProvider: { true },
            isSessionLockedProvider: { false },
            automaticallyInitialize: false
        )
        manager.setHotkeyMode(.toggle)
        manager.setCancelCallback { request.cancel(); return .cancelled }
        let previousCancel = SettingsStore.shared.cancelRecordingHotkeyShortcut
        SettingsStore.shared.cancelRecordingHotkeyShortcut = HotkeyShortcut(keyCode: 53, modifierFlags: [])
        defer { SettingsStore.shared.cancelRecordingHotkeyShortcut = previousCancel }
        func chord() throws {
            for down in [true, false] {
                let event = try XCTUnwrap(CGEvent(keyboardEventSource: nil, virtualKey: 49, keyDown: down))
                event.flags = [.maskControl, .maskAlternate]
                _ = manager.handleKeyEvent(type: down ? .keyDown : .keyUp, event: event)
            }
        }
        try chord()
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(starts, 1)
        try chord()
        await fulfillment(of: [entered], timeout: 2)
        let escape = try XCTUnwrap(CGEvent(keyboardEventSource: nil, virtualKey: 53, keyDown: true))
        escape.flags = []
        _ = manager.handleKeyEvent(type: .keyDown, event: escape)
        await fulfillment(of: [stopFinished], timeout: 2)
        for _ in 0..<10 {
            await Task.yield()
        }
        XCTAssertFalse(backend.finished)
        try chord()
        await fulfillment(of: [restarted], timeout: 2)
        XCTAssertEqual(starts, 2)
        backend.release()
    }

    func testCarbonEscapeStaysRegisteredThroughPostProcessingAndUnregistersWhenIdle() async throws {
        let previousCancel = SettingsStore.shared.cancelRecordingHotkeyShortcut
        SettingsStore.shared.cancelRecordingHotkeyShortcut = HotkeyShortcut(keyCode: 53, modifierFlags: [])
        defer { SettingsStore.shared.cancelRecordingHotkeyShortcut = previousCancel }
        let asr = ASRService()
        let chord = HotkeyShortcut(keyCode: 25, modifierFlags: [.command, .control, .option, .shift])
        let manager = GlobalHotkeyManager(
            asrService: asr,
            primaryShortcuts: [chord],
            promptModeShortcut: chord,
            commandModeShortcut: nil,
            rewriteModeShortcut: chord,
            promptModeShortcutEnabled: false,
            commandModeShortcutEnabled: false,
            rewriteModeShortcutEnabled: false,
            isSessionLockedProvider: { false },
            isAccessibilityTrustedProvider: { false },
            isCarbonShortcutPressedProvider: { _ in true }
        )
        defer {
            asr.isRunning = false
            manager.setPostProcessingCancellationActive(false)
        }
        let registered = expectation(description: "Real Carbon fallback registered")
        var didRegister = false
        manager.setAvailabilityDidChange { availability in
            if availability.arePrimaryShortcutsActive, !didRegister {
                didRegister = true
                registered.fulfill()
            }
        }
        var cancels = 0
        manager.setCancelCallback { cancels += 1; return .cancelled }
        await fulfillment(of: [registered], timeout: 5)

        var event: EventRef?
        XCTAssertEqual(CreateEvent(
            nil,
            OSType(kEventClassKeyboard),
            UInt32(kEventHotKeyPressed),
            ProcessInfo.processInfo.systemUptime,
            0,
            &event
        ), noErr)
        let nativeEvent = try XCTUnwrap(event)
        defer { ReleaseEvent(nativeEvent) }
        var id = EventHotKeyID(signature: manager.carbonEventSignature, id: CarbonHotKeyAction.cancel.id)
        XCTAssertEqual(SetEventParameter(
            nativeEvent,
            EventParamName(kEventParamDirectObject),
            EventParamType(typeEventHotKeyID),
            MemoryLayout<EventHotKeyID>.size,
            &id
        ), noErr)
        XCTAssertEqual(manager.handleNativeCarbonHotKeyEvent(nativeEvent), OSStatus(eventNotHandledErr))
        asr.isRunning = true
        manager.setPostProcessingCancellationActive(true)
        asr.isRunning = false
        XCTAssertEqual(manager.handleNativeCarbonHotKeyEvent(nativeEvent), noErr)
        XCTAssertEqual(cancels, 1)
        manager.setPostProcessingCancellationActive(false)
        XCTAssertEqual(manager.handleNativeCarbonHotKeyEvent(nativeEvent), OSStatus(eventNotHandledErr))
        XCTAssertEqual(cancels, 1)
    }
}
