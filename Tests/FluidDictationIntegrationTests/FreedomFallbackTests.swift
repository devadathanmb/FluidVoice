import AppKit
import Carbon
@testable import FluidVoice_Debug
import XCTest

private nonisolated func ownershipTestCarbonHandler(
    _: EventHandlerCallRef?, event: EventRef?, userData: UnsafeMutableRawPointer?
) -> OSStatus {
    guard let event, let userData else { return OSStatus(eventNotHandledErr) }
    let manager = Unmanaged<GlobalHotkeyManager>.fromOpaque(userData).takeUnretainedValue()
    return MainActor.assumeIsolated { manager.handleNativeCarbonHotKeyEvent(event) }
}

final class DictationTextDeliveryPolicyTests: XCTestCase {
    func testNoAccessibilityUsesClipboardInsteadOfExternalInsertion() {
        for frontmost in [false, true] {
            for copyPreference in [false, true] {
                let policy = DictationTextDeliveryPolicy(
                    deliversText: true,
                    accessibilityTrusted: false,
                    isFluidFrontmost: frontmost,
                    copyPreference: copyPreference
                )
                XCTAssertTrue(policy.shouldCopyToClipboard)
                XCTAssertFalse(policy.shouldTypeExternally)
            }
        }
    }

    func testSandboxAndHistoryOnlyNeverCopyOrInsertWithoutAccessibility() {
        let policy = DictationTextDeliveryPolicy(
            deliversText: false,
            accessibilityTrusted: false,
            isFluidFrontmost: false,
            copyPreference: true,
            aiFailed: true
        )
        XCTAssertFalse(policy.shouldCopyToClipboard)
        XCTAssertFalse(policy.shouldTypeExternally)
    }

    func testTrustedAccessibilityPreservesInsertionAndClipboardPreference() {
        let external = DictationTextDeliveryPolicy(
            deliversText: true,
            accessibilityTrusted: true,
            isFluidFrontmost: false,
            copyPreference: false
        )
        XCTAssertTrue(external.shouldTypeExternally)
        XCTAssertFalse(external.shouldCopyToClipboard)

        let internalEditor = DictationTextDeliveryPolicy(
            deliversText: true,
            accessibilityTrusted: true,
            isFluidFrontmost: true,
            copyPreference: true
        )
        XCTAssertFalse(internalEditor.shouldTypeExternally)
        XCTAssertTrue(internalEditor.shouldCopyToClipboard)
    }

    func testFailedAICleanupStillCopiesRawFallback() {
        let policy = DictationTextDeliveryPolicy(
            deliversText: true,
            accessibilityTrusted: true,
            isFluidFrontmost: false,
            copyPreference: false,
            aiFailed: true
        )
        XCTAssertTrue(policy.shouldCopyToClipboard)
        XCTAssertTrue(policy.shouldTypeExternally)
    }

    func testSendingExistingDraftDoesNotOverwriteClipboardWithEmptyText() {
        let policy = DictationTextDeliveryPolicy(
            deliversText: true,
            accessibilityTrusted: true,
            isFluidFrontmost: false,
            sendsExistingDraft: true,
            copyPreference: true
        )
        XCTAssertFalse(policy.shouldCopyToClipboard)
        XCTAssertTrue(policy.shouldTypeExternally)
    }
}

@MainActor
final class FocusedCarbonShortcutTests: XCTestCase {
    private let shortcut = HotkeyShortcut(keyCode: 49, modifierFlags: [.control, .option])

    private func makeManager(
        asr: ASRService,
        trusted: Bool = false,
        keyPressed: @escaping () -> Bool,
        start: @escaping () async -> Void,
        stop: @escaping () async -> Void = {}
    ) -> GlobalHotkeyManager {
        let manager = GlobalHotkeyManager(
            asrService: asr,
            primaryShortcuts: [self.shortcut],
            promptModeShortcut: self.shortcut,
            commandModeShortcut: nil,
            rewriteModeShortcut: self.shortcut,
            promptModeShortcutEnabled: false,
            commandModeShortcutEnabled: false,
            rewriteModeShortcutEnabled: false,
            startRecordingCallback: start,
            dictationModeCallback: start,
            stopAndProcessCallback: { _ in await stop() },
            isDictateRecordingProvider: { true },
            isSessionLockedProvider: { false },
            isAccessibilityTrustedProvider: { trusted },
            isCarbonShortcutPressedProvider: { _ in keyPressed() },
            automaticallyInitialize: false
        )
        manager.trackCarbonRegistration(.init(action: .primary(0), shortcut: self.shortcut, label: "Primary dictation"))
        manager.setHotkeyMode(.toggle)
        return manager
    }

    func testFocusedAndCarbonEventsTriggerOnlyOncePerPress() async {
        let asr = ASRService()
        var starts = 0
        let manager = self.makeManager(asr: asr, keyPressed: { true }, start: { starts += 1 })
        XCTAssertTrue(manager.handleFocusedAppPrimaryShortcut(keyCode: 49, modifiers: [.control, .option], isPressed: true))
        manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: true)
        XCTAssertTrue(manager.handleFocusedAppPrimaryShortcut(keyCode: 49, modifiers: [], isPressed: false))
        manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: false)
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(starts, 1)
    }

    private func withNativeHotKeyEvent(
        signature: OSType,
        id: UInt32 = CarbonHotKeyAction.primary(0).id,
        isPressed: Bool,
        body: (EventRef) -> Void
    ) throws {
        var event: EventRef?
        XCTAssertEqual(CreateEvent(
            nil,
            OSType(kEventClassKeyboard),
            UInt32(isPressed ? kEventHotKeyPressed : kEventHotKeyReleased),
            ProcessInfo.processInfo.systemUptime,
            0,
            &event
        ), noErr)
        let nativeEvent = try XCTUnwrap(event)
        defer { ReleaseEvent(nativeEvent) }
        var hotKeyID = EventHotKeyID(signature: signature, id: id)
        XCTAssertEqual(SetEventParameter(
            nativeEvent,
            EventParamName(kEventParamDirectObject),
            EventParamType(typeEventHotKeyID),
            MemoryLayout<EventHotKeyID>.size,
            &hotKeyID
        ), noErr)
        body(nativeEvent)
    }

    func testNativeCarbonEventBridgeUsesTheSamePressTracker() async throws {
        var starts = 0
        let manager = self.makeManager(asr: ASRService(), keyPressed: { true }, start: { starts += 1 })
        try self.withNativeHotKeyEvent(signature: manager.carbonEventSignature, isPressed: true) { event in
            XCTAssertTrue(manager.handleAppKitCarbonEvent(event))
            manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: true, timestamp: GetEventTime(event))
        }
        try self.withNativeHotKeyEvent(signature: manager.carbonEventSignature, isPressed: false) { event in
            XCTAssertTrue(manager.handleAppKitCarbonEvent(event))
            manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: false, timestamp: GetEventTime(event))
        }
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(starts, 1)
    }

    func testNativeBridgeLeavesOtherCarbonOwnersAndUnknownRegistrationsAlone() throws {
        let manager = self.makeManager(asr: ASRService(), keyPressed: { true }, start: {})
        try self.withNativeHotKeyEvent(signature: 0x50524f42, isPressed: true) { event in
            XCTAssertFalse(manager.handleAppKitCarbonEvent(event))
        }
        try self.withNativeHotKeyEvent(signature: manager.carbonEventSignature, id: CarbonHotKeyAction.primary(99).id, isPressed: true) { event in
            XCTAssertFalse(manager.handleAppKitCarbonEvent(event))
        }
    }

    func testTrustedAccessibilityDoesNotAlsoHandleAppKitCarbonEvents() throws {
        let manager = self.makeManager(asr: ASRService(), trusted: true, keyPressed: { true }, start: {})
        try self.withNativeHotKeyEvent(signature: manager.carbonEventSignature, isPressed: true) { event in
            XCTAssertFalse(manager.handleAppKitCarbonEvent(event))
        }
    }

    func testNativeBridgeDoesNotConsumeUnrelatedSystemDefinedEvents() throws {
        let manager = self.makeManager(asr: ASRService(), keyPressed: { true }, start: {})
        let event = try XCTUnwrap(NSEvent.otherEvent(
            with: .systemDefined,
            location: .zero,
            modifierFlags: [],
            timestamp: ProcessInfo.processInfo.systemUptime,
            windowNumber: 0,
            context: nil,
            subtype: 8,
            data1: 0,
            data2: 0
        ))
        XCTAssertFalse(manager.handleAppKitCarbonHotKey(event))
    }

    func testFocusedMonitorPassesThroughSystemDefinedEventsWithoutKeyCodes() throws {
        let manager = self.makeManager(asr: ASRService(), keyPressed: { true }, start: {})
        // Subtype 7 is a mouse-button event; AppKit throws if its keyCode is read.
        for pressed in [1, 0] {
            let event = try XCTUnwrap(NSEvent.otherEvent(
                with: .systemDefined,
                location: .zero,
                modifierFlags: [],
                timestamp: ProcessInfo.processInfo.systemUptime,
                windowNumber: 0,
                context: nil,
                subtype: 7,
                data1: 1,
                data2: pressed
            ))
            XCTAssertTrue(manager.handleFocusedAppMonitorEvent(event) === event)
        }
    }

    func testNewerManagersCannotStealEventsOnTheRealCarbonHandlerChain() async throws {
        var firstStarts = 0
        var secondStarts = 0
        let first = self.makeManager(asr: ASRService(), keyPressed: { true }, start: { firstStarts += 1 })
        let second = self.makeManager(asr: ASRService(), keyPressed: { true }, start: { secondStarts += 1 })
        XCTAssertNotEqual(first.carbonEventSignature, second.carbonEventSignature)
        let target = try XCTUnwrap(GetEventDispatcherTarget())
        var types = [
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased)),
        ]
        var firstHandler: EventHandlerRef?
        var secondHandler: EventHandlerRef?
        defer {
            if let secondHandler {
                RemoveEventHandler(secondHandler)
            }
            if let firstHandler {
                RemoveEventHandler(firstHandler)
            }
        }
        XCTAssertEqual(InstallEventHandler(
            target,
            ownershipTestCarbonHandler,
            types.count,
            &types,
            Unmanaged.passUnretained(first).toOpaque(),
            &firstHandler
        ), noErr)
        XCTAssertEqual(InstallEventHandler(
            target,
            ownershipTestCarbonHandler,
            types.count,
            &types,
            Unmanaged.passUnretained(second).toOpaque(),
            &secondHandler
        ), noErr)
        try self.withNativeHotKeyEvent(signature: first.carbonEventSignature, isPressed: true) { event in
            XCTAssertEqual(second.handleNativeCarbonHotKeyEvent(event), OSStatus(eventNotHandledErr))
            XCTAssertEqual(SendEventToEventTarget(event, target), noErr)
        }
        try self.withNativeHotKeyEvent(signature: first.carbonEventSignature, isPressed: false) { event in
            XCTAssertEqual(SendEventToEventTarget(event, target), noErr)
        }
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(firstStarts, 1)
        XCTAssertEqual(secondStarts, 0)
    }

    func testMissingReleaseAfterWindowClosesDoesNotLeaveToggleStuck() async {
        let asr = ASRService()
        defer { asr.isRunning = false }
        var keyPressed = true
        var starts = 0
        var stops = 0
        let manager = self.makeManager(
            asr: asr,
            keyPressed: { keyPressed },
            start: { starts += 1; asr.isRunning = true },
            stop: { stops += 1; asr.isRunning = false }
        )
        XCTAssertTrue(manager.handleFocusedAppPrimaryShortcut(keyCode: 49, modifiers: [.control, .option], isPressed: true))
        // The window closes before AppKit/Carbon delivers key-up.
        keyPressed = false
        manager.recoverReleasedCarbonHotKeys()
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(starts, 1)

        // A delayed release cannot toggle again; the next global press still works.
        manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: false)
        keyPressed = true
        manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: true)
        keyPressed = false
        manager.recoverReleasedCarbonHotKeys()
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(starts, 1)
        XCTAssertEqual(stops, 1)
    }

    func testReleaseRecoveryDoesNotReleaseAPhysicallyHeldChord() async {
        var starts = 0
        let manager = self.makeManager(asr: ASRService(), keyPressed: { true }, start: { starts += 1 })
        manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: true)
        manager.recoverReleasedCarbonHotKeys()
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(starts, 0)
        manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: false)
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(starts, 1)
    }

    func testLocalShortcutRecoversMissingReleaseWithoutAnotherEvent() async {
        let started = expectation(description: "Missing local key-up is recovered")
        var keyPressed = true
        var starts = 0
        let manager = self.makeManager(
            asr: ASRService(),
            keyPressed: { keyPressed },
            start: { starts += 1; started.fulfill() }
        )
        XCTAssertTrue(manager.handleFocusedAppPrimaryShortcut(keyCode: 49, modifiers: [.control, .option], isPressed: true))
        keyPressed = false
        await fulfillment(of: [started], timeout: 2)
        XCTAssertEqual(starts, 1)
        withExtendedLifetime(manager) {}
    }

    func testTrustedAccessibilityDoesNotAlsoUseFocusedCarbonRoute() {
        let manager = self.makeManager(asr: ASRService(), trusted: true, keyPressed: { true }, start: {})
        XCTAssertFalse(manager.handleFocusedAppPrimaryShortcut(keyCode: 49, modifiers: [.control, .option], isPressed: true))
    }

    func testDelayedCarbonDownAfterRecoveredReleaseDoesNotToggleAgain() async {
        var keyPressed = true
        var starts = 0
        let manager = self.makeManager(asr: ASRService(), keyPressed: { keyPressed }, start: { starts += 1 })
        let eventTime = ProcessInfo.processInfo.systemUptime
        manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: true, timestamp: eventTime)
        keyPressed = false
        manager.recoverReleasedCarbonHotKeys()
        manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: true, timestamp: eventTime)
        manager.recoverReleasedCarbonHotKeys()
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(starts, 1)
    }

    func testMissingHoldReleaseStopsRecordingOnlyOnce() async {
        let asr = ASRService()
        defer { asr.isRunning = false }
        var keyPressed = true
        var starts = 0
        var stops = 0
        let manager = self.makeManager(
            asr: asr,
            keyPressed: { keyPressed },
            start: { starts += 1; asr.isRunning = true },
            stop: { stops += 1; asr.isRunning = false }
        )
        manager.setHotkeyMode(.hold)
        manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: true)
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(starts, 1)
        manager.recoverReleasedCarbonHotKeys()
        XCTAssertEqual(stops, 0)
        keyPressed = false
        manager.recoverReleasedCarbonHotKeys()
        manager.handleCarbonHotKey(id: CarbonHotKeyAction.primary(0).id, isPressed: false)
        for _ in 0..<20 {
            await Task.yield()
        }
        XCTAssertEqual(stops, 1)
    }

    func testUnrelatedLocalKeysRemainAvailableToTheWindow() {
        let manager = self.makeManager(asr: ASRService(), keyPressed: { true }, start: {})
        XCTAssertFalse(manager.handleFocusedAppPrimaryShortcut(keyCode: 49, modifiers: [], isPressed: true))
        XCTAssertFalse(manager.handleFocusedAppPrimaryShortcut(keyCode: 0, modifiers: [.control, .option], isPressed: true))
    }
}

final class CarbonPressTimestampTests: XCTestCase {
    func testNoInputMonitoringUsesModifierReleaseInsteadOfFalseRawKeyState() {
        let shortcut = HotkeyShortcut(keyCode: 49, modifierFlags: [.control, .option])
        XCTAssertEqual(shortcut.carbonPressIsHeld(physicalKeyState: nil, modifiers: [.control, .option]), true)
        XCTAssertEqual(shortcut.carbonPressIsHeld(physicalKeyState: nil, modifiers: [.control]), false)
        XCTAssertEqual(shortcut.carbonPressIsHeld(physicalKeyState: nil, modifiers: []), false)
        XCTAssertEqual(shortcut.carbonPressIsHeld(physicalKeyState: true, modifiers: []), true)
        XCTAssertEqual(shortcut.carbonPressIsHeld(physicalKeyState: false, modifiers: [.control, .option]), false)
        XCTAssertNil(HotkeyShortcut(keyCode: 53, modifierFlags: []).carbonPressIsHeld(physicalKeyState: nil, modifiers: []))
    }

    func testDelayedEventsCannotReplayCompletedPressOrReleaseANewerPress() {
        var tracker = CarbonHotKeyPressTracker()
        XCTAssertTrue(tracker.beginPress(id: 1, timestamp: 10))
        XCTAssertTrue(tracker.endPress(id: 1, timestamp: 11))
        XCTAssertFalse(tracker.beginPress(id: 1, timestamp: 10))
        XCTAssertTrue(tracker.beginPress(id: 1, timestamp: 12))
        XCTAssertFalse(tracker.endPress(id: 1, timestamp: 11))
        XCTAssertTrue(tracker.isPressed(id: 1))
        XCTAssertTrue(tracker.endPress(id: 1, timestamp: 13))
    }
}
