import Foundation

enum CarbonHotKeyAction: Hashable {
    case primary(Int)
    case promptMode
    case commandMode
    case rewriteMode
    case promptAssignment(Int)
    case cancel
    case meetingRecording

    var id: UInt32 {
        switch self {
        case let .primary(index): 0x10000000 | UInt32(index)
        case .promptMode: 0x20000000
        case .commandMode: 0x30000000
        case .rewriteMode: 0x40000000
        case let .promptAssignment(index): 0x50000000 | UInt32(index)
        case .cancel: 0x60000000
        case .meetingRecording: 0x70000000
        }
    }

    var isPrimary: Bool {
        if case .primary = self {
            return true
        }
        return false
    }
}

struct CarbonHotKeyRegistrationCandidate {
    let action: CarbonHotKeyAction
    let shortcut: HotkeyShortcut
    let label: String
}

struct CarbonHotKeyRegistrationPlan {
    let registrations: [CarbonHotKeyRegistrationCandidate]
    let issues: [String]

    var primaryRegistrationCount: Int {
        self.registrations.count { $0.action.isPrimary }
    }

    init(candidates: [CarbonHotKeyRegistrationCandidate]) {
        var registrations: [CarbonHotKeyRegistrationCandidate] = []
        var issues: [String] = []

        for candidate in candidates {
            if let reason = candidate.shortcut.carbonIneligibilityReason {
                issues.append("\(candidate.label): \(reason.rawValue)")
                continue
            }
            if let duplicate = registrations.first(where: { $0.shortcut == candidate.shortcut }) {
                issues.append("\(candidate.label) (\(candidate.shortcut.displayString)): duplicates \(duplicate.label).")
                continue
            }
            registrations.append(candidate)
        }

        self.registrations = registrations
        self.issues = issues
    }
}

enum CarbonCancelHotKeyRegistrationMode: Equatable {
    case persistent
    case whileRecording
    case unsupported
}

struct CarbonHotKeyPressTracker {
    private var pressedIDs: Set<UInt32> = []
    private var pressTimes: [UInt32: TimeInterval] = [:]
    private var releaseTimes: [UInt32: TimeInterval] = [:]

    var activeIDs: Set<UInt32> {
        self.pressedIDs
    }

    mutating func beginPress(id: UInt32, timestamp: TimeInterval? = nil) -> Bool {
        if let timestamp, let releasedAt = self.releaseTimes[id], timestamp <= releasedAt {
            return false
        }
        guard self.pressedIDs.insert(id).inserted else { return false }
        self.pressTimes[id] = timestamp
        return true
    }

    mutating func endPress(id: UInt32, timestamp: TimeInterval? = nil) -> Bool {
        if let timestamp, let pressedAt = self.pressTimes[id], timestamp < pressedAt {
            return false
        }
        guard self.pressedIDs.remove(id) != nil else { return false }
        self.pressTimes.removeValue(forKey: id)
        self.releaseTimes[id] = timestamp
        return true
    }

    mutating func cancelPress(id: UInt32) {
        self.pressedIDs.remove(id)
        self.pressTimes.removeValue(forKey: id)
        self.releaseTimes.removeValue(forKey: id)
    }

    func isPressed(id: UInt32) -> Bool {
        self.pressedIDs.contains(id)
    }

    mutating func reset() {
        self.pressedIDs.removeAll()
        self.pressTimes.removeAll()
        self.releaseTimes.removeAll()
    }
}
