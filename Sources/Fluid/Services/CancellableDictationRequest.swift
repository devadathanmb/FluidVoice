import Foundation

nonisolated enum DictationAIRequestError: LocalizedError {
    case timedOut
    case preparationTimedOut

    var errorDescription: String? {
        switch self {
        case .timedOut:
            "AI enhancement exceeded 10 seconds. Using raw dictation."
        case .preparationTimedOut:
            "The AI model did not load within 60 seconds. Using raw dictation."
        }
    }
}

/// Not generic over the result: Swift 6.3's Release optimizer crashes inlining this
/// class's deinit when the class itself is generic (Xcode 26.6, EarlyPerfInliner).
@MainActor
final class CancellableDictationRequest {
    static let postProcessingTimeoutSeconds: TimeInterval = 10
    /// Loading a local model from disk is not generation; it gets its own, longer cap.
    static let preparationTimeoutSeconds: TimeInterval = 60
    private var continuation: CheckedContinuation<any Sendable, Error>?
    private var worker: Task<Void, Never>?
    private var timeoutTask: Task<Void, Never>?
    private var didStart = false
    private var isFinished = false
    private(set) var isCancelled = false

    /// `prepare` runs before `operation` under `preparationTimeoutSeconds`; the
    /// `timeoutSeconds` clock starts only once it returns.
    func value<Value: Sendable>(
        timeoutSeconds: TimeInterval? = nil,
        prepare: (@MainActor () async -> Void)? = nil,
        operation: @escaping @MainActor () async throws -> Value
    ) async throws -> Value {
        if Task.isCancelled {
            self.cancel()
        }
        guard !self.isCancelled else { throw CancellationError() }
        precondition(!self.didStart, "A dictation request can only be started once")
        self.didStart = true

        let result = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<any Sendable, Error>) in
                self.continuation = continuation
                self.worker = Task { @MainActor [weak self] in
                    do {
                        try Task.checkCancellation()
                        if let prepare {
                            if timeoutSeconds != nil {
                                self?.armTimeout(Self.preparationTimeoutSeconds, error: .preparationTimedOut)
                            }
                            await prepare()
                            try Task.checkCancellation()
                        }
                        if let timeoutSeconds {
                            self?.armTimeout(timeoutSeconds, error: .timedOut)
                        }
                        let result = try await operation()
                        try Task.checkCancellation()
                        self?.finish(.success(result))
                    } catch {
                        self?.finish(.failure(error))
                    }
                }
            }
        } onCancel: {
            Task { @MainActor [weak self] in self?.cancel() }
        }
        guard let value = result as? Value else {
            preconditionFailure("A dictation request resumed with a result of the wrong type")
        }
        return value
    }

    func cancel() {
        guard !self.isFinished else { return }
        self.isCancelled = true
        let worker = self.worker
        // Release the UI waiter before a provider's cancellation acknowledgement.
        // The worker still owns its resource leases until it actually drains.
        self.finish(.failure(CancellationError()))
        worker?.cancel()
    }

    private func armTimeout(_ seconds: TimeInterval, error: DictationAIRequestError) {
        self.timeoutTask?.cancel()
        guard !self.isFinished else { return }
        self.timeoutTask = Task { @MainActor [weak self] in
            do {
                try await Task.sleep(for: .seconds(seconds))
            } catch {
                return
            }
            guard let self, !self.isFinished else { return }
            let worker = self.worker
            self.finish(.failure(error))
            worker?.cancel()
        }
    }

    private func finish(_ result: Result<any Sendable, Error>) {
        guard !self.isFinished else { return }
        self.isFinished = true
        self.timeoutTask?.cancel()
        self.timeoutTask = nil
        self.worker = nil
        let continuation = self.continuation
        self.continuation = nil
        continuation?.resume(with: result)
    }
}
