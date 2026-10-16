import Foundation

@MainActor
final class CancellableDictationRequest<Value: Sendable> {
    private var continuation: CheckedContinuation<Value, Error>?
    private var worker: Task<Void, Never>?
    private var didStart = false
    private var isFinished = false
    private(set) var isCancelled = false

    func value(operation: @escaping @MainActor () async throws -> Value) async throws -> Value {
        if Task.isCancelled {
            self.cancel()
        }
        guard !self.isCancelled else { throw CancellationError() }
        precondition(!self.didStart, "A dictation request can only be started once")
        self.didStart = true

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                self.continuation = continuation
                self.worker = Task { @MainActor [weak self] in
                    do {
                        try Task.checkCancellation()
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

    private func finish(_ result: Result<Value, Error>) {
        guard !self.isFinished else { return }
        self.isFinished = true
        self.worker = nil
        let continuation = self.continuation
        self.continuation = nil
        continuation?.resume(with: result)
    }
}
