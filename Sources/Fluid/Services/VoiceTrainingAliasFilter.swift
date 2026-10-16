import AppKit

/// Voice captures may teach pronunciation without creating a global text replacement.
/// A spell-check miss is only an eligibility guard, not proof that an alias is correct.
@MainActor
enum VoiceTrainingAliasFilter {
    struct Result {
        let accepted: [String]
        let rejected: [String]
        let lookupAvailable: Bool
    }

    enum LookupResult: Sendable {
        case checked([NSRange])
        case unavailable
    }

    typealias Lookup = @MainActor (String) async -> LookupResult

    static func filter(_ aliases: [String], lookup: Lookup = checkSpelling) async -> Result {
        // Training already caps captures at 20. Bound this API independently too.
        let bounded = Array(aliases.prefix(20))
        var text = ""
        var candidates: [(alias: String, range: NSRange)] = []
        for alias in bounded {
            guard !alias.isEmpty, alias.count <= 160, alias.contains(where: \.isLetter) else { continue }
            let normalized = alias.lowercased().split(whereSeparator: \.isWhitespace).joined(separator: " ")
            let range = NSRange(location: text.utf16.count, length: normalized.utf16.count)
            text += normalized + "\n"
            candidates.append((alias, range))
        }
        guard !candidates.isEmpty else {
            return Result(accepted: [], rejected: aliases, lookupAvailable: true)
        }
        guard case let .checked(misspellings) = await lookup(text), !Task.isCancelled else {
            // Never interpret an unavailable dictionary as permission to add every alias.
            return Result(accepted: [], rejected: aliases, lookupAvailable: false)
        }
        let accepted = candidates.filter { candidate in
            misspellings.contains { NSIntersectionRange($0, candidate.range).length > 0 }
        }.map(\.alias)
        let acceptedSet = Set(accepted)
        return Result(accepted: accepted, rejected: aliases.filter { !acceptedSet.contains($0) }, lookupAvailable: true)
    }

    /// English vocabulary is checked locally; no remote API or model inference.
    /// This is called only when saving voice training, never during live dictation/rendering.
    static func checkSpelling(_ text: String) async -> LookupResult {
        let request = SpellingRequest()
        return await withCheckedContinuation { continuation in
            request.start(text, continuation: continuation)
        }
    }

    @MainActor
    private final class SpellingRequest {
        private var continuation: CheckedContinuation<LookupResult, Never>?
        private var timeout: Task<Void, Never>?
        private var documentTag: Int?

        func start(_ text: String, continuation: CheckedContinuation<LookupResult, Never>) {
            self.continuation = continuation
            let tag = NSSpellChecker.uniqueSpellDocumentTag()
            self.documentTag = tag
            self.timeout = Task { @MainActor in
                do { try await Task.sleep(for: .seconds(2)) } catch { return }
                self.finish(.unavailable)
            }
            NSSpellChecker.shared.requestChecking(
                of: text,
                range: NSRange(text.startIndex..., in: text),
                types: NSTextCheckingResult.CheckingType.spelling.rawValue,
                options: [.orthography: NSOrthography.defaultOrthography(forLanguage: "en_US")],
                inSpellDocumentWithTag: tag
            ) { @Sendable _, results, _, wordCount in
                let ranges = results.filter { $0.resultType == .spelling }.map(\.range)
                Task { @MainActor in
                    self.finish(wordCount > 0 ? .checked(ranges) : .unavailable)
                }
            }
        }

        private func finish(_ result: LookupResult) {
            guard let continuation = self.continuation else { return }
            self.continuation = nil
            self.timeout?.cancel()
            self.timeout = nil
            if let documentTag = self.documentTag {
                NSSpellChecker.shared.closeSpellDocument(withTag: documentTag)
                self.documentTag = nil
            }
            continuation.resume(returning: result)
        }
    }
}
