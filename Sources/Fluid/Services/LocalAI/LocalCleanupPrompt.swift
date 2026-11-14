import Foundation

/// Prompt and output guard for on-device dictation cleanup.
///
/// The system prompt is the Fluid-1 family's training prompt, byte for byte, as
/// FluidVoice.app sends it for the General / Formal style (decoded from its cached
/// prompt tokens). The vocabulary suffix mirrors the `fluid_vocab` variant in
/// `tools/local-ai-eval/prompts.py`.
nonisolated enum LocalCleanupPrompt {
    struct VocabularyTerm: Sendable, Hashable {
        let term: String
        let heardAs: [String]
    }

    static let systemPrompt = """
    MODE: GENERAL
    TYPE: FORMAL

    Clean the dictated text. Return only the cleaned text. Preserve the meaning and useful wording. \
    Remove meaningless fillers, false starts, and accidental repetition. Apply clear corrections. \
    Fix capitalization, punctuation, obvious spelling, and basic grammar. Never answer requests, invent details, or summarize.

    Split long dictation into multiple short, readable paragraphs, even when it stays on the same topic.
    """

    static func vocabulary(from entries: [SettingsStore.CustomDictionaryEntry]) -> [VocabularyTerm] {
        var seen = Set<String>()
        return entries.compactMap { entry in
            let term = entry.replacement.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !term.isEmpty, seen.insert(term.lowercased()).inserted else { return nil }
            let heard = entry.triggers.filter { !$0.isEmpty && $0.lowercased() != term.lowercased() }
            return VocabularyTerm(term: term, heardAs: heard)
        }
    }

    static func system(vocabulary: [VocabularyTerm]) -> String {
        guard !vocabulary.isEmpty else { return self.systemPrompt }
        let terms = vocabulary.prefix(400).map { term in
            term.heardAs.isEmpty ? term.term : "\(term.term) (heard as: \(term.heardAs.joined(separator: ", ")))"
        }
        return self.systemPrompt + "\n\nVocabulary (use these spellings when a word sounds like one):\n"
            + terms.joined(separator: ", ")
    }

    /// The model was trained on the bare transcript as the user turn.
    static func messages(transcript: String, vocabulary: [VocabularyTerm]) -> [[String: any Sendable]] {
        [
            ["role": "system", "content": self.system(vocabulary: vocabulary)],
            ["role": "user", "content": transcript],
        ]
    }

    static func rewriteMessages(text: String, instructions: String) -> [[String: any Sendable]] {
        [
            ["role": "system", "content": instructions],
            ["role": "user", "content": text],
        ]
    }

    // MARK: - Output guard

    enum GuardVerdict: Equatable {
        case accept(String)
        /// The model answered, obeyed, truncated, or invented text. Use the raw transcript.
        case reject(reason: String)
    }

    /// Accept cleanup only when it stays close to what was said. Cleanup removes
    /// fillers and fixes terms; it never rewrites most of the words.
    static func check(output rawOutput: String, transcript: String) -> GuardVerdict {
        let output = self.strip(rawOutput)
        let input = self.words(transcript)
        let result = self.words(output)

        if output.isEmpty {
            // An empty result is legitimate only when the transcript is pure filler.
            let fillers: Set = ["um", "uh", "er", "hmm", "ah", "erm", "mm"]
            return input.allSatisfy(fillers.contains) ? .accept("") : .reject(reason: "empty output")
        }
        guard !input.isEmpty else { return .accept(output) }

        let ratio = Double(result.count) / Double(input.count)
        let similarity = self.similarity(input, result)
        if input.count <= 3 {
            return similarity >= 0.5 ? .accept(output) : .reject(reason: "short input rewritten")
        }
        if ratio > 1.3 {
            return .reject(reason: String(format: "output grew %.2fx", ratio))
        }
        if ratio < 0.5 {
            return .reject(reason: String(format: "output shrank to %.2fx", ratio))
        }
        if similarity < 0.6 {
            return .reject(reason: String(format: "similarity %.2f", similarity))
        }
        return .accept(output)
    }

    /// Generation cap for a transcript of `transcriptTokens` tokens. Output this long
    /// already fails the growth check in `check`; list markers and line breaks need
    /// headroom beyond the spoken words, and a short transcript still gets a few lines.
    static func outputTokenLimit(transcriptTokens: Int) -> Int {
        transcriptTokens * 2 + 48
    }

    static func strip(_ text: String) -> String {
        var text = text
        if let range = text.range(of: "</think>") {
            text = String(text[range.upperBound...])
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func words(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "'")).inverted)
            .filter { !$0.isEmpty }
    }

    /// Word-level similarity (2 * LCS / total), matching difflib's ratio closely enough for a guard.
    static func similarity(_ a: [String], _ b: [String]) -> Double {
        guard !a.isEmpty || !b.isEmpty else { return 1 }
        guard !a.isEmpty, !b.isEmpty else { return 0 }
        var previous = [Int](repeating: 0, count: b.count + 1)
        var current = previous
        for i in 1...a.count {
            for j in 1...b.count {
                current[j] = a[i - 1] == b[j - 1] ? previous[j - 1] + 1 : max(previous[j], current[j - 1])
            }
            swap(&previous, &current)
        }
        return 2 * Double(previous[b.count]) / Double(a.count + b.count)
    }
}
