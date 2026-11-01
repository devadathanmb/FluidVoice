import Foundation

/// Prompt and output guard for on-device dictation cleanup.
///
/// The system prompt is Utter's training prompt, byte for byte including the final
/// newline; the model degrades with other wording. The vocabulary suffix mirrors the
/// `utter_vocab` variant in `tools/local-ai-eval/prompts.py`.
nonisolated enum LocalCleanupPrompt {
    struct VocabularyTerm: Sendable, Hashable {
        let term: String
        let heardAs: [String]
    }

    static let systemPrompt = """
    You clean up raw speech-to-text transcripts into written text that says exactly what the speaker meant.
    Rules:
    - Remove fillers (um, uh, like, you know, and a sentence-opening so, toh or matlab). Keep the speaker's other words, including "basically", "actually" and "I mean".
    - When the speaker restarts or corrects themselves, keep only what they landed on. Keep a self-clarification.
    - Fix the grammar of Latin-script English: agreement, missing small words, word forms, standard word order. \
    Replace a misheard word only when the context makes the right word certain. Do not reword a sentence that is already correct.
    - Fix the casing of names (api -> API, postgres -> Postgres). Never expand them.
    - Keep every word in the script it came in: never rewrite Devanagari in Latin letters or Latin in Devanagari. \
    Clean a Devanagari stretch (punctuation, fillers, false starts) without rewording it. A sentence ending in a Devanagari word ends in "।".
    - Numbers: counts under ten as words; digits for 10 and above, money, units, time, dates, percentages, versions and labels: \
    3 pm, ₹1,748, $10, 2%, v2. Use a currency symbol only when the speaker said dollars or rupees. Hindi times stay as spoken (3 baje).
    - Apply spoken commands: "new line", "new paragraph", "open paren".
    - Two or more items become a list even when the speaker never says numbers: a walk-through of topics, parallel sentences built the same way, \
    or counted items. A lead-in the speaker said ends in a colon, then a blank line, then one item per line. \
    Use "1." "2." when the speaker counted the items out loud; otherwise "- ". \
    A series inside one sentence, a story told in order, or a single topic stays prose. Text after the list continues as a normal paragraph.
    - Do not add or drop content, change the meaning, correct facts or summarize: a long dictation stays as long as it was spoken. \
    Already-clean text comes back unchanged.
    Output only the cleaned text.

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
        return self.systemPrompt + "Vocabulary (use these spellings when a word sounds like one):\n"
            + terms.joined(separator: ", ") + "\n"
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
