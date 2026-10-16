import AppKit

@main
struct VoiceTrainingAliasFilterTests {
    @MainActor
    static func main() async {
        let ordinary = ["but", "NOW", "okay", "hello", "right now", "let’s do that"]
        let native = await VoiceTrainingAliasFilter.filter(ordinary + ["fluwidvois"])
        FileHandle.standardOutput.write(Data("native accepted=\(native.accepted) rejected=\(native.rejected) available=\(native.lookupAvailable)\n".utf8))
        precondition(native.lookupAvailable, "Native English word lookup must be available for this smoke test")
        precondition(native.accepted == ["fluwidvois"])
        precondition(native.rejected == ordinary)

        // Deterministic policy coverage, independent of the installed dictionary version.
        let aliases = ["now", "right now", "fluid zqxword", "🙂 zqxword", "123", "", String(repeating: "x", count: 161)]
        let filtered = await VoiceTrainingAliasFilter.filter(aliases) { text in
            guard let regex = try? NSRegularExpression(pattern: "zqxword") else { preconditionFailure("Invalid fixture pattern") }
            return .checked(regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).map(\.range))
        }
        precondition(filtered.accepted == ["fluid zqxword", "🙂 zqxword"])
        precondition(filtered.rejected == ["now", "right now", "123", "", String(repeating: "x", count: 161)])

        let unavailable = await VoiceTrainingAliasFilter.filter(["now", "zqxword"]) { _ in .unavailable }
        precondition(!unavailable.lookupAvailable && unavailable.accepted.isEmpty)

        let cancelled = Task { @MainActor in
            await VoiceTrainingAliasFilter.filter(["zqxword"]) { text in
                try? await Task.sleep(for: .milliseconds(30))
                return .checked([NSRange(text.startIndex..., in: text)])
            }
        }
        cancelled.cancel()
        let cancelledResult = await cancelled.value
        precondition(cancelledResult.accepted.isEmpty)

        let results = await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<3 {
                group.addTask {
                    let result = await VoiceTrainingAliasFilter.filter(["now", "fluwidvois"])
                    return result.accepted == ["fluwidvois"]
                }
            }
            var values: [Bool] = []
            for await result in group {
                values.append(result)
            }
            return values
        }
        precondition(results.count == 3 && results.allSatisfy { $0 })
        print("PASS: native vocabulary, common phrases, rare aliases, UTF-16 ranges, invalid input, unavailable lookup, cancellation, concurrent requests")
    }
}
