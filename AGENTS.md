# FluidVoice Freedom Fork

## Purpose And Invariants

- `freedom` supports managed macOS machines where Accessibility (AX) permission may be unavailable. Describe this as a Carbon hotkey fallback, never as an AX bypass.
- Without AX, Carbon supports eligible modified keyboard chords only. Mouse, Fn, and modifier-only shortcuts remain unavailable.
- AX is still required for cross-app text selection, focus capture/restoration, typing, and paste insertion. `TypingService` and `TextSelectionService` must stop when `AXIsProcessTrusted()` is false.
- Clipboard copying is independent of AX and uses `NSPasteboard` through `ClipboardService`. `Copy to Clipboard` is the no-AX handoff; it does not insert text into another app.

## Upstream Integration

- Base `freedom` only on stable upstream release tags such as `v1.6.9`. Do not integrate `upstream/main`, beta tags, or arbitrary upstream branches or commits.
- Bring a newer stable release into `freedom` with a merge commit; do not rebase the branch. Preserve fork commits and make the release boundary explicit in history.
- Fetch upstream tags and verify the selected stable release before merging. Do not assume the newest upstream commit is releasable.
- Preserve the current branch tip with a backup branch before any exceptional history rewrite.

## Hotkey Requirements

- Trusted AX uses the CG event tap; untrusted AX uses Carbon. Never register the same action through both paths.
- Preserve Carbon action-ID namespaces, duplicate-candidate reporting, and press coalescing in `Sources/Fluid/Models/CarbonHotKeySupport.swift`.
- When FluidVoice is active without AX, local key-down and key-up events must route through `handleFocusedAppPrimaryShortcut`. The local and Carbon paths share press tracking to prevent missed or duplicate triggers.
- Escape is a transient Carbon registration while recording. Unregister it and clear its press state afterward so later recordings can register it again.
- Migrate Right Option to Control-Option-Space only for the exact old persisted default. Never replace a customized shortcut; see `CarbonPrimaryShortcutDefaultMigrated` in `SettingsStore`.

## Code Map

- `fluidApp.swift` starts the app; `AppDelegate.swift` manages startup and shutdown; `ContentView.swift` coordinates dictation completion and output.
- `AppServices.swift` owns the main-actor service graph. `GlobalHotkeyManager.swift` owns AX and Carbon hotkeys. `TypingService.swift` owns AX-gated insertion.
- `Sources/CoreAudioCaptureSupport` is the C capture target used by `DirectCoreAudioInput`.
- The app targets macOS 15+ and builds through `Fluid.xcodeproj`; Swift package versions are locked in `Package.resolved`.

## Verification And Distribution

- Never commit personal signing-team changes. Build and test unsigned with `CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO`.
- Run `scripts/format-and-lint.sh` for Swift changes. It formats `Sources/` before running strict SwiftLint.
- Run tests with `xcodebuild test -project Fluid.xcodeproj -scheme Fluid -destination 'platform=macOS' CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO`.
- Keep the Tiny Whisper E2E test for local validation. CI may skip it because its output is nondeterministic on hosted macOS.
- `./build.sh` creates the public unsigned OSS build. `./build.sh fi` depends on the untracked private `build_with_FI_incremental.sh`; do not assume it exists.
- `.github/workflows/personal-macos-dmg.yml` must remain manually dispatched, unsigned, ARM64, retain the hosted Whisper exclusion, and upload `FluidVoice-freedom-arm64.dmg` for 30 days.
