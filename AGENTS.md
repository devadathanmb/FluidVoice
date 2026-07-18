# FluidVoice Agent Guide

## Fork Purpose And Permission Boundary

- This `freedom` branch is a FluidVoice fork for managed macOS machines where Accessibility (AX) permission may be unavailable. Its Carbon fallback preserves eligible global **keyboard-chord** hotkeys without AX; do not describe it as an Accessibility bypass.
- AX remains required for cross-app text selection, focus capture/restoration, typing, and paste insertion. `TypingService` and `TextSelectionService` intentionally stop when `AXIsProcessTrusted()` is false.
- Without AX, only modified keyboard chords can be registered through Carbon. Mouse, Fn, and modifier-only shortcuts are unavailable. The unmodified Escape cancel shortcut is deliberately registered only while recording.
- Clipboard copying is independent of Carbon and uses `NSPasteboard` via `ClipboardService`. The `Copy to Clipboard` setting is the usable no-AX handoff; it does not insert text into the target app.
- `Info.plist` still says AX is needed for global hotkeys. If changing permission copy, follow the current runtime and onboarding behavior instead: AX is optional for eligible Carbon chords but required for cross-app text access and insertion.

## Hotkey Changes

- Keep the paths mutually exclusive: trusted AX uses the CG event tap; untrusted AX uses Carbon. Never register an action through both.
- Preserve Carbon action-ID namespaces, duplicate-candidate reporting, and press coalescing in `Sources/Fluid/Models/CarbonHotKeySupport.swift`.
- Preserve the transient Escape lifecycle: unregister it and clear press tracking after recording so a later recording can register it again.
- The primary default migrated from Right Option to Control-Option-Space only for the exact old persisted default. Do not migrate customized shortcuts; see `CarbonPrimaryShortcutDefaultMigrated` in `SettingsStore`.

## Source Map

- `Sources/Fluid/fluidApp.swift` starts the SwiftUI app; `AppDelegate.swift` performs service startup and shutdown; `ContentView.swift` wires dictation completion, copying, and insertion.
- `AppServices.swift` owns the main-actor app service graph. `GlobalHotkeyManager.swift` owns AX/Carbon hotkey registration. `TypingService.swift` owns AX-gated injection and temporary-paste fallback.
- `Sources/CoreAudioCaptureSupport` is the C CoreAudio capture target used by `DirectCoreAudioInput`.
- The app requires macOS 15.0+ and is built through `Fluid.xcodeproj` (Swift Package Manager dependencies are locked in `Package.resolved`).

## Development And Verification

- Open `Fluid.xcodeproj` in Xcode. Local runs need a personal signing team; never commit `DEVELOPMENT_TEAM` changes.
- `./build.sh` builds the public unsigned OSS profile. `./build.sh fi` requires the untracked executable `build_with_FI_incremental.sh`; do not assume private Fluid Intelligence sources are present.
- Run unsigned tests with `xcodebuild test -project Fluid.xcodeproj -scheme Fluid -destination 'platform=macOS' CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO`.
- Run `scripts/format-and-lint.sh` before submitting Swift changes. It formats `Sources/` in place, then runs strict SwiftLint, and installs missing tools through Homebrew unless `FLUIDVOICE_NO_TOOL_INSTALL=1`.
- CI skips `FluidDictationIntegrationTests/DictationE2ETests/testDictationEndToEnd_whisperTiny_transcribesFixture` on hosted macOS because Tiny Whisper output is nondeterministic. Keep it for local validation; do not remove the test to make CI pass.
- If replacing `Tests/FluidDictationIntegrationTests/Resources/dictation_fixture.wav`, retain mono 16 kHz, 16-bit PCM WAV, roughly 2-4 seconds.

## Distribution

- `.github/workflows/personal-macos-dmg.yml` is manually dispatched. It runs unsigned ARM64 tests, builds Release, and uploads `FluidVoice-freedom-arm64.dmg` for 30 days. Preserve its no-signing flags and the Whisper E2E exclusion when changing this workflow.
