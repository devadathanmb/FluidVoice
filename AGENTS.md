# FluidVoice Freedom Fork

`freedom` adds two things to FluidVoice: a Carbon hotkey fallback for managed Macs where Accessibility (AX) may be unavailable, and on-device AI post-processing that replaces upstream's private Fluid Intelligence build.

## Accessibility Invariants

- Describe the no-AX path as a Carbon hotkey fallback, never as an AX bypass.
- Without AX, Carbon supports eligible modified keyboard chords only. Mouse, Fn, and modifier-only shortcuts remain unavailable.
- AX is still required for cross-app text selection, focus capture/restoration, typing, and paste insertion. `TypingService` and `TextSelectionService` must stop when `AXIsProcessTrusted()` is false.
- Clipboard copying is independent of AX and uses `NSPasteboard` through `ClipboardService`. `Copy to Clipboard` is the no-AX handoff; it does not insert text into another app.
- Without AX, completed dictation uses clipboard-only delivery, not a failed paste attempt or an Accessibility warning. Keep sandbox/history-only output free of clipboard side effects, and do not arm Spoken Send without AX.

## Local AI Post-Processing

- `Sources/Fluid/Services/LocalAI/` implements upstream's private provider seam (`PrivateAIProviderBridge`, compiled with `PRIVATE_AI_PROVIDER` on the app target only). Keep local AI changes inside that seam so release merges stay clean.
- `LocalCleanupRuntime` runs mlx-swift-lm in process and must stay light beside dev tools: one resident model, a capped MLX buffer cache, and the instruction prefix prefilled once per vocabulary.
- `LocalCleanupModelCatalog` downloads standard MLX checkpoints pinned to a Hugging Face revision, size-checked and SHA-256-checked. Publish converted or fine-tuned weights to Hugging Face; do not add on-device weight conversion.
- `LocalCleanupPrompt.systemPrompt` is the model's training prompt byte for byte, pinned by a SHA-256 test. Change prompt and model together and keep `tools/local-ai-eval/prompts.py` in sync.
- Custom dictionary entries become the prompt vocabulary. The output guard returns the raw transcript when the model answers, summarizes, truncates, or invents text; keep that fallback.
- Choose or change models with `tools/local-ai-eval` (uv, Python 3.12) and judge them on the held-out real dictation history in its `private/`, not only synthetic cases. Personal data and history results are git-ignored; never commit them.

## Upstream Integration

- Base `freedom` only on stable upstream release tags such as `v1.6.9`. Do not integrate `upstream/main`, beta tags, or arbitrary upstream branches or commits.
- Bring a newer stable release into `freedom` with a merge commit; do not rebase the branch. Preserve fork commits and make the release boundary explicit in history.
- Fetch upstream tags and verify the selected stable release before merging. Do not assume the newest upstream commit is releasable.
- Preserve the current branch tip with a backup branch before any exceptional history rewrite.

## Hotkey Requirements

- Trusted AX uses the CG event tap; untrusted AX uses Carbon. Never register the same action through both paths.
- Preserve Carbon action-ID namespaces, duplicate-candidate reporting, and press coalescing in `Sources/Fluid/Models/CarbonHotKeySupport.swift`.
- When FluidVoice is active without AX, local key-down and key-up events must route through `handleFocusedAppPrimaryShortcut`. The local and Carbon paths share press tracking to prevent missed or duplicate triggers.
- Registered Carbon chords arrive in AppKit as `.systemDefined` hotkey events, not ordinary key-down/up. Keep the signature-checked native-event bridge and use the same dispatcher target for permanent and transient registrations.
- Multiple window roots can create managers. Give each manager a distinct Carbon owner signature while retaining action-ID namespaces; return `eventNotHandledErr` for unowned events and remove the handler if registration fails. A failed newer manager must never swallow an older manager's shortcuts.
- Local events must not depend on `NSApp.isActive`, which can lag window changes. Recover missed Carbon releases from physical state, using modifier release when raw key state is unavailable without Input Monitoring. Retain event timestamps so delayed local/Carbon duplicates cannot replay a completed press.
- Escape is a transient Carbon registration while recording or cancellable stop/post-processing is active. Unregister it and clear its press state once both end so idle apps receive Escape normally and later recordings can register it again.
- Escape during AI enhancement must cancel the request and release the foreground stop gates without waiting for provider acknowledgement. Invalidate previews and ignore late results; preserve raw History when enabled. Do not release ASR capture/finalization safety gates early.
- Migrate Right Option to Control-Option-Space only for the exact old persisted default. Never replace a customized shortcut; see `CarbonPrimaryShortcutDefaultMigrated` in `SettingsStore`.

## Code Map

- `fluidApp.swift` starts the app; `AppDelegate.swift` manages startup and shutdown; `ContentView.swift` coordinates dictation completion and output.
- `AppServices.swift` owns the main-actor service graph. `GlobalHotkeyManager.swift` owns AX and Carbon hotkeys. `TypingService.swift` owns AX-gated insertion.
- `Sources/CoreAudioCaptureSupport` is the C capture target used by `DirectCoreAudioInput`.
- `PrivateAIIntegrationService.swift` routes AI post-processing to the private provider; `DictationPostProcessingService.swift` picks between it and remote providers.
- The app targets macOS 15+ and builds through `Fluid.xcodeproj`; Swift package versions are locked in `Package.resolved`.

## Verification And Distribution

- Never commit personal signing-team changes. Build and test unsigned with `CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO`.
- Run `scripts/format-and-lint.sh` for Swift changes. It formats `Sources/` before running strict SwiftLint. If a newer local SwiftFormat rewrites files you did not touch, revert those and keep only your changes.
- MLX needs Xcode's Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`.
- Run tests with `xcodebuild test -project Fluid.xcodeproj -scheme Fluid -destination 'platform=macOS' CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO`.
- Keep the Tiny Whisper E2E test for local validation. CI may skip it because its output is nondeterministic on hosted macOS.
- The local cleanup E2E test downloads and runs the default model only with `TEST_RUNNER_FLUID_LOCAL_CLEANUP_E2E=1`.
- `./build.sh` creates the public unsigned OSS build. `./build.sh fi` depends on the untracked private `build_with_FI_incremental.sh`; do not assume it exists.
- `.github/workflows/personal-macos-dmg.yml` must remain manually dispatched, unsigned, ARM64, retain the hosted Whisper exclusion, and upload `FluidVoice-freedom-arm64.dmg` for 30 days.
