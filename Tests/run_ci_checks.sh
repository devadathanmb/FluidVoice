#!/bin/bash
# The same regression and Xcode test sequence locally and on GitHub Actions.
set -euo pipefail
cd "$(dirname "$0")/.."

task_developer_dir="${DEVELOPER_DIR:-$(xcode-select -p)}"
if [ ! -d "$task_developer_dir/Platforms/MacOSX.platform" ]; then
    echo "Full Xcode is required; set DEVELOPER_DIR to an installed Xcode.app." >&2
    exit 1
fi
export DEVELOPER_DIR="$task_developer_dir"
task_result_dir="${CI_RESULT_DIR:-build/test-results/local-ci-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$task_result_dir"

task_tree_fingerprint() {
    python3 - <<'PY'
import hashlib, os, pathlib, subprocess
paths = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).split(b'\0')
digest = hashlib.sha256()
for raw in sorted(set(paths) - {b''}):
    path = pathlib.Path(os.fsdecode(raw))
    digest.update(raw + b'\0')
    if path.is_symlink():
        digest.update(b'link:' + os.fsencode(os.readlink(path)))
    elif path.is_file():
        digest.update(str(path.stat().st_mode & 0o777).encode() + b':')
        digest.update(path.read_bytes())
    else:
        digest.update(b'deleted')
print(digest.hexdigest())
PY
}
task_start_tree="$(task_tree_fingerprint)"
{
    git rev-parse HEAD
    echo "tree=$task_start_tree"
    sw_vers
    xcodebuild -version
} > "$task_result_dir/environment.txt"

task_phase="preflight"
trap 'task_status=$?; if [ "$task_status" != 0 ]; then echo "CI check failed: $task_phase (exit $task_status); results: $task_result_dir" >&2; fi' EXIT
run_check() {
    task_phase="$1"
    shift
    echo "Running: $task_phase"
    "$@"
}

# CI already gates this job on its Linux SwiftLint job. Local runs must perform
# that check too; formatting validation must not silently edit the tested tree.
if [ "${GITHUB_ACTIONS:-false}" != "true" ]; then
    task_changed_swift=()
    while IFS= read -r task_file; do
        task_changed_swift+=("$task_file")
    done < <(scripts/changed-swift-files.sh)
    if [ "${#task_changed_swift[@]}" -gt 0 ]; then
        run_check formatting swiftformat --lint --config .swiftformat "${task_changed_swift[@]}"
    fi
    run_check lint swiftlint --strict --config .swiftlint.yml Sources Tests Package.swift
fi
run_check text-normalization sh Tests/TextNormalization/run.sh
run_check clipboard sh Tests/run_clipboard_backup_reproduction.sh
run_check clipboard-read-receipts sh Tests/run_paste_read_receipt_tests.sh
run_check paste-events sh Tests/run_paste_command_event_tests.sh
run_check keyboard-layouts sh Tests/run_paste_key_cache_tests.sh
run_check paste-recovery sh Tests/run_paste_recovery_policy_tests.sh
run_check accessibility-preparation sh Tests/run_dictation_accessibility_preparation_tests.sh
run_check onboarding sh Tests/run_onboarding_tests.sh
run_check private-ai-settings sh Tests/run_private_ai_settings_tests.sh
run_check release-highlights sh Tests/run_release_highlights_tests.sh
run_check history-decode sh Tests/run_history_decode_recovery_tests.sh
run_check dictionary-correction-archive sh Tests/run_dictionary_correction_archive_tests.sh
run_check dictionary-correction-timing python3 Tests/run_dictionary_correction_timing_tests.py
run_check settings-backup python3 Tests/run_settings_backup_concurrency_tests.py
run_check meeting-backup python3 Tests/run_meeting_backup_refresh_tests.py
run_check meeting-end-countdown python3 Tests/run_meeting_end_countdown_tests.py
run_check meeting-picker python3 Tests/run_meeting_picker_tests.py
run_check meeting-model-policy python3 Tests/run_meeting_model_policy_tests.py
run_check meeting-computer-audio python3 Tests/run_meeting_computer_audio_tests.py

task_phase="Xcode build and tests"
task_build_args=(test -project Fluid.xcodeproj -scheme Fluid
    -destination "platform=macOS,arch=$(uname -m)"
    -resultBundlePath "$task_result_dir/Fluid.xcresult")
while IFS= read -r task_skipped_test; do
    task_build_args+=("-skip-testing:$task_skipped_test")
done < <(grep -Ev '^[[:space:]]*(#|$)' Tests/ci-skipped-tests.txt)
# The fork builds and tests unsigned. Signing a local test host makes codesign
# ask for the login keychain password.
task_build_args+=(CODE_SIGN_IDENTITY= CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO)
if command -v xcpretty >/dev/null 2>&1; then
    xcodebuild "${task_build_args[@]}" 2>&1 | tee "$task_result_dir/xcodebuild.log" | xcpretty
else
    xcodebuild "${task_build_args[@]}" 2>&1 | tee "$task_result_dir/xcodebuild.log"
fi

task_phase="source tree verification"
if [ "$task_start_tree" != "$(task_tree_fingerprint)" ]; then
    echo "Source tree changed during validation. Rerun the gate on the final tree." >&2
    exit 1
fi
echo "PASS: complete CI sequence; tree=$task_start_tree" | tee "$task_result_dir/result.txt"
