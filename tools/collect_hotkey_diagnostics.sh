#!/bin/sh
set -eu

# Restrict reports to routing metadata; ordinary logs may contain private text.

output="${1:-$HOME/Desktop/FluidVoice-hotkey-diagnostics-$(date +%Y%m%d-%H%M%S).txt}"
{
    printf 'FluidVoice hotkey routing diagnostics\n'
    printf 'Contains shortcut lifecycle/state metadata only, not transcription text.\n\n'
    for log_file in "$HOME/Library/Logs/Fluid/Fluid.log.1" "$HOME/Library/Logs/Fluid/Fluid.log"; do
        if [ -r "$log_file" ]; then
            grep -F '[HotkeyTrace]' "$log_file" || true
        fi
    done
} > "$output"
printf 'Saved hotkey diagnostics to %s\n' "$output"
if [ "$#" -eq 0 ]; then
    open -R "$output"
fi
