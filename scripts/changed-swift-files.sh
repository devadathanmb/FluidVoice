#!/bin/bash
# Print Swift files under Sources/ that differ from the remote-tracking branch
# (or FORMAT_BASE), including uncommitted and untracked files. Upstream's code is
# not SwiftFormat-clean, so formatting is applied only to files the fork changes.
set -euo pipefail
cd "$(dirname "$0")/.."

base="${FORMAT_BASE:-}"
if [ -z "$base" ]; then
  if upstream_ref="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"; then
    base="$(git merge-base HEAD "$upstream_ref")"
  else
    base=HEAD
  fi
fi

{
  git diff --name-only --diff-filter=d "$base" -- 'Sources/*.swift'
  git ls-files --others --exclude-standard -- 'Sources/*.swift'
} | sort -u
