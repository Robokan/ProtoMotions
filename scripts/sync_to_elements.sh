#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Push everything needed to TRAIN ELSEWHERE (e.g. the DGX Spark) onto the WD
# Elements drive. Git carries the code; this carries what git does not:
#
#   data/*.pt              packaged corpora (motion libraries)
#   data/motions/          retargeted .motion clip sets the corpora are built from
#   protomotions/data/assets/  robot USD/MJCF/meshes/textures (mostly untracked)
#   results/<run>/         last.ckpt + epoch_1000.ckpt + configs, per run
#
# The CODE is not copied -- clone it from
# https://github.com/Robokan/ProtoMotions.git (branch: battle).
#
# The drive is NTFS (fuseblk), so permissions/ownership cannot be preserved:
# --no-perms/--no-owner/--no-group avoid per-file errors and
# --modify-window=1 absorbs the 1-second timestamp granularity, without which
# every file looks modified and the whole 30 GB re-copies each run.
#
#   scripts/sync_to_elements.sh --dry-run     # show what would move
#   scripts/sync_to_elements.sh               # do it (incremental, resumable)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVE="${ELEMENTS_DIR:-/mnt/usb-WD_Elements_2621_575846324434334A5246534B-0:0-part1}"
DEST="$DRIVE/ProtoMotionsData"

# The mount POINT is a permanent root-owned directory that exists whether or
# not the drive is plugged in, so `[ -d ]` is not a mount check: with the drive
# absent it passes, and the run then dies on the first mkdir with a permission
# error that looks like a completed run. Demand an actual mounted filesystem,
# and prove it is writable before copying 32 GB at it.
if ! mountpoint -q "$DRIVE"; then
    echo "Elements drive is NOT mounted at $DRIVE"
    echo "Plug it in, or point ELEMENTS_DIR at where it actually mounted:"
    lsblk -o NAME,LABEL,SIZE,MOUNTPOINT | grep -iE 'elements|part1' || true
    exit 1
fi
mkdir -p "$DEST" 2>/dev/null || { echo "Cannot create $DEST (read-only or no permission)"; exit 1; }
[ -w "$DEST" ] || { echo "$DEST is not writable -- NTFS may have mounted read-only after an unclean eject"; exit 1; }

DRY=()
[ "${1:-}" = "--dry-run" ] && DRY=(--dry-run) && echo "== DRY RUN =="

# --inplace writes straight into the destination file instead of rsync's
# default "write .tmp then rename". The drive is NTFS, and a directory whose
# index is damaged accepts writes but fails EVERY rename with EIO -- which is
# how a whole stage can fail while the disk looks fine. --inplace also avoids
# a second full-size temp copy, which matters for 173 MB checkpoints.
RS=(rsync -rlt --inplace --info=progress2 --human-readable
    --no-perms --no-owner --no-group --modify-window=1 "${DRY[@]}")

cd "$REPO"

# Each stage is independent, so a failure must not abort the ones after it:
# with `set -e` a corrupt data/ directory took the whole script down before it
# ever reached the checkpoints, silently, looking like a clean run.
FAILED=()
stage() {  # stage <label> <rsync args...>
    local label="$1"; shift
    echo "== $label =="
    "${RS[@]}" "$@" || { FAILED+=("$label"); echo "!! STAGE FAILED: $label"; }
}

# Checkpoints go FIRST: they are the payload that cannot be regenerated from
# git, and they used to sit behind three stages that could fail ahead of them.
mkdir -p "$DEST/results"
stage "checkpoints (last + epoch_1000 + configs per run)" \
      --include='*/' \
      --include='last.ckpt' --include='epoch_1000.ckpt' \
      --include='config.yaml' --include='resolved_configs*.pt' \
      --include='resolved_configs*.yaml' --include='experiment_config.py' \
      --exclude='*' --prune-empty-dirs results/ "$DEST/results/"

mkdir -p "$DEST/data"
stage "corpora (data/*.pt, recipes, manifests)" \
      --include='*.pt' --include='*.yaml' --include='*.txt' --include='*.json' \
      --exclude='*' data/ "$DEST/data/"

stage "motion clip sets (data/motions/)" \
      --delete-excluded --exclude='*_pre_*' --exclude='*.bak*' \
      data/motions/ "$DEST/data/motions/"

mkdir -p "$DEST/protomotions/data/assets"
stage "robot assets (USD / MJCF / meshes / textures)" \
      --exclude='*.bak*' --exclude='*_pre_*' --exclude='*pre_scale*' \
      protomotions/data/assets/ "$DEST/protomotions/data/assets/"

echo
echo "== verify: newest local run vs the copy on the drive =="
# Only last.ckpt and epoch_1000.ckpt are copied per run -- the epoch_N series
# stays behind. Print the newest run both sides so a silent no-op is visible.
NEWEST="$(ls -1dt results/*/ 2>/dev/null | head -1 | xargs -r basename)"
if [ -n "$NEWEST" ]; then
    echo "newest local run: $NEWEST"
    ls -l "$DEST/results/$NEWEST/" 2>&1 | tail -n +2 || echo "  MISSING on drive"
fi

echo
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "!! ${#FAILED[@]} STAGE(S) FAILED: ${FAILED[*]}"
    echo "   Repeated 'Input/output error' means a damaged NTFS directory index."
    echo "   Fix: unmount, then 'sudo ntfsfix -d /dev/sdb1' (or chkdsk /f on Windows)."
    echo "$DEST holds $(du -sh "$DEST" 2>/dev/null | cut -f1)"
    exit 1
fi
echo "done. $DEST now holds $(du -sh "$DEST" 2>/dev/null | cut -f1)"
