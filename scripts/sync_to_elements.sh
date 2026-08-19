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

[ -d "$DRIVE" ] || { echo "Elements drive not mounted at $DRIVE"; exit 1; }

DRY=()
[ "${1:-}" = "--dry-run" ] && DRY=(--dry-run) && echo "== DRY RUN =="

RS=(rsync -rlt --info=progress2 --human-readable
    --no-perms --no-owner --no-group --modify-window=1 "${DRY[@]}")

cd "$REPO"

echo "== corpora (data/*.pt, recipes, manifests) =="
mkdir -p "$DEST/data"
"${RS[@]}" --include='*.pt' --include='*.yaml' --include='*.txt' --include='*.json' \
           --exclude='*' data/ "$DEST/data/"

echo "== motion clip sets (data/motions/) =="
"${RS[@]}" --delete-excluded --exclude='*_pre_*' --exclude='*.bak*' \
           data/motions/ "$DEST/data/motions/"

echo "== robot assets (USD / MJCF / meshes / textures) =="
mkdir -p "$DEST/protomotions/data/assets"
"${RS[@]}" --exclude='*.bak*' --exclude='*_pre_*' --exclude='*pre_scale*' \
           protomotions/data/assets/ "$DEST/protomotions/data/assets/"

echo "== checkpoints (last + epoch_1000 + configs per run) =="
mkdir -p "$DEST/results"
"${RS[@]}" --include='*/' \
           --include='last.ckpt' --include='epoch_1000.ckpt' \
           --include='config.yaml' --include='resolved_configs*.pt' \
           --include='resolved_configs*.yaml' --include='experiment_config.py' \
           --exclude='*' --prune-empty-dirs results/ "$DEST/results/"

echo
echo "done. $DEST now holds $(du -sh "$DEST" 2>/dev/null | cut -f1)"
