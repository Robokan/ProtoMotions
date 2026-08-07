#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Back up everything training needs that git does NOT hold, to the WD Elements
# drive under ProtoMotionsData/, mirroring repo-relative paths so restoring is
# a straight rsync back into a fresh clone.
#
# WHAT GIT ALREADY HAS IS NOT COPIED. Corpora, retargeted motion sets,
# generated USD and results/ are untracked or gitignored -- those are the
# irreplaceable artifacts. Tracked source is skipped.
#
# ---------------------------------------------------------------------------
# ONE-TIME MIGRATION
#
# The drive already carries atlas_tracker_4090_resume/ and
# t800_tracker_4090_resume/ (25 GB) holding corpora, results and 16 GB of
# retarget sources. --migrate-resume MOVES those into their repo-relative
# homes under ProtoMotionsData/ so there is one canonical location and nothing
# is stored twice:
#
#   <name>_resume/data/*            -> ProtoMotionsData/data/
#   <name>_resume/results/*         -> ProtoMotionsData/results/
#   <name>_resume/retarget_sources/*-> ProtoMotionsData/retarget_sources/
#   <name>_resume/code, README.md   -> ProtoMotionsData/legacy/<name>/
#
# code/ is a snapshot of the repo at that run, so it is preserved rather than
# merged -- the working tree is what git is for. The READMEs carry the resume
# instructions for those runs and are kept beside them.
#
# Moves are renames within one filesystem: instant, no data rewritten. Every
# move is appended to ProtoMotionsData/MIGRATION_MANIFEST.tsv so it can be
# undone. Run it ONCE, with --dry-run first.
#
#   ./data/scripts/backup_to_elements.sh --migrate-resume --dry-run
#   ./data/scripts/backup_to_elements.sh --migrate-resume
#   ./data/scripts/backup_to_elements.sh                  # the routine backup
#   ./data/scripts/backup_to_elements.sh --lean           # skip old ckpts
#
# Re-running the backup is cheap: rsync sends only what changed.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST_NAME="ProtoMotionsData"

LEAN=0; DRY=""; MIGRATE=0
for a in "$@"; do
  case "$a" in
    --lean)           LEAN=1 ;;
    --dry-run)        DRY="--dry-run" ;;
    --migrate-resume) MIGRATE=1 ;;
    -h|--help)        sed -n '3,44p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- find drive
# Located by label, not hardcoded: the mount path embeds the device serial and
# differs between machines.
ELEMENTS="$(ls -d /mnt/*Elements* /media/*/*Elements* 2>/dev/null | head -1)"
if [[ -z "$ELEMENTS" || ! -d "$ELEMENTS" ]]; then
  echo "ERROR: Elements drive not mounted (looked for *Elements* under /mnt and /media)." >&2
  exit 1
fi
DEST="$ELEMENTS/$DEST_NAME"
MANIFEST="$DEST/MIGRATION_MANIFEST.tsv"

echo "repo : $REPO"
echo "drive: $ELEMENTS"
echo "dest : $DEST"
echo "free : $(df -h "$ELEMENTS" | awk 'NR==2{print $4}')"
[[ -n "$DRY" ]] && echo "MODE : DRY RUN -- nothing will be written or moved"
echo

# An unwritable mount would otherwise give a long, quiet run that saves nothing.
if [[ -z "$DRY" ]]; then
  if ! touch "$ELEMENTS/.pmdata_write_test" 2>/dev/null; then
    echo "ERROR: $ELEMENTS is not writable (read-only mount?)." >&2; exit 1
  fi
  rm -f "$ELEMENTS/.pmdata_write_test"
  mkdir -p "$DEST"
fi

# ------------------------------------------------------------- migrate step
migrate() {
  local moved=0 conflict=0
  [[ -z "$DRY" ]] && { mkdir -p "$DEST"; [[ -f "$MANIFEST" ]] || printf 'src\tdst\n' > "$MANIFEST"; }

  for r in "$ELEMENTS"/*_resume; do
    [[ -d "$r" ]] || continue
    local name; name="$(basename "$r")"
    echo ">> $name"

    # data/, results/, retarget_sources/ merge into the mirrored layout
    for sub in data results retarget_sources; do
      [[ -d "$r/$sub" ]] || continue
      for item in "$r/$sub"/*; do
        [[ -e "$item" ]] || continue
        local base tgt; base="$(basename "$item")"; tgt="$DEST/$sub/$base"
        if [[ -e "$tgt" ]]; then
          # Never clobber. A same-name file already in place is reported and
          # left alone in BOTH locations for you to reconcile.
          echo "    CONFLICT, left in place: $sub/$base"
          conflict=$((conflict+1)); continue
        fi
        if [[ -n "$DRY" ]]; then
          echo "    would move $sub/$base"
        else
          mkdir -p "$DEST/$sub"
          if mv -n "$item" "$tgt" 2>/dev/null; then
            printf '%s\t%s\n' "$item" "$tgt" >> "$MANIFEST"
            echo "    moved $sub/$base"
          else
            echo "    FAILED to move $sub/$base" >&2; continue
          fi
        fi
        moved=$((moved+1))
      done
    done

    # code/ and README.md are per-run snapshots: preserved, not merged
    for keep in code README.md; do
      [[ -e "$r/$keep" ]] || continue
      local tgt="$DEST/legacy/$name/$keep"
      if [[ -e "$tgt" ]]; then echo "    CONFLICT, left in place: legacy/$name/$keep"; conflict=$((conflict+1)); continue; fi
      if [[ -n "$DRY" ]]; then
        echo "    would move $keep -> legacy/$name/"
      else
        mkdir -p "$DEST/legacy/$name"
        if mv -n "$r/$keep" "$tgt" 2>/dev/null; then
          printf '%s\t%s\n' "$r/$keep" "$tgt" >> "$MANIFEST"
          echo "    moved $keep -> legacy/$name/"
        fi
      fi
      moved=$((moved+1))
    done

    # only remove the shell if it actually emptied
    if [[ -z "$DRY" ]] && [[ -d "$r" ]] && [[ -z "$(ls -A "$r" 2>/dev/null)" ]]; then
      rmdir "$r" && echo "    removed empty $name/"
    elif [[ -z "$DRY" ]] && [[ -n "$(ls -A "$r" 2>/dev/null)" ]]; then
      echo "    $name/ still holds: $(ls -A "$r" | tr '\n' ' ')"
    fi
  done
  echo
  echo "migration: $moved item(s) moved, $conflict conflict(s)"
  [[ $conflict -gt 0 ]] && echo "  conflicts were NOT overwritten -- reconcile by hand"
  [[ -z "$DRY" ]] && echo "  undo log: $MANIFEST"
  echo
}

if [[ $MIGRATE -eq 1 ]]; then
  migrate
  [[ -n "$DRY" ]] && { echo "dry run: stopping before backup."; exit 0; }
fi

# ------------------------------------------------------------------ transfer
copy() {                       # copy <repo-relative path> [extra rsync args]
  local rel="$1"; shift
  local src="$REPO/$rel"
  if [[ ! -e "$src" ]]; then echo "  (skip, absent) $rel"; return 0; fi
  echo ">> $rel"
  [[ -z "$DRY" ]] && mkdir -p "$DEST/$(dirname "$rel")"
  rsync -a --info=stats2 --human-readable --partial $DRY "$@" \
    "$src" "$DEST/$(dirname "$rel")/" 2>&1 | tail -3
}

# 1. corpora training loads directly
echo ">> data/*.pt"
[[ -z "$DRY" ]] && mkdir -p "$DEST/data"
rsync -a --info=stats2 --human-readable --partial $DRY \
  --include='*.pt' --exclude='*' "$REPO/data/" "$DEST/data/" 2>&1 | tail -3

# 2. retargeted motion sets: expensive, and need the source FBX to rebuild
copy "data/motions"

# 3. pretrained bundles the configs reference
copy "data/pretrained_models"

# 4. generated assets. usd/atlas and mesh/Atlas are gitignored outright;
#    overlay/ holds 91 untracked character USDs (the skinned meshes).
copy "protomotions/data/assets/usd"
copy "protomotions/data/assets/mesh"
copy "protomotions/data/assets/overlay"

# 5. results: 93 GB in full. --lean keeps only what a resume needs
#    (last.ckpt + resolved configs + tensorboard + logs), roughly 20 GB.
if [[ $LEAN -eq 1 ]]; then
  echo ">> results (lean: last.ckpt, configs, tensorboard, logs)"
  [[ -z "$DRY" ]] && mkdir -p "$DEST/results"
  rsync -a --info=stats2 --human-readable --partial $DRY \
    --include='*/' --include='last.ckpt' --include='resolved_configs*.pt' \
    --include='*.log' --include='events.out.tfevents.*' \
    --include='*.yaml' --include='*.json' --exclude='*' \
    "$REPO/results/" "$DEST/results/" 2>&1 | tail -3
else
  copy "results"
fi

# 6. retarget inputs that live OUTSIDE the repo and cannot be rebuilt from it
if [[ -d "$HOME/sparkpack/UnrealExportedAssets" ]]; then
  echo ">> UnrealExportedAssets (external retarget source)"
  [[ -z "$DRY" ]] && mkdir -p "$DEST/retarget_sources"
  rsync -a --info=stats2 --human-readable --partial $DRY \
    "$HOME/sparkpack/UnrealExportedAssets" "$DEST/retarget_sources/" 2>&1 | tail -3
fi

# -------------------------------------------------------------------- report
echo
if [[ -z "$DRY" ]]; then
  echo "on drive: $(du -sh "$DEST" 2>/dev/null | cut -f1)  at $DEST"
  echo "free    : $(df -h "$ELEMENTS" | awk 'NR==2{print $4}')"
  broken=$(find "$DEST" -xtype l 2>/dev/null | wc -l)
  [[ "$broken" -gt 0 ]] && { echo "BROKEN SYMLINKS: $broken"; find "$DEST" -xtype l | sed 's/^/    /'; }
fi
echo "done."
