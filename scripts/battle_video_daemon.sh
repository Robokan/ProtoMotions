#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Battle progression video daemon.
#
# Polls a league run for new policy_*.ckpt snapshots and records each new one
# as a headless real IsaacSim render (see battle_tournament.py --record). Over
# a training run this yields a chronological gallery —
# output/fight_videos/fightNNNN_<snap>_vs_<baseline>.mp4 — so you can scrub
# through how the fighters improve.
#
# Each new snapshot is recorded against a FIXED baseline (the earliest
# snapshot by default), so every clip is the current fighter vs the same
# rookie — the clearest "getting better over time" signal. Override the
# baseline with BASELINE_CKPT.
#
# MEMORY SAFETY: recording spins up a second (small, 2-env inference) Isaac
# process alongside training. On the GB10's shared 128GB that adds pressure,
# and unified-memory exhaustion deadlocks the driver (hard reboot). So the
# daemon SKIPS a snapshot if host MemAvailable is below MIN_GB at tick time,
# and retries next tick. Keep scripts/memory_watchdog.sh armed regardless.
#
# Usage:
#   scripts/battle_video_daemon.sh [run_name] [interval_sec] [container]
# Env overrides:
#   MIN_GB=25            skip recording below this many GB free (default 25)
#   FRAMES=600           clip length in control steps (default 600)
#   BASELINE_CKPT=path   fixed opponent (default: earliest snapshot found)
#   WORKDIR=/workspace/sparkpack/ProtoMotions   repo root inside container
set -uo pipefail

RUN="${1:-soma_battle_league_v3}"
INTERVAL="${2:-3600}"
CONTAINER="${3:-battle}"
MIN_GB="${MIN_GB:-25}"
FRAMES="${FRAMES:-600}"
WORKDIR="${WORKDIR:-/workspace/sparkpack/ProtoMotions}"

RESOLVED="results/${RUN}/resolved_configs_inference.pt"
LEAGUE_GLOB="results/${RUN}/lightning_logs/*/league/policy_*.ckpt"
OUT_DIR="output/fight_videos"
STATE_DIR="${OUT_DIR}/.recorded"

dex() { docker exec "$CONTAINER" bash -c "$1"; }

# Newline-separated snapshot paths, oldest first (mtime order — matches the
# tournament's own adapter ordering).
list_snaps() {
    dex "cd '$WORKDIR' && ls -1tr $LEAGUE_GLOB 2>/dev/null" || true
}

dex "cd '$WORKDIR' && mkdir -p '$STATE_DIR'" || true

echo "video-daemon: run=$RUN interval=${INTERVAL}s min_free=${MIN_GB}GB frames=$FRAMES"
echo "video-daemon: gallery -> $WORKDIR/$OUT_DIR (inside $CONTAINER)"

idx=0
while true; do
    snaps="$(list_snaps)"
    if [ -z "$snaps" ]; then
        echo "video-daemon: no snapshots yet for $RUN; waiting"
        sleep "$INTERVAL"
        continue
    fi

    baseline="${BASELINE_CKPT:-$(echo "$snaps" | head -n1)}"
    base_stem="$(basename "$baseline" .ckpt)"

    # Walk snapshots oldest -> newest so the FIRST run backfills the whole
    # existing history into a chronological gallery, and later runs pick up
    # each newly-produced snapshot. One recording per tick keeps memory
    # pressure bounded; the remaining backlog records on subsequent ticks.
    recorded_one=0
    while IFS= read -r snap; do
        [ -z "$snap" ] && continue
        stem="$(basename "$snap" .ckpt)"
        # Marker keyed by full path so a re-created policy_N in a new version
        # dir still records as its own entry.
        marker="${STATE_DIR}/$(echo "$snap" | tr '/' '_').done"
        if dex "cd '$WORKDIR' && [ -f '$marker' ]"; then
            continue
        fi

        # Memory pre-check (host-side; unified memory is shared with the GPU).
        avail_gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
        if [ "$avail_gb" -lt "$MIN_GB" ]; then
            echo "video-daemon: MemAvailable=${avail_gb}GB < ${MIN_GB}GB — skipping this tick"
            break
        fi

        printf -v n4 '%04d' "$idx"
        out="${OUT_DIR}/fight${n4}_${stem}_vs_${base_stem}.mp4"
        title="${stem} vs ${base_stem}"
        echo "video-daemon: [$(date '+%F %T')] recording $stem vs $base_stem (free=${avail_gb}GB) -> $out"

        if dex "cd '$WORKDIR' && timeout 1200 /workspace/isaaclab/isaaclab.sh -p protomotions/battle_tournament.py \
                --resolved-configs '$RESOLVED' \
                --exhibition '$snap' '$baseline' \
                --record '$out' --record-frames $FRAMES --record-title '$title' \
                --num-envs 2 --headless --deterministic >> '${OUT_DIR}/daemon.log' 2>&1"; then
            dex "cd '$WORKDIR' && touch '$marker'"
            echo "video-daemon: done -> $out"
            idx=$((idx + 1))
            recorded_one=1
        else
            echo "video-daemon: recording FAILED for $stem (see ${OUT_DIR}/daemon.log); will retry next tick"
        fi
        break  # one per tick
    done <<< "$snaps"

    # If the whole backlog is already recorded, wait a full interval for the
    # next snapshot; if we just recorded one and more remain, take a short
    # breather so backfill drains without hammering memory.
    if [ "$recorded_one" -eq 1 ]; then
        sleep 15
    else
        sleep "$INTERVAL"
    fi
done
