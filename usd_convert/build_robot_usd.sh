#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Build a robot's USD from its MJCF, end to end, in one command:
#
#     usd_convert/build_robot_usd.sh atlas
#
# The conversion proper is only step 2 of 6. The other five used to live in
# script docstrings and commit messages, and skipping any of them produces an
# asset that LOADS and then misbehaves in a way that looks like a physics or
# training bug:
#
#   - wrong --output-dir  -> writes usd/<stem>_flat/ while the robot config
#                            reads usd/<robot>/, so training silently keeps
#                            using the OLD body (see the fossil at
#                            protomotions/data/assets/usd/config.yaml)
#   - no contact excludes -> the Isaac Lab MJCF converter DROPS MuJoCo's
#                            <contact><exclude> pairs, so limbs that the MJCF
#                            says never collide do collide in Isaac
#   - no worldbody strip  -> two prims carry ArticulationRootAPI and the sim
#                            dies with "Failed to find a single articulation"
#
# Most robots' USD is committed; ATLAS is gitignored (.gitignore:29), so a
# fresh checkout has its MJCF but not its USD and must rebuild or restore it.
#
#     usd_convert/build_robot_usd.sh --list          # what can be built
#     usd_convert/build_robot_usd.sh atlas --dry-run # print the commands only
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Isaac Sim 5 / IsaacLab 2.3.2. NOT .venv-isaacsim6: IsaacLab 3.0 removed
# PhysxCfg, which the converter imports.
PY="${ISAACSIM5_PYTHON:-$HOME/sparkpack/.venv-isaacsim5/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES

MJCF_DIR="protomotions/data/assets/mjcf"
USD_DIR="protomotions/data/assets/usd"

# Robots whose MJCF is the source of truth. Anything not listed here either has
# no MJCF (anymal_d, go2 -- imported as USD) or was never run through the
# flatten step (samurai).
KNOWN="atlas t800 tiger raptor utahraptor"

usage() { echo "usage: $0 <robot> [--dry-run]   (robots: $KNOWN)"; }
[ "${1:-}" = "--list" ] && { echo "$KNOWN"; exit 0; }
[ $# -ge 1 ] || { usage; exit 1; }

ROBOT="$1"; shift
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY=1

case " $KNOWN " in *" $ROBOT "*) ;; *) echo "unknown robot: $ROBOT"; usage; exit 1;; esac

MJCF="$MJCF_DIR/$ROBOT.xml"
FLAT="$MJCF_DIR/${ROBOT}_flat.xml"
OUT="$USD_DIR/$ROBOT"
PHYSICS="$OUT/configuration/${ROBOT}_flat_physics.usd"
EXPECTED="$OUT/${ROBOT}_flat.usda"   # must match robot_configs/<robot>.py

[ -f "$MJCF" ] || { echo "no MJCF at $MJCF"; exit 1; }
[ -x "$PY" ] || { echo "no isaacsim5 python at $PY (set ISAACSIM5_PYTHON)"; exit 1; }

run() {
    echo "+ $*"
    [ -n "$DRY" ] || "$@"
}

echo "=== [1/6] flatten MJCF (resolve <default> classes, freejoint -> free) ==="
run "$PY" usd_convert/flatten_mjcf.py "$MJCF"

echo
echo "=== [2/6] MJCF -> USD ==="
# --output-dir is NOT optional. The default derives from the MJCF stem, which
# is "<robot>_flat", giving usd/<robot>_flat/ -- a directory no config reads.
run "$PY" usd_convert/convert_robot_mjcf_to_usda.py "$FLAT" --output-dir "$OUT"

echo
echo "=== [3/6] material / visual patches (robot-specific) ==="
case "$ROBOT" in
    atlas)
        run "$PY" usd_convert/patch_atlas_usd_bindings.py
        ;;
    t800)
        run "$PY" usd_convert/patch_t800_usd_bindings.py
        run "$PY" usd_convert/hide_t800_collision_visuals.py
        ;;
    *)
        echo "  (none for $ROBOT)"
        ;;
esac

echo
echo "=== [4/6] re-apply MJCF contact excludes (converter drops them) ==="
run "$PY" usd_convert/apply_mjcf_contact_excludes.py --mjcf "$MJCF" --usd "$PHYSICS"

echo
echo "=== [5/6] strip the duplicate worldbody articulation root ==="
run "$PY" usd_convert/strip_worldbody_articulation.py --usd "$PHYSICS"

echo
echo "=== [6/6] verify ==="
[ -n "$DRY" ] && { echo "(dry run -- nothing built)"; exit 0; }

FAIL=0
# The trap check: the asset must exist at the path the robot config names.
if [ -f "$EXPECTED" ]; then
    echo "  OK   $EXPECTED"
else
    echo "  FAIL $EXPECTED does not exist -- did --output-dir land elsewhere?"
    FAIL=1
fi
CFG="protomotions/robot_configs/$ROBOT.py"
if [ -f "$CFG" ]; then
    WANT="$(grep -oE 'usd_asset_file_name="[^"]+"' "$CFG" | head -1 | cut -d'"' -f2)"
    if [ -z "$WANT" ]; then
        echo "  WARN $CFG names no usd_asset_file_name"
    elif [ -f "protomotions/data/assets/$WANT" ]; then
        echo "  OK   $CFG wants assets/$WANT -- present"
    else
        echo "  FAIL $CFG wants assets/$WANT, which is missing"
        FAIL=1
    fi
fi
# Stray provenance at the assets root means someone passed the parent dir.
if [ -f "$USD_DIR/config.yaml" ]; then
    echo "  WARN $USD_DIR/config.yaml exists -- leftover from an --output-dir"
    echo "       mistake (harmless, but it is not provenance for any robot)"
fi
if [ "$ROBOT" = "atlas" ]; then
    echo "  -- Atlas joint audit (expect revolute=30 d6=0 with_drive=30):"
    run "$PY" usd_convert/inspect_atlas_joints.py || FAIL=1
fi

echo
[ "$FAIL" -eq 0 ] && echo "$ROBOT USD build COMPLETE" || { echo "$ROBOT USD build FAILED verification"; exit 1; }
