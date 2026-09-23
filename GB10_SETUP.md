# go2 on the GB10 — what the bundle's CLAUDE.md could not tell you

The Elements bundle (`go2_gb10/`) was prepared on an x86 box, so it documents
everything except the ARM64/GB10 launch requirements. Those are here. Read the
bundle's `CLAUDE.md` for the actual work (checkpoints, tasks, the open
ball-behind problem); read this for how to make it start.

Set up 2026-09-22. Validated: `go2_masked_mimic_v4` loads and simulates here
(Isaac Sim 6 / Isaac Lab 3, PhysX, aarch64).

## Where things are

| | |
|---|---|
| repo | `~/sparkpack/ProtoMotions` (branch `battle`) |
| interpreter | `~/sparkpack/.venv-isaacsim6/bin/python` (ARM64, torch 2.10+cu130) |
| checkpoints | `results/go2_{tracker_v1,masked_mimic_v4,masked_mimic_v5}` |
| corpus | `data/motions/go2/go2_flat_mirrored_balanced.pt` (426 clips) |

The go2 source came from `origin/battle` with a normal pull; only `results/`
and the corpus came off the Elements drive (both gitignored, so they are not
in git and exist only on this box).

## Use the launcher

    ./run_go2.sh protomotions/inference_agent.py \
      --checkpoint results/go2_masked_mimic_v4/last.ckpt \
      --experiment-path examples/experiments/masked_mimic/transformer.py \
      --simulator isaaclab --physics physx --num-envs 4

`run_go2.sh` sets the environment and execs the ARM venv. Drop `--headless`
for a window. Everything in the bundle's CLAUDE.md works, just prefixed with
the launcher instead of a bare `$PY`.

## The three GB10 traps (all hit during setup)

1. **`LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` is mandatory.** Without
   it Kit prints a *warning* (not an error) and then **hangs forever at 0% CPU
   with no GPU allocation** — it looks like a slow start, it is a dead process.
   Cost 10 minutes of staring at it. The launcher sets this.

2. **`OMNI_KIT_ACCEPT_EULA=YES`.** Otherwise Kit asks "Do you accept the EULA?"
   on stdin and dies with `Unable to bootstrap inner kit kernel: EOF when
   reading a line`. The launcher sets this.

3. **Clear stale carb shm after any `kill -9`**, or the next launch dies
   instantly with exit 139 in `libcarb.tasking.plugin.so`:

       rm -f /dev/shm/carb-RStringInternals-* \
             /dev/shm/sem.carb-RStringInternals-* \
             /dev/shm/sem.carbonite-sharedmemory

   Only when no Kit process is alive. The launcher does this automatically.
   Note `kill -9` is *required* to stop Isaac (SIGTERM wedges it), so this and
   trap 3 always come as a pair.

## Notes

- Use **PhysX** (`--physics physx`). Isaac Lab's Newton path is broken on this
  box (diverges / has frozen the machine). Newton is eval-only here.
- Startup to "Evaluating policy..." takes ~10 minutes cold.
- 4 envs reserved ~83 GB of the GB10's unified memory. Budget accordingly
  before scaling `--num-envs`; the bundle's "2048 envs = 20 GB on a 4090" does
  not translate.
- `pgrep -f inference_agent` matches your own shell wrapper — check
  `ps -eo pid,comm | grep python` instead when verifying a kill.
