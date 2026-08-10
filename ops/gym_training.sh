#!/usr/bin/env bash
# Mechanic drills on the LAN box (192.168.1.130, Ryzen 5 4500 + GTX 1650 4 GB).
#
#   ops/gym_training.sh bank     # rebuild the bank so it carries game state
#   ops/gym_training.sh gyms     # carve it into situations
#   ops/gym_training.sh gate     # DO THIS BEFORE TRAINING -- see below
#   ops/gym_training.sh blocking # drill one mechanic
#
# THE GATE IS NOT OPTIONAL
# ------------------------
# A gym is a filtered start-state distribution, so it can only teach a mechanic
# the world model can represent. `scripts/block_effect.py` measured the JEPA
# model forecasting LESS damage when the defender holds no direction at all --
# the sign of guarding, inverted. Drilling `blocking` against a model in that
# state teaches the agent not to block, efficiently and with a falling loss.
#
# Three objectives failed that gate (spatial AUC 0.540 -> 0.651 -> 0.688, none
# flipping the sign), which is why the corpus was re-captured with the game's
# own state. So: train the world model with `state_coef > 0` on the new labels,
# re-run block_effect, and only then spend GPU on a drill.
#
# WHY THESE SIZES
# ---------------
# Turing has no bf16, so `Config.amp_dtype` must be fp16 here; a bf16 default
# silently falls back to fp32 and halves throughput. The 1650's 4 GB fits
# 2048 rollouts x 4 steps in about 1.7 GB at ~1.9 s/step, which is the measured
# operating point -- RL reads a cached latent bank and only runs the 10M
# predictor, which is why this box can do RL at all while being hopeless for
# world-model training (at 448 px its max batch is 4).
set -euo pipefail

PY="${PY:-$HOME/sb/bin/python}"
ART="${ART:-$HOME/sokubot-art}"
CORPUS="${CORPUS:-$HOME/corpus}"
BANK="${BANK:-$HOME/bank_state.npz}"
GYMS="${GYMS:-$HOME/gyms_state.npz}"
WM="${WM:-$ART/wm_state.pt}"
PROBE="${PROBE:-$ART/reward_probe.npz}"
HORIZON="${HORIZON:-4}"

cd "$HOME/SokuBot"

case "${1:-help}" in
bank)
  # --replays is the whole corpus now; the bank is latents plus labels, not
  # video, so it is bounded by RAM rather than disk.
  "$PY" -m scripts.build_hud_bank --ckpt "$WM" --corpus "$CORPUS" \
        --replays "${REPLAYS:-400}" --out "$BANK" --device cuda
  ;;
gyms)
  "$PY" -m scripts.build_gyms --bank "$BANK" --out "$GYMS" \
        --horizon "$HORIZON" --history 3
  ;;
gate)
  # Does the model represent blocking at all? Anything but "the world model
  # DOES" here means stop and fix the representation, not tune the drill.
  "$PY" -m scripts.block_effect --wm "$WM" --bank "$BANK" \
        --probe "$PROBE" || true
  "$PY" -m scripts.spatial_probe --wm "$WM" --bank "$BANK" || true
  ;;
*)
  GYM="$1"
  # 2048 rollouts = 128 starts x 16 group. Fits 4 GB with headroom; raising
  # --starts is the knob, not --group-size, which is the baseline estimator.
  exec "$PY" -m scripts.train_grpo \
      --wm "$WM" --probe "$PROBE" --bank "$BANK" \
      --gyms "$GYMS" --gym "$GYM" \
      --horizon "$HORIZON" --starts "${STARTS:-128}" --group-size 16 \
      --steps "${STEPS:-20000}" --device cuda \
      --out "$ART/gym_$GYM"
  ;;
esac
