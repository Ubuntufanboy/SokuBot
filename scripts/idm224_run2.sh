#!/bin/bash
# Inverse-dynamics run 2. ONLY LAUNCH IF THE GATE SAYS RUN 1 IS NOT ENOUGH --
# if block_effect already shows the model predicting that guarding reduces
# damage, the next move is the blocking gym, not more world-model work.
#
# ONE CHANGE: --idm-pos-weight 9.0
#
# Run 1 left the inverse-dynamics term as the largest thing in the objective and
# almost stationary:
#
#     step 20000   pred 0.0215   idm 0.251
#     step 30000   pred 0.0149   idm 0.284      (chance is 0.3025)
#
# Fifteen to twenty times the prediction loss, and only ~15% of the way from
# chance to zero. That is not a term to turn up -- raising idm_coef amplifies a
# gradient with nowhere to go, which is why the earlier "raise the coefficient"
# recommendation was retracted.
#
# 9.0 is not tuned. A human holds 9.85% of buttons, so (1 - p) / p = 9.15 is the
# weight at which pressed and released contribute equally, and rounding it is
# the only judgement involved. At 1.0 the gradient is mostly about correctly
# saying "not pressed" nine times in ten.
#
# WHAT RUN 2 CAN ANSWER THAT RUN 1 COULD NOT
# ------------------------------------------
# Run 1 could not distinguish "the head predicts released everywhere" from "the
# head is learning slowly", because `idm_acc` was added to the printer after it
# launched. Run 2 prints it. It also prints `mirror_play` every eval and writes
# `best_spatial.pt`, so the checkpoint that is best at the thing the agent needs
# is no longer overwritten by the one that is best at skill -- which is how the
# 0.764 model from step 4000 was lost.
#
# Read `idm_acc` first. Near 0.5 means the head is degenerate and the fix is the
# loss or the head, not the coefficient. Rising means the balance worked.
cd /root/SokuBot
exec /venv/main/bin/python -m scripts.train_full \
  --corpus /root/corpus --image-size 224 \
  --steps 60000 --batch-size 128 --num-workers 12 --shuffle-gb 18 \
  --eval-every 4000 --ckpt-every 4000 \
  --warmup 2000 --lr 2e-4 \
  --idm-coef 1.0 --idm-pos-weight 9.0 --cf-coef 0.0 --hud-coef 0.0 \
  --ckpt-dir /root/ckpt_idm224_v2 --log /root/train_idm224_v2.json --device cuda
