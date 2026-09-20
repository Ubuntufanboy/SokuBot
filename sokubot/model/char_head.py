"""Which character is in each screen row, read at the place it is standing.

WHY THE OBVIOUS HEAD DOES NOT WORK
----------------------------------
The first version of this was forty extra logits hanging off the encoder's
final linear layer, sharing it with twenty state regressions. Measured on a
100-replay corpus it reached per-row character accuracy 0.46 and an identity
decision of 0.4987 -- chance -- and the diagnosis was not "identity is absent
from the pixels":

    predicted PAIR (unordered) correct   0.474
    predicts the SAME character twice    0.388   (no true pair is a mirror)
    given the pair correct, order right  0.542   (chance 0.5)

A classifier that cannot reliably name WHICH TWO characters are on screen is
not evidence about which one is on the left. The readout was underpowered, and
this project has already paid for that mistake three times -- velocity called
unlearnable from a loss that never funded it, projectile position nearly called
resolution-limited from a 514-parameter probe.

THE TWO THINGS THIS FIXES
-------------------------
**Capacity, and the right pooling.** Character identity is an APPEARANCE
question, and the encoder's spatial head is a soft-argmax built for
COORDINATES: it reduces the feature map to K (x, y) pairs plus a global mean.
A global mean is exactly the operation that destroys "which of the two". So
this pools the feature map twice, under two learned spatial attention maps, and
classifies each pooled vector separately.

**Left/right by construction, not by learning.** The two attention maps are
ORDERED BY THEIR OWN x CENTROID before classification, so the first output is
always the leftmost thing the head attended to. The network cannot mix up which
row is which, because the ordering is computed rather than predicted -- which
is the whole failure of the `p1_left` output, stuck at 0.52 across seven arms.

That leaves the head to answer only "what is standing here", which is the part
that is genuinely visible.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharHead(nn.Module):
    """Feature map -> (2, n_char) logits, ordered left row first."""

    def __init__(self, ch: int, n_char: int, width: int = 256):
        super().__init__()
        # Two attention maps: one per character on screen. Deliberately 2 and
        # not K -- the question has exactly two answers and a wider head would
        # have to learn to merge them.
        self.attn = nn.Conv2d(ch, 2, 1)
        self.cls = nn.Sequential(
            nn.Linear(ch, width), nn.GELU(), nn.Linear(width, n_char))

    def forward(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (logits [B, 2, n_char], centroid_x [B, 2]) with row 0 leftmost."""
        B, C, H, W = f.shape
        a = self.attn(f).flatten(2).softmax(-1).view(B, 2, H, W)   # [B,2,H,W]
        xs = torch.linspace(-1, 1, W, device=f.device).view(1, 1, 1, W)
        cx = (a * xs).sum((2, 3))                                   # [B,2]
        # Pool the FEATURES under each attention map: this is what carries
        # appearance, and it is not averaged over the whole frame.
        pooled = torch.einsum("bkhw,bchw->bkc", a, f)               # [B,2,C]
        # Order by centroid so index 0 is always the left one. argsort is not
        # differentiable but it does not need to be -- it permutes, and the
        # gradient flows to whichever slot each map ended up in.
        order = cx.argsort(dim=1)
        pooled = torch.gather(pooled, 1, order[..., None].expand(-1, -1, C))
        cx = torch.gather(cx, 1, order)
        return self.cls(pooled), cx
