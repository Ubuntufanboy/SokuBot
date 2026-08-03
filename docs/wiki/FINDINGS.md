# What the wiki confirmed and contradicted

Source: hisouten.koumakan.jp HUD (oldid 19513) and Weather (oldid 26276), saved
in this directory. The live site serves generated filler to non-browser clients
-- HTTP 200, ~1 KB, no game content -- so these local copies are the reference.

## Confirmed

* **"Red Life - Shows how much damage the current combo has done."** The red
  channel means what `hud.combo_size` assumed. It was shipped marked unvalidated;
  the semantics are now sourced, though the reading itself is still only checked
  against invariants, not ground truth.
* **Win Counter** exists as its own indicator -- a round-boundary signal
  independent of the life bar.
* **"All lit cards are used up when casting the currently selected card."**
  Explains why every slot greys out on activation.
* **Golden border = the card is currently usable**, which is the cost signal:
  the spirit level where the border lights is that card's cost.

## Contradicted -- weather changes health with no damage dealt

`damage_dealt = opponent's health drop` is wrong under at least three weathers:

| weather | effect |
|---|---|
| Calm | first to land a hit gains a spotlight and **regenerates life** until hit |
| Scorching Sun | attack power rises with altitude "at the cost of HP, which will drain more quickly the higher you are" |
| Heavy Fog | "Vampirism. 50% of all damage to the opponent is transferred to the **attacker's** life bar" |

Under Scorching Sun the agent would be paid for an opponent flying high. Under
Heavy Fog it would be paid *and* its own bar would rise, reading as phantom
regeneration. Rewards must be weather-conditional.

Weather is identifiable: `WeatherName001..021.png` here are the sprites the HUD
renders in the timer circle, so a template match against that region names the
current weather.

## Also worth encoding

* **Spirit orbs turn red when "crushed"** and are unusable until restored. Guard
  crush is orbs going *red*, not merely absent -- the current detector counts
  blue orbs, which measures usable spirit but does not separate crushed from
  spent.
* **Limit Seal** ends any combo when it fills, which bounds combo length
  independently of what either player does.
* Other weathers touch spirit rather than health (Hail doubles spirit recovery,
  Sunshower makes crushed orbs recover faster, Sunny stops orb regeneration on
  Border Escape).
