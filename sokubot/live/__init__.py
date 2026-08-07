"""Live control: the loop that lets the policy play a real match.

Everything the agent may observe enters through :mod:`sokubot.live.capture`
(window pixels) and everything it may do leaves through
:mod:`sokubot.live.pad` (a virtual gamepad). Nothing here reads game memory,
and nothing here may start to -- see ``docs/HANDOFF.md`` section 8.
"""
