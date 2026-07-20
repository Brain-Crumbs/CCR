"""MinecraftSurvivalBox: the original Program -- now legacy/quarantined (issue #176).

All Minecraft knowledge lives here (and in program-specific policies).
The core runtime never imports from this package.

Quarantine note (V2 hardening, issue #176, tracked under #165): this
package is nursery-era legacy. Its elaborate survival economy -- hunger,
crafting, inventory, and the profile-driven ``--reward-profile`` system
(``reward_engine.py``/``reward_profile.py``) -- exists to shape behavior
with crafted rewards, which the V2 predictive objective doesn't need: it's
self-supervised on the world's own future, not on reward. The Crafter
nursery world (``cognitive_runtime.programs.crafter``) is the live V2
nursery and the CLI's default ``--world``.

Minecraft stays fully functional and supported, opt-in via
``--world minecraft`` (or direct construction), and stays in the design as
the eventual *graduation* world -- a first-person environment the organism
graduates into once the nursery loop is a real organism (post Milestone 5;
see ``docs/v2/``). Until then it is not on the default path: nothing here
is imported by the CLI, ``training.nursery``, or any other module unless a
caller actually selects ``--world minecraft`` (or otherwise constructs
``MinecraftSurvivalBox``/loads a reward profile) -- see
``cognitive_runtime/cli.py``'s ``--world`` selector and
``MinecraftSurvivalBox.__init__``'s lazy ``reward_profile`` import.
"""

from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig

__all__ = ["MinecraftSurvivalBox", "SurvivalBoxConfig"]
