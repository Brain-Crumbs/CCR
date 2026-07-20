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
see ``docs/v2/``). Until then it is not on the default path: the heavy
survival-economy modules (``adapter.py``/``world.py``/``rewards.py``, and
-- lazily even from ``adapter.py`` itself -- the profile-driven
``reward_engine.py``/``reward_profile.py``) are only imported once a caller
actually selects ``--world minecraft``, constructs ``MinecraftSurvivalBox``,
or loads a reward profile. ``MinecraftSurvivalBox``/``SurvivalBoxConfig``
are re-exported below via a lazy ``__getattr__`` (PEP 562), not a plain
import, specifically so that importing any lightweight submodule of this
package (e.g. ``programs.minecraft.actions``, which the CLI does import
eagerly -- Python always runs a package's ``__init__`` before any
submodule) does not itself drag in ``adapter.py``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
    from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig

__all__ = ["MinecraftSurvivalBox", "SurvivalBoxConfig"]


def __getattr__(name: str):
    if name == "MinecraftSurvivalBox":
        from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox

        return MinecraftSurvivalBox
    if name == "SurvivalBoxConfig":
        from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig

        return SurvivalBoxConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
