"""MinecraftSurvivalBox: the first Program.

All Minecraft knowledge lives here (and in program-specific policies).
The core runtime never imports from this package.
"""

from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig

__all__ = ["MinecraftSurvivalBox", "SurvivalBoxConfig"]
