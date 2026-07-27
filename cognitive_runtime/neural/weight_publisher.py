"""Compatibility shim for :mod:`sleep.weight_publisher`.

Weight hand-off is part of sleep consolidation as of Phase 5.  The old neural
module path remains supported for existing integrations.
"""

from sleep.weight_publisher import WeightPublisher, WeightSubscriber

__all__ = ["WeightPublisher", "WeightSubscriber"]
