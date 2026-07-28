"""Crafter's compact semantic vocabulary is stable and action-relevant."""

import numpy as np

from cognitive_runtime.programs.crafter.streams import (
    FRAME_LEGEND,
    RAW_TO_SEMANTIC_ID,
    SEMANTIC_CLASS_IDS,
    SEMANTIC_CLASSES,
    SEMANTIC_VOCABULARY_VERSION,
    crop_semantic_grid,
)


def test_crafter_semantic_grid_uses_six_contiguous_action_classes():
    assert SEMANTIC_CLASSES == (
        "ground", "water", "solid", "resource", "entity", "agent",
    )
    assert FRAME_LEGEND == dict(enumerate(SEMANTIC_CLASSES))
    assert set(RAW_TO_SEMANTIC_ID.values()) == set(range(len(SEMANTIC_CLASSES)))
    assert SEMANTIC_VOCABULARY_VERSION


def test_raw_crafter_ids_are_compacted_at_the_stream_boundary():
    # raw ids: void, water, grass, stone, tree, lava, table, player, cow
    raw = np.array([[value] * 9 for value in (0, 1, 2, 3, 6, 7, 11, 13, 14)], dtype=np.int64)

    compact = crop_semantic_grid(raw, position=(4, 4), radius=4)

    assert [row[4] for row in compact] == [
        SEMANTIC_CLASS_IDS["ground"],
        SEMANTIC_CLASS_IDS["water"],
        SEMANTIC_CLASS_IDS["ground"],
        SEMANTIC_CLASS_IDS["solid"],
        SEMANTIC_CLASS_IDS["resource"],
        SEMANTIC_CLASS_IDS["solid"],
        SEMANTIC_CLASS_IDS["solid"],
        SEMANTIC_CLASS_IDS["agent"],
        SEMANTIC_CLASS_IDS["entity"],
    ]
