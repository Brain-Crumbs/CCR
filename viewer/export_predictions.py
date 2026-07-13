"""Back-compat shim: the exporter now lives in the installed package at
``cognitive_runtime.training.prediction_export`` so the nursery harness and
CLI can use it directly.  This wrapper keeps the documented
``python -m viewer.export_predictions`` / ``from viewer.export_predictions
import ...`` entry points working.  See that module's docstring for the
``pixel-predictions-v1`` format and usage.
"""

from cognitive_runtime.training.prediction_export import (  # noqa: F401
    FULL_MODEL_FORMAT,
    export_prediction_file,
    export_session_predictions,
    load_full_visual_model,
    main,
    save_full_visual_model,
)

if __name__ == "__main__":
    main()
