from __future__ import annotations

import pytest

from cognitive_runtime.training.prediction_export import ExperimentIdentity, experiment_directory


def test_experiment_directory_requires_explicit_validated_resume(tmp_path):
    experiment = ExperimentIdentity.create("run-1", "Pixel")
    path = experiment_directory(str(tmp_path), experiment)

    with pytest.raises(FileExistsError, match="resume=True"):
        experiment_directory(str(tmp_path), experiment)
    assert experiment_directory(str(tmp_path), experiment, resume=True) == path


def test_experiment_directory_rejects_unidentified_legacy_directory(tmp_path):
    experiment = ExperimentIdentity.create("run-1", "Pixel")
    legacy = tmp_path / "Pixel" / "run-1"
    legacy.mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot safely resume legacy"):
        experiment_directory(str(tmp_path), experiment, resume=True)
