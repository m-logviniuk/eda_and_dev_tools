"""Tests for project path resolution."""

import os

from abalone import config


def test_project_root_uses_env_variable(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model.joblib").write_bytes(b"test")

    previous = os.environ.get("ABALONE_PROJECT_ROOT")
    os.environ["ABALONE_PROJECT_ROOT"] = str(tmp_path)
    try:
        root = config._resolve_project_root()
        assert root == tmp_path.resolve()
    finally:
        if previous is None:
            os.environ.pop("ABALONE_PROJECT_ROOT", None)
        else:
            os.environ["ABALONE_PROJECT_ROOT"] = previous


def test_project_root_falls_back_to_cwd_with_artifacts(tmp_path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.delenv("ABALONE_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    root = config._resolve_project_root()
    assert root == tmp_path.resolve()
