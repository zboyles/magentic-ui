"""Tests for magentic-ui mlx-serve (no live MLX processes)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from magentic_ui.backend.cli import app
from magentic_ui.backend.mlx_serve import (
    DEFAULT_BRAIN_MODEL,
    DEFAULT_FARA_MODEL,
    Role,
    build_specs,
    is_apple_silicon,
)


class TestBuildSpecs:
    def test_all_includes_brain_and_fara(self) -> None:
        specs = build_specs(
            role=Role.ALL,
            host="127.0.0.1",
            brain_model=DEFAULT_BRAIN_MODEL,
            fara_model=DEFAULT_FARA_MODEL,
            brain_port=8100,
            fara_port=8101,
        )
        labels = [s.label for s in specs]
        assert labels == ["MagenticBrain", "Fara"]
        assert specs[0].url == "http://127.0.0.1:8100/v1"
        assert specs[1].url == "http://127.0.0.1:8101/v1"
        assert specs[0].binary == "mlx_lm.server"
        assert specs[1].binary == "mlx_vlm.server"

    def test_brain_only(self) -> None:
        specs = build_specs(
            role=Role.BRAIN,
            host="127.0.0.1",
            brain_model="local/brain",
            fara_model=DEFAULT_FARA_MODEL,
            brain_port=9000,
            fara_port=8101,
        )
        assert len(specs) == 1
        assert specs[0].label == "MagenticBrain"
        assert specs[0].model == "local/brain"
        assert specs[0].url == "http://127.0.0.1:9000/v1"

    def test_fara_only(self) -> None:
        specs = build_specs(
            role=Role.FARA,
            host="0.0.0.0",
            brain_model=DEFAULT_BRAIN_MODEL,
            fara_model="local/fara",
            brain_port=8100,
            fara_port=8102,
        )
        assert len(specs) == 1
        assert specs[0].label == "Fara"
        assert "--host" in specs[0].argv
        assert "0.0.0.0" in specs[0].argv


class TestPlatform:
    def test_is_apple_silicon_false_on_linux(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "magentic_ui.backend.mlx_serve.platform.system", lambda: "Linux"
        )
        monkeypatch.setattr(
            "magentic_ui.backend.mlx_serve.platform.machine", lambda: "x86_64"
        )
        assert is_apple_silicon() is False

    def test_is_apple_silicon_true_on_darwin_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "magentic_ui.backend.mlx_serve.platform.system", lambda: "Darwin"
        )
        monkeypatch.setattr(
            "magentic_ui.backend.mlx_serve.platform.machine", lambda: "arm64"
        )
        assert is_apple_silicon() is True


class TestCli:
    def test_rejects_non_apple_silicon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "magentic_ui.backend.mlx_serve.is_apple_silicon", lambda: False
        )
        result = CliRunner().invoke(app, ["mlx-serve"])
        assert result.exit_code == 1
        assert "Apple Silicon" in result.output

    def test_rejects_missing_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "magentic_ui.backend.mlx_serve.is_apple_silicon", lambda: True
        )
        monkeypatch.setattr(
            "magentic_ui.backend.mlx_serve.resolve_binary", lambda _name: None
        )
        result = CliRunner().invoke(app, ["mlx-serve"])
        assert result.exit_code == 1
        assert "magentic_ui[mlx]" in result.output

    def test_help_lists_command(self) -> None:
        result = CliRunner().invoke(app, ["mlx-serve", "--help"])
        assert result.exit_code == 0
        assert "--brain-model" in result.output
        assert "--fara-model" in result.output
