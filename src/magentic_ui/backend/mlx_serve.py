"""Launch local MLX OpenAI-compatible servers for MagenticBrain and Fara.

Apple Silicon only. Requires the optional ``mlx`` extra:

    uv pip install "magentic_ui[mlx]"

This module must not import ``mlx_lm`` / ``mlx_vlm`` at import time so a
default install can still load the CLI.
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import typer

DEFAULT_BRAIN_MODEL = "mlx-community/MagenticBrain-8bit"
DEFAULT_FARA_MODEL = "mlx-community/Fara1.5-9B-8bit"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_BRAIN_PORT = 8100
DEFAULT_FARA_PORT = 8101


class Role(str, Enum):
    ALL = "all"
    BRAIN = "brain"
    FARA = "fara"


_MISSING_EXTRA = (
    "MLX extras are not installed. On Apple Silicon run:\n"
    '  uv pip install "magentic_ui[mlx]"'
)
_UNSUPPORTED_PLATFORM = (
    "magentic-ui mlx-serve requires macOS on Apple Silicon "
    f"(got {platform.system()} {platform.machine()})."
)


@dataclass(frozen=True)
class ServerSpec:
    """One mlx_lm / mlx_vlm HTTP server to spawn."""

    label: str
    binary: str
    argv: list[str]
    url: str
    model: str


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {
        "arm64",
        "aarch64",
    }


def resolve_binary(name: str) -> str | None:
    """Return an executable on PATH, or None if missing."""
    return shutil.which(name)


def build_specs(
    *,
    role: Role,
    host: str,
    brain_model: str,
    fara_model: str,
    brain_port: int,
    fara_port: int,
) -> list[ServerSpec]:
    """Build the subprocess specs for the requested role."""
    specs: list[ServerSpec] = []
    if role in (Role.ALL, Role.BRAIN):
        specs.append(
            ServerSpec(
                label="MagenticBrain",
                binary="mlx_lm.server",
                argv=[
                    "mlx_lm.server",
                    "--model",
                    brain_model,
                    "--host",
                    host,
                    "--port",
                    str(brain_port),
                ],
                url=f"http://{host}:{brain_port}/v1",
                model=brain_model,
            )
        )
    if role in (Role.ALL, Role.FARA):
        specs.append(
            ServerSpec(
                label="Fara",
                binary="mlx_vlm.server",
                argv=[
                    "mlx_vlm.server",
                    "--model",
                    fara_model,
                    "--host",
                    host,
                    "--port",
                    str(fara_port),
                ],
                url=f"http://{host}:{fara_port}/v1",
                model=fara_model,
            )
        )
    return specs


def _missing_extra_message(missing: Sequence[str]) -> str:
    names = ", ".join(missing)
    return f"Could not find {names} on PATH.\n{_MISSING_EXTRA}"


def mlx_serve(
    role: Role = typer.Option(
        Role.ALL,
        "--role",
        help="Which model server(s) to start: all, brain, or fara.",
    ),
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        help="Bind host for the MLX HTTP servers.",
    ),
    brain_model: str = typer.Option(
        DEFAULT_BRAIN_MODEL,
        "--brain-model",
        help="Hugging Face repo or local path for MagenticBrain.",
    ),
    fara_model: str = typer.Option(
        DEFAULT_FARA_MODEL,
        "--fara-model",
        help="Hugging Face repo or local path for Fara.",
    ),
    brain_port: int = typer.Option(
        DEFAULT_BRAIN_PORT,
        "--brain-port",
        help="Port for MagenticBrain (mlx_lm.server).",
    ),
    fara_port: int = typer.Option(
        DEFAULT_FARA_PORT,
        "--fara-port",
        help="Port for Fara (mlx_vlm.server).",
    ),
) -> None:
    """Start local MLX OpenAI-compatible servers for MagenticBrain and/or Fara.

    MagenticLite is not started. Point Settings → Models (or config.yaml)
    at the printed /v1 URLs. Requires the optional mlx extra on Apple Silicon.
    """
    if not is_apple_silicon():
        typer.echo(_UNSUPPORTED_PLATFORM, err=True)
        raise typer.Exit(code=1)

    specs = build_specs(
        role=role,
        host=host,
        brain_model=brain_model,
        fara_model=fara_model,
        brain_port=brain_port,
        fara_port=fara_port,
    )
    missing = [s.binary for s in specs if resolve_binary(s.binary) is None]
    if missing:
        typer.echo(_missing_extra_message(missing), err=True)
        raise typer.Exit(code=1)

    typer.echo("Starting MLX OpenAI-compatible servers (Ctrl-C to stop).")
    for spec in specs:
        typer.echo(f"  {spec.label}: {spec.url}  ({spec.model})")
    typer.echo("")
    typer.echo("Paste into MagenticLite onboarding / Settings → Models:")
    for spec in specs:
        role_name = "Orchestrator" if spec.label == "MagenticBrain" else "Browser use"
        typer.echo(f"  {role_name}")
        typer.echo(f"    Endpoint URL: {spec.url}")
        typer.echo(f"    Model Name:   {spec.model}")
        typer.echo("    API Key:      not-needed")

    procs: list[subprocess.Popen[bytes]] = []
    try:
        for spec in specs:
            procs.append(
                subprocess.Popen(
                    spec.argv,
                    start_new_session=sys.platform != "win32",
                )
            )
        _wait_for_interrupt(procs)
    finally:
        _stop_procs(procs)


def _wait_for_interrupt(procs: Sequence[subprocess.Popen[bytes]]) -> None:
    try:
        while True:
            for proc in procs:
                code = proc.poll()
                if code is not None:
                    typer.echo(
                        f"A server exited with code {code}; stopping the rest.",
                        err=True,
                    )
                    raise typer.Exit(code=code or 1)
            try:
                procs[0].wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        typer.echo("\nStopping MLX servers.")


def _stop_procs(procs: Sequence[subprocess.Popen[bytes]]) -> None:
    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            if sys.platform != "win32" and proc.pid:
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except (ProcessLookupError, PermissionError):
            continue
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def register(app: typer.Typer) -> None:
    """Attach ``mlx-serve`` to the MagenticLite Typer app."""
    app.command("mlx-serve")(mlx_serve)
