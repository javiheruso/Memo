"""Run the dependency-free browser-JS unit tests and syntax checks.

The voice client is plain JavaScript served as a static asset. These tests use
Node's built-in test runner and parser, so they add no project dependency. They
are skipped only when the Node executable is unavailable.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    NODE is None, reason="node executable is required for browser-JS tests"
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [NODE, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_voice_client_scripts_parse() -> None:
    for path in (
        "app/web/static/voice-silence.js",
        "app/web/static/voice.js",
    ):
        result = _run("--check", path)
        assert result.returncode == 0, result.stderr


def test_voice_silence_detector_unit_suite() -> None:
    result = _run("--test", "tests/js/voice_silence.test.js")
    assert result.returncode == 0, result.stdout + result.stderr
