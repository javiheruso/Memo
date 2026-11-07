"""The container entrypoint migrates once, then hands off to Uvicorn."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "docker" / "entrypoint.sh"
DOCKERFILE = ROOT / "Dockerfile"


@dataclass
class Harness:
    bin_dir: Path
    capture_dir: Path

    def run(self, *args: str, env: dict[str, str] | None = None):
        base_env = {
            "PATH": f"{self.bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE_DIR": str(self.capture_dir),
        }
        base_env.update(env or {})
        return subprocess.run(
            [str(ENTRYPOINT), *args],
            capture_output=True,
            text=True,
            env=base_env,
            check=False,
        )

    def alembic_calls(self) -> list[str]:
        path = self.capture_dir / "alembic_calls"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def uvicorn_args(self) -> list[str]:
        path = self.capture_dir / "uvicorn_args"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def read(self, name: str) -> str:
        return (self.capture_dir / name).read_text(encoding="utf-8")


def _write_stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    bin_dir = tmp_path / "bin"
    capture_dir = tmp_path / "capture"
    bin_dir.mkdir()
    capture_dir.mkdir()

    _write_stub(
        bin_dir / "alembic",
        'printf \'%s\\n\' "$*" >> "$CAPTURE_DIR/alembic_calls"\n'
        'exit "${ALEMBIC_EXIT:-0}"\n',
    )
    _write_stub(
        bin_dir / "uvicorn",
        'printf \'%s\\n\' "$@" > "$CAPTURE_DIR/uvicorn_args"\n'
        'printf \'%s\' "${MEMO_AUTO_MIGRATE-unset}" > "$CAPTURE_DIR/auto_migrate"\n'
        'printf \'%s\' "${PORT-unset}" > "$CAPTURE_DIR/port"\n'
        'printf \'%s\' "$$" > "$CAPTURE_DIR/pid"\n'
        'exit "${UVICORN_EXIT:-0}"\n',
    )
    return Harness(bin_dir=bin_dir, capture_dir=capture_dir)


def test_default_command_migrates_once_then_disables_lifespan_migration(
    harness: Harness,
) -> None:
    result = harness.run("serve", env={"MEMO_AUTO_MIGRATE": "true", "PORT": "9123"})

    assert result.returncode == 0
    assert harness.alembic_calls() == ["upgrade head"]
    assert harness.uvicorn_args() == [
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "9123",
    ]
    # The app process must not run the same migrations again in its lifespan.
    assert harness.read("auto_migrate") == "false"


def test_no_arguments_also_uses_the_default_command(harness: Harness) -> None:
    result = harness.run(env={"MEMO_AUTO_MIGRATE": "true"})

    assert result.returncode == 0
    assert harness.alembic_calls() == ["upgrade head"]
    assert harness.uvicorn_args()[-1] == "8000"


def test_truthy_auto_migrate_variants_are_accepted(harness: Harness) -> None:
    result = harness.run("serve", env={"MEMO_AUTO_MIGRATE": "TRUE"})

    assert result.returncode == 0
    assert harness.alembic_calls() == ["upgrade head"]
    assert harness.read("auto_migrate") == "false"


def test_migrations_are_skipped_when_auto_migrate_is_disabled(
    harness: Harness,
) -> None:
    result = harness.run("serve", env={"MEMO_AUTO_MIGRATE": "false"})

    assert result.returncode == 0
    assert harness.alembic_calls() == []
    assert harness.read("auto_migrate") == "false"
    assert harness.uvicorn_args()[-1] == "8000"


def test_migration_failure_fails_fast_without_starting_uvicorn(
    harness: Harness,
) -> None:
    result = harness.run(
        "serve", env={"MEMO_AUTO_MIGRATE": "true", "ALEMBIC_EXIT": "1"}
    )

    assert result.returncode == 1
    assert harness.alembic_calls() == ["upgrade head"]
    assert harness.uvicorn_args() == []
    assert "database migrations failed" in result.stderr


def test_explicit_command_bypasses_the_default_migration_step(
    harness: Harness,
) -> None:
    result = harness.run(
        "alembic", "upgrade", "head", env={"MEMO_AUTO_MIGRATE": "true"}
    )

    assert result.returncode == 0
    # Exactly one invocation: the explicit command, not a wrapper pre-run.
    assert harness.alembic_calls() == ["upgrade head"]
    assert harness.uvicorn_args() == []


def test_entrypoint_replaces_itself_with_uvicorn(tmp_path: Path) -> None:
    harness = Harness(bin_dir=tmp_path / "bin", capture_dir=tmp_path / "capture")
    harness.bin_dir.mkdir()
    harness.capture_dir.mkdir()
    _write_stub(
        harness.bin_dir / "uvicorn",
        'printf \'%s\' "$$" > "$CAPTURE_DIR/pid"\n',
    )

    process = subprocess.Popen(
        [str(ENTRYPOINT), "serve"],
        env={
            "PATH": f"{harness.bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE_DIR": str(harness.capture_dir),
            "MEMO_AUTO_MIGRATE": "false",
        },
    )
    process.wait(timeout=10)

    assert process.returncode == 0
    # `exec` must make Uvicorn the same process, so it receives container signals.
    assert harness.read("pid") == str(process.pid)


def test_dockerfile_installs_the_entrypoint_and_default_command() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY docker/entrypoint.sh /usr/local/bin/memo-entrypoint" in text
    assert 'ENTRYPOINT ["/usr/local/bin/memo-entrypoint"]' in text
    assert 'CMD ["serve"]' in text


def test_entrypoint_is_posix_and_uses_exec() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")

    assert text.startswith("#!/bin/sh\n")
    assert "set -eu" in text
    assert 'exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"' in text
    assert '"$@"' in text
