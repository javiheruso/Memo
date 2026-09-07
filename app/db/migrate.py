"""Run Alembic migrations programmatically.

Used by the optional auto-migrate-on-startup path so a deployment can apply
migrations without invoking the ``alembic`` CLI. The repository layout keeps
``alembic.ini`` and ``migrations/`` next to the ``app`` package in source, but a
Docker build copies them to the image working directory (``/app``) while the
``app`` package is installed into site-packages; ``_project_root`` handles both.
"""

import os
from pathlib import Path

from alembic import command
from alembic.config import Config


def project_root() -> str:
    """Return the directory that holds ``alembic.ini`` and ``migrations/``."""
    here = Path(__file__).resolve().parent  # app/db
    root = here.parent.parent  # project root in a source checkout
    if (root / "alembic.ini").exists() and (root / "migrations").exists():
        return str(root)
    return os.getcwd()


def run_migrations() -> None:
    """Apply all pending migrations to the configured database."""
    root = project_root()
    cfg = Config(os.path.join(root, "alembic.ini"))
    # Make script_location absolute so migrations resolve regardless of the
    # process working directory. ``migrations/env.py`` reads the database URL
    # from the application settings (``MEMO_DATABASE_URL``).
    cfg.set_main_option("script_location", os.path.join(root, "migrations"))
    cfg.set_main_option("prepend_sys_path", root)
    command.upgrade(cfg, "head")
