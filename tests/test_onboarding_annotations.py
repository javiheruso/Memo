"""Regression guard for annotation evaluation in the onboarding service.

``OnboardingService`` exposes an async method named ``list``. On Python 3.12
annotations are evaluated eagerly while the class body runs, so any later
annotation such as ``list[str]`` resolves to that method and aborts the import.
Deferred annotations (PEP 563) keep the module importable; this test protects
that guarantee for the Python 3.12 CI target.
"""

import __future__

from pathlib import Path

from app.services.onboarding import OnboardingService

SOURCE_PATH = Path(__file__).parents[1] / "app" / "services" / "onboarding.py"


def test_onboarding_service_keeps_deferred_annotations() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module_code = compile(source, str(SOURCE_PATH), "exec")

    assert module_code.co_flags & __future__.annotations.compiler_flag
    assert "list" in vars(OnboardingService)
    assert (
        OnboardingService.validation_problems.__annotations__["return"]
        == "list[str]"
    )
