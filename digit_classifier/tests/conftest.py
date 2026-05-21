"""
conftest.py — Shared pytest fixtures and markers for digit_classifier tests.

Adds the parent directory to sys.path so tests can import modules without
installing the package.

Provides the `skip_if_no_nest` fixture: any test that calls it is
automatically skipped when PyNN + NEST are not installed.
"""

import sys
from pathlib import Path
import pytest

# Allow `from digit_generator import ...` style imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "nest: tests requiring PyNN + NEST backend (skipped if not installed)",
    )


@pytest.fixture(scope="session")
def pynn_available() -> bool:
    try:
        import pyNN.nest  # noqa: F401
        return True
    except (ImportError, Exception):
        return False


@pytest.fixture
def skip_if_no_nest(pynn_available):
    """Skip the test if NEST is not installed."""
    if not pynn_available:
        pytest.skip("PyNN + NEST not installed — skipping simulation test")


# ── Shared data fixtures ──────────────────────────────────────────────────────

@pytest.fixture(params=range(10))
def each_digit(request):
    """Parametrised fixture: yields each digit 0–9 in turn."""
    return request.param


@pytest.fixture
def all_digit_images():
    from digit_generator import create_all_digit_images
    return create_all_digit_images()


@pytest.fixture
def all_template_indices():
    from digit_generator import get_all_template_indices
    return get_all_template_indices()


@pytest.fixture
def random_image():
    from digit_generator import create_random_image
    return create_random_image(seed=42, density=0.25)
