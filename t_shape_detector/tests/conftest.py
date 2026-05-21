"""
conftest.py — Shared pytest configuration and fixtures.

Makes the parent directory (t_shape_detector/) importable from tests so we
can do `from image_generator import ...` without installing the package.

Also provides the `nest_sim` fixture that skips tests requiring PyNN/NEST
when those packages are not installed.
"""

import sys
from pathlib import Path
import pytest

# Allow `from image_generator import ...` style imports inside tests
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Markers ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "nest: marks tests that require the NEST simulator backend (skipped if not installed)",
    )


# ── PyNN/NEST availability fixture ────────────────────────────────────────────

@pytest.fixture(scope="session")
def pynn_available() -> bool:
    """Session-scoped check: is PyNN with NEST backend importable?"""
    try:
        import pyNN.nest  # noqa: F401
        return True
    except (ImportError, Exception):
        return False


@pytest.fixture
def skip_if_no_nest(pynn_available):
    """
    Request this fixture in any test that needs a live NEST simulation.
    The test is automatically skipped if NEST is not installed.

    Example
    -------
    def test_something(skip_if_no_nest):
        ...  # only runs if NEST is available
    """
    if not pynn_available:
        pytest.skip("PyNN + NEST backend not installed — skipping simulation test")


# ── Shared image fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def t_image():
    """A canonical 20×20 T-shape binary image."""
    from image_generator import create_t_shape_image
    return create_t_shape_image()


@pytest.fixture
def random_image():
    """A reproducible 20×20 random binary image (seed=42, density=30%)."""
    from image_generator import create_random_image
    return create_random_image(seed=42, density=0.30)
