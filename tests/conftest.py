"""Common Test Fixtures."""
import shutil
import tempfile
from typing import Generator

import pytest


@pytest.fixture(scope="function")
def tmpdir() -> Generator[str, None, None]:
    """Create a temporary data root directory.

    Yields:
        Generator[str, None, None]: Directory path.
    """
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)
