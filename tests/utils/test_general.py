# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Test General utils
"""
import os
import shutil
import stat
from unittest.mock import mock_open, patch

from _pytest.monkeypatch import MonkeyPatch

import utils.general as general_utils


class TestIsKaggle:
    """Test `utils.general.is_kaggle` function."""
    def test_with_valid_env_variables(self, monkeypatch: MonkeyPatch):
        """Test it returns True when valid environment variables are provided.

        Args:
            monkeypatch (MonkeyPatch): Fixture for performing monkeypatching.
        """
        monkeypatch.setenv("PWD", "/kaggle/working")
        monkeypatch.setenv("KAGGLE_URL_BASE", "https://www.kaggle.com")
        assert general_utils.is_kaggle() is True

    def test_without_valid_env_variables(self):
        """Test it returns False when required environment varaibles are not provided."""
        assert general_utils.is_kaggle() is False


class TestIsWriteable:
    """Test `utils.general.is_writeable` function."""
    def test_on_readonly_directory_with_test_flag_set(self, tmpdir: str):
        """Test the provided directory is not writeable when `test=True`.

        Args:
            tmpdir (str): Fixture that creates and provides the path to a temporary directory.
        """
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = OSError()
            writeable = general_utils.is_writeable(tmpdir, test=True)
            assert writeable is False

    def test_on_writeable_directory_with_test_flag_set(self, tmpdir: str):
        """Test the provided directory is writeable when `test=True`.

        Args:
            tmpdir (str): Fixture that creates and provides the path to a temporary directory.
        """
        writeable = general_utils.is_writeable(tmpdir, test=True)
        assert writeable is True

    def test_on_writeable_directory_without_test_flag(self, tmpdir: str):
        """Test the provided directory is writeable when `test=False`.

        Args:
            tmpdir (str): Fixture that creates and provides the path to a temporary directory.
        """
        writeable = general_utils.is_writeable(tmpdir, test=False)
        assert writeable is True
