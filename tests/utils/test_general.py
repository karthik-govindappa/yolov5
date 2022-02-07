# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Test General utils
"""
import logging
import os
import platform
import time
from unittest.mock import mock_open, patch

import pytest
from _pytest.capture import CaptureFixture
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


# TODO: pytest's logging is interfering with yolov5's logging. Find a better way of testing logging parts.
# class TestSetLogging:
#     """Test `utils.general.set_logging` function."""
#     def test_default_logger(self):
#         logger = general_utils.set_logging("test")
#         log_level_number = logger.level
#         log_level = logging.getLevelName(log_level_number)
#         assert log_level == "INFO"
#         assert len(logger.handlers) == 1

#     def test_logger_for_kaggle(self, monkeypatch: MonkeyPatch):
#         monkeypatch.setenv("PWD", "/kaggle/working")
#         monkeypatch.setenv("KAGGLE_URL_BASE", "https://www.kaggle.com")
#         logger = general_utils.set_logging("test")
#         log_level_number = logger.level
#         log_level = logging.getLevelName(log_level_number)
#         assert log_level == "INFO"
#         assert len(logger.handlers) == 1

class TestUserConfigDir:
    """Test `utils.general.user_config_dir` function."""
    def test_default_config_dir(self):
        """Test default user_config_dir."""
        path = general_utils.user_config_dir()
        if platform.system() == "Linux":
            assert str(path) == "/root/.config/Ultralytics"
        assert os.path.isdir(path)

    def test_custom_config_dir(self, monkeypatch: MonkeyPatch, tmpdir: str):
        """Test custom user_config_dir.

        Args:
            monkeypatch (MonkeyPatch): Fixture for performing monkeypatching.
            tmpdir (str): Fixture that creates and provides the path to a temporary directory.
        """
        monkeypatch.setenv("YOLOV5_CONFIG_DIR", tmpdir)
        path = general_utils.user_config_dir()
        assert str(path) == tmpdir
        assert os.path.isdir(path)


class TestProfile:
    """Test `utils.general.Profile` class."""
    def test_output(self, capsys: CaptureFixture):
        """Test post profiling output.

        Args:
            capsys (CaptureFixture): Fixture for capturing stdout/stderr output.
        """
        with general_utils.Profile():
            pass
        out, _ = capsys.readouterr()
        assert "Profile results:" in out


class TestTimeout:
    """Test `utils.general.Timeout` class."""
    def test_timeout_without_exception(self):
        """Test timeout without raising error."""
        with general_utils.Timeout(seconds=1, suppress_timeout_errors=True):
            time.sleep(1.1)

    def test_timeout_with_exception(self):
        """Test timeout which raises an exception."""
        with pytest.raises(TimeoutError) as error:
            with general_utils.Timeout(seconds=1, timeout_msg="time's up", suppress_timeout_errors=False):
                time.sleep(1.1)
        assert str(error.value) == "time's up"


class TestWorkingDirectory:
    """Test `utils.general.WorkingDirectory` class."""
    def test_directory_change(self, tmpdir: str):
        """Test switching of directories."""
        pre_cwd = os.getcwd()
        with general_utils.WorkingDirectory(new_dir=tmpdir):
            new_cwd = os.getcwd()
        post_cwd = os.getcwd()
        assert new_cwd == tmpdir
        assert pre_cwd == post_cwd


class TestTryExcept:
    """Test `utils.general.try_except` function."""
    def test_exception_is_captured(self, capsys: CaptureFixture):
        """Test given exception is captured and conveyed."""
        @general_utils.try_except
        def func():
            raise Exception("test exception")
        func()
        out, _ = capsys.readouterr()
        assert "test exception" in out


class TestMethods:
    """Test `utils.general.methods` function."""

    def test_listing_methods(self):
        """Test getting a list of methods of an instance."""
        class Dummy:
            def __init__(self) -> None:
                pass

            def method1(self):
                pass

            def method2(self):
                pass

            def _method3(self):
                pass

        dummy = Dummy()
        methods = general_utils.methods(dummy)
        assert len(methods) == 3
        assert "method1" in methods
        assert "method2" in methods
        assert "_method3" in methods
