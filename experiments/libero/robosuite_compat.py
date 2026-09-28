"""Compatibility helpers for importing robosuite in shared environments."""

import importlib.util
import sys
from pathlib import Path


def disable_robosuite_file_logging() -> None:
    """Disable robosuite's hard-coded shared ``/tmp/robosuite.log`` handler.

    Robosuite 1.4 imports ``robosuite.macros_private`` before constructing its
    default logger. When that optional module is absent, the packaged defaults
    enable file logging to a single global path. On a multi-user cluster that
    file can be owned by another user, making even ``import robosuite`` fail.

    Load the installed macro file without importing the robosuite package,
    override only ``FILE_LOGGING_LEVEL``, and expose it under the private-module
    name expected by robosuite. All other installed/custom macro values remain
    unchanged.
    """

    private_module_name = "robosuite.macros_private"
    if private_module_name in sys.modules:
        sys.modules[private_module_name].FILE_LOGGING_LEVEL = None
        if "robosuite.macros" in sys.modules:
            sys.modules["robosuite.macros"].FILE_LOGGING_LEVEL = None
        return
    if "robosuite" in sys.modules:
        raise RuntimeError(
            "disable_robosuite_file_logging() must run before importing robosuite."
        )

    package_spec = importlib.util.find_spec("robosuite")
    if package_spec is None or package_spec.submodule_search_locations is None:
        return

    package_dir = Path(next(iter(package_spec.submodule_search_locations)))
    private_macros_path = package_dir / "macros_private.py"
    macros_path = private_macros_path if private_macros_path.is_file() else package_dir / "macros.py"
    if not macros_path.is_file():
        return

    macros_spec = importlib.util.spec_from_file_location(private_module_name, macros_path)
    if macros_spec is None or macros_spec.loader is None:
        return
    macros_module = importlib.util.module_from_spec(macros_spec)
    macros_spec.loader.exec_module(macros_module)
    macros_module.FILE_LOGGING_LEVEL = None
    sys.modules[private_module_name] = macros_module
    # Internal robosuite modules import ``robosuite.macros`` directly. Reuse
    # the same patched module so a later log_utils import cannot re-enable the
    # hard-coded file handler.
    sys.modules["robosuite.macros"] = macros_module
