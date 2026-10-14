# Make the repository importable as `railtwin_common` for the tests, whatever the folder is called
# (e.g. a clone named RailTwin-Common). Not needed after `pip install -e .`.
import importlib.util
import sys
from pathlib import Path

if "railtwin_common" not in sys.modules:
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("railtwin_common", root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["railtwin_common"] = module
    spec.loader.exec_module(module)
