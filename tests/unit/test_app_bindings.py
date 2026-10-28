"""Every studio.<module>.<name> the GUI references must exist.

The Analyze button once called roi_utils.flatten_for_analysis and
analyzer.analyze, neither of which was ever defined, and nothing caught it.
This scans app.py statically so the check needs no display and no PyTorch.
"""
import ast
import importlib
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app.py"
STUDIO_MODULES = ("sampler", "roi_utils", "autolabel", "trainer", "analyzer", "qaqc")


def _referenced_attrs():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    refs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in STUDIO_MODULES:
            refs.add((node.value.id, node.attr))
    return refs


def test_gui_references_resolve():
    refs = _referenced_attrs()
    assert refs, "no studio.* references found; scan is broken"
    missing = []
    for mod_name, attr in sorted(refs):
        mod = importlib.import_module(f"studio.{mod_name}")
        if not hasattr(mod, attr):
            missing.append(f"studio.{mod_name}.{attr}")
    assert not missing, f"app.py references undefined names: {missing}"


def test_analyze_handler_uses_real_entry_points():
    refs = _referenced_attrs()
    assert ("analyzer", "analyze_video") in refs
    assert ("roi_utils", "flatten_for_analysis") not in refs
    assert ("analyzer", "analyze") not in refs
