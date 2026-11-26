"""Fakes for functions of other modules that these tests reach through their documented seams.

The consumers import the owning module and look the function up at call time, so a fake set on
that module wins for the length of the test. `install` refuses an attribute the module does not
have: a renamed or removed seam fails here instead of leaving a fake nobody calls.
`tests/api/test_agentar_flow.py` runs the real functions end to end.
"""
import importlib


def install(monkeypatch, module_name: str, **attributes) -> None:
    module = importlib.import_module(module_name)
    for key, value in attributes.items():
        monkeypatch.setattr(module, key, value, raising=True)
