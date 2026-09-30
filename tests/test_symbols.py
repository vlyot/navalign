from navalign.symbols import enclosing_chain, index_symbols, module_candidates

SOURCE = """\
import os


def outer():
    def inner():
        return 1
    return inner


class C:
    @staticmethod
    def m():
        pass

    async def am(self):
        pass
"""


def test_module_candidates():
    assert module_candidates("src/requests/utils.py") == ["src.requests.utils", "requests.utils", "utils"]
    assert module_candidates("rich/__init__.py") == ["rich"]


def test_graph_ids_drop_enclosing_functions_keep_classes():
    by_qual = {s.qualname: s for s in index_symbols(SOURCE, "pkg.mod")}
    assert by_qual["pkg.mod.outer.inner"].graph_id == "pkg.mod.inner"
    assert by_qual["pkg.mod.C.m"].graph_id == "pkg.mod.C.m"
    assert by_qual["pkg.mod.C.am"].is_async


def test_decorator_line_belongs_to_function():
    by_qual = {s.qualname: s for s in index_symbols(SOURCE, "pkg.mod")}
    assert by_qual["pkg.mod.C.m"].start == 11  # @staticmethod line


def test_enclosing_chain_innermost_first():
    syms = index_symbols(SOURCE, "pkg.mod")
    assert [s.qualname for s in enclosing_chain(syms, 6)] == ["pkg.mod.outer.inner", "pkg.mod.outer"]
    assert [s.qualname for s in enclosing_chain(syms, 16)] == ["pkg.mod.C.am", "pkg.mod.C"]
    assert enclosing_chain(syms, 1) == []
