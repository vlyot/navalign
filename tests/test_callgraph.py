import json
import textwrap

import pytest

from navalign.callgraph import build_callgraph, main, write_graph
from navalign.data import example_dir, load_graph
from navalign.symbols import index_symbols


def repo(tmp_path, files: dict[str, str]):
    for path, text in files.items():
        f = tmp_path / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(textwrap.dedent(text), encoding="utf-8")
    return tmp_path


def edges(tmp_path, files):
    return set(build_callgraph(repo(tmp_path, files)).edges)


def test_async_functions_are_nodes(tmp_path):
    g = build_callgraph(repo(tmp_path, {"pkg/a.py": """
        async def fetch():
            return parse()
        def parse():
            pass
    """}))
    assert {"pkg.a.fetch", "pkg.a.parse"} <= set(g.nodes)
    assert ("pkg.a.fetch", "pkg.a.parse") in g.edges


def test_self_call_resolves_to_the_method_including_an_inherited_one(tmp_path):
    e = edges(tmp_path, {"pkg/a.py": """
        class Base:
            def send(self):
                pass
        class Session(Base):
            def prepare(self):
                pass
            def request(self):
                self.prepare()
                return self.send()
    """})
    assert {("pkg.a.Session.request", "pkg.a.Session.prepare"), ("pkg.a.Session.request", "pkg.a.Base.send")} <= e


def test_super_call_resolves_to_the_base_method(tmp_path):
    e = edges(tmp_path, {"pkg/a.py": """
        class Base:
            def run(self):
                pass
        class Child(Base):
            def run(self):
                super().run()
    """})
    assert ("pkg.a.Child.run", "pkg.a.Base.run") in e


def test_imported_alias_and_relative_reexport_resolve(tmp_path):
    e = edges(tmp_path, {
        "pkg/__init__.py": "from .models import Request\n",
        "pkg/models.py": """
            class Request:
                def __init__(self):
                    pass
        """,
        "pkg/utils.py": "def helper():\n    pass\n",
        "pkg/api.py": """
            import pkg.utils as u
            from . import Request
            def get():
                u.helper()
                return Request()
        """,
    })
    assert {("pkg.api.get", "pkg.utils.helper"), ("pkg.api.get", "pkg.models.Request"),
            ("pkg.api.get", "pkg.models.Request.__init__")} <= e


def test_classes_contain_their_members_and_inherit_their_bases(tmp_path):
    e = build_callgraph(repo(tmp_path, {"pkg/a.py": """
        class Base:
            pass
        class Child(Base):
            def run(self):
                pass
            class Meta:
                pass
    """})).edges
    assert e[("pkg.a.Child", "pkg.a.Child.run")] == e[("pkg.a.Child", "pkg.a.Child.Meta")] == "contains"
    assert e[("pkg.a.Child", "pkg.a.Base")] == "inherits"


def test_ambiguous_and_builtin_names_make_no_edge(tmp_path):
    e = edges(tmp_path, {"pkg/a.py": """
        class A:
            def close(self):
                pass
            def open(self):
                pass
        class B:
            def close(self):
                pass
        def run(f):
            f.close()
            open("x")
    """})
    assert not [edge for edge in e if edge[0] == "pkg.a.run"]


def test_unique_method_name_resolves_on_an_unknown_object(tmp_path):
    e = edges(tmp_path, {"pkg/a.py": """
        class Console:
            def render_lines(self):
                pass
        def show(c):
            c.render_lines()
    """})
    assert ("pkg.a.show", "pkg.a.Console.render_lines") in e


def test_node_ids_match_the_symbol_index(tmp_path):
    source = """
        import functools
        class Outer:
            class Inner:
                def m(self):
                    def helper():
                        pass
                    return helper()
        @functools.cache
        def top():
            def nested():
                pass
            return nested()
    """
    g = build_callgraph(repo(tmp_path, {"src/pkg/mod.py": source}))
    assert set(g.nodes) == {s.graph_id for s in index_symbols(textwrap.dedent(source), "pkg.mod")}
    assert ("pkg.mod.top", "pkg.mod.nested") in g.edges
    assert ("pkg.mod.Outer.Inner.m", "pkg.mod.Outer.Inner.helper") in g.edges
    assert g.nodes["pkg.mod.top"].startswith("@functools.cache")


def test_unparseable_files_are_skipped(tmp_path):
    g = build_callgraph(repo(tmp_path, {"pkg/ok.py": "def f():\n    pass\n", "pkg/py2.py": "print 'x'\n"}))
    assert g.skipped_files == ["pkg/py2.py"] and set(g.nodes) == {"pkg.ok.f"}


def test_written_graph_round_trips_through_load_graph(tmp_path):
    g = build_callgraph(example_dir() / "snapshots" / "toyshop_1", "toy")
    write_graph(g, tmp_path / "g.json")
    loaded = load_graph(tmp_path / "g.json")
    assert set(loaded.nodes) == set(g.nodes)
    assert {(a, b) for a, b in loaded.edges()} == set(g.edges)
    assert {d["type"] for *_, d in loaded.edges(data=True)} == {"calls", "contains"}


def test_toy_example_has_its_known_edges():
    g = build_callgraph(example_dir() / "snapshots" / "toyshop_1")
    shipped = load_graph(next((example_dir() / "graphs").glob("*.json")))
    assert set(g.nodes) == set(shipped.nodes)
    assert {("toyshop.cart.Cart.total", "toyshop.pricing.apply_discount"),
            ("toyshop.receipt.render", "toyshop.cart.Cart.total")} <= set(g.edges)


@pytest.mark.parametrize("twice", [False, True])
def test_main_writes_one_graph_per_commit(tmp_path, twice):
    snapshots = tmp_path / "snapshots"
    repo(snapshots / "toy_1", {"toy/a.py": "def f():\n    pass\n"})
    rows = [{"instance_id": "toy_1", "repo": "me/toy", "base_commit": "abc"}]
    if twice:
        repo(snapshots / "toy_2", {"toy/a.py": "def f():\n    pass\n"})
        rows.append({"instance_id": "toy_2", "repo": "me/toy", "base_commit": "abc"})
    (tmp_path / "tasks.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert main(["--tasks", str(tmp_path / "tasks.jsonl"), "--snapshots", str(snapshots), "--out", str(tmp_path / "g")]) == 0
    assert [p.name for p in (tmp_path / "g").iterdir()] == ["toy_abc.json"]
