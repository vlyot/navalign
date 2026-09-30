import pytest

from navalign.patch import PatchApplyError, apply_file_change, parse_patch

BASE = "a = 1\n\n\ndef f():\n    return 1\n"

MODIFY = """\
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -4,2 +4,4 @@
 def f():
-    return 1
+    x = 2
+
+    return x
"""

NEW_FILE = """\
--- a/pkg/new.py
+++ b/pkg/new.py
@@ -0,0 +1,2 @@
+def g():
+    pass
"""


def test_parse_modify_lines_skip_blank():
    (change,) = parse_patch(MODIFY)
    assert change.path == "pkg/mod.py"
    assert not change.is_new and not change.is_deleted and change.is_python
    assert change.removed_lines == [5]
    assert change.added_lines == [5, 7]  # blank +6 is skipped


def test_parse_new_file():
    (change,) = parse_patch(NEW_FILE)
    assert change.is_new
    assert change.added_lines == [1, 2]


def test_apply_modify():
    (change,) = parse_patch(MODIFY)
    assert apply_file_change(BASE, change) == "a = 1\n\n\ndef f():\n    x = 2\n\n    return x\n"


def test_apply_new_file():
    (change,) = parse_patch(NEW_FILE)
    assert apply_file_change("", change) == "def g():\n    pass"


def test_apply_mismatch_raises():
    (change,) = parse_patch(MODIFY)
    with pytest.raises(PatchApplyError):
        apply_file_change(BASE.replace("return 1", "return 9"), change)


def test_multiple_files():
    changes = parse_patch(MODIFY + NEW_FILE)
    assert [c.path for c in changes] == ["pkg/mod.py", "pkg/new.py"]
