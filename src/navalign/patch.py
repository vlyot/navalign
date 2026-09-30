"""Parse unified-diff patches into per-file changed line numbers, and apply them to base text."""

from __future__ import annotations

from dataclasses import dataclass, field

from unidiff import PatchSet


class PatchApplyError(ValueError):
    """The patch's context or removed lines don't match the base file."""


@dataclass
class FileChange:
    path: str
    is_new: bool
    is_deleted: bool
    removed_lines: list[int] = field(default_factory=list)  # base-side line numbers, non-blank only
    added_lines: list[int] = field(default_factory=list)  # post-patch line numbers, non-blank only
    hunks: list = field(default_factory=list, repr=False)  # unidiff Hunk objects

    @property
    def is_python(self) -> bool:
        return self.path.endswith(".py")


def parse_patch(patch_text: str) -> list[FileChange]:
    """Split a patch into one FileChange per file, in patch order."""
    changes = []
    for pf in PatchSet(patch_text):
        hunks = list(pf)
        is_new = pf.is_added_file or (
            len(hunks) == 1 and hunks[0].source_start == 0 and hunks[0].source_length == 0
        )
        is_deleted = pf.is_removed_file or (
            len(hunks) == 1 and hunks[0].target_start == 0 and hunks[0].target_length == 0
        )
        change = FileChange(path=pf.path, is_new=is_new, is_deleted=is_deleted, hunks=hunks)
        for hunk in hunks:
            for line in hunk:
                if not line.value.strip():
                    continue  # blank lines carry no location signal
                if line.is_removed:
                    change.removed_lines.append(line.source_line_no)
                elif line.is_added:
                    change.added_lines.append(line.target_line_no)
        changes.append(change)
    return changes


def apply_file_change(base_text: str, change: FileChange) -> str:
    """Return the post-patch text of one file. Raises PatchApplyError if the hunks don't match."""
    base = [] if change.is_new else base_text.split("\n")
    out: list[str] = []
    pos = 0  # index into base of the next line not yet copied
    for hunk in change.hunks:
        start = hunk.source_start if hunk.source_length == 0 else hunk.source_start - 1
        if start < pos:
            raise PatchApplyError(f"{change.path}: overlapping hunks at line {hunk.source_start}")
        out.extend(base[pos:start])
        pos = start
        for line in hunk:
            text = line.value.rstrip("\r\n")
            if line.is_added:
                out.append(text)
                continue
            if pos >= len(base) or base[pos].rstrip("\r") != text:
                got = base[pos] if pos < len(base) else "<EOF>"
                raise PatchApplyError(f"{change.path}:{pos + 1}: expected {text!r}, found {got!r}")
            if line.is_context:
                out.append(base[pos])
            pos += 1
    out.extend(base[pos:])
    return "\n".join(out)
