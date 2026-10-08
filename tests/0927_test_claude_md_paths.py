"""Every CLAUDE.md folder guide carries a `Last verified:` date and names only paths that exist.

A stale guide is worse than none: a fresh agent trusts it instead of looking. The guides are
gitignored (local-only), so on a clone without them this collects nothing and passes.

A backticked token counts as a path when it contains `/` and only path characters (so commands,
code, `<placeholders>` and `*` globs are skipped). It resolves against the guide's own folder,
then the repo root.
"""

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PRUNE = {".git", ".venv", "wandb", "_decipher_models", "__pycache__", "trained", "figs", "node_modules"}
PATH_TOKEN = re.compile(r"^[\w .\-/]+$")
DATE_LINE = re.compile(r"^Last verified: \d{4}-\d{2}-\d{2}$", re.M)


def _guides():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in PRUNE]
        if "CLAUDE.md" in files:
            yield Path(root) / "CLAUDE.md"


GUIDES = sorted(_guides())


def _path_tokens(text):
    for tok in re.findall(r"`([^`\n]+)`", text):
        if "/" in tok and " -" not in tok and PATH_TOKEN.match(tok) and not tok.startswith("~"):
            yield tok


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: str(p.relative_to(REPO)))
def test_guide_has_last_verified_date(guide):
    head = "\n".join(guide.read_text().splitlines()[:5])
    assert DATE_LINE.search(head), f"{guide}: add 'Last verified: YYYY-MM-DD' near the top"


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: str(p.relative_to(REPO)))
def test_guide_paths_exist(guide):
    missing = [
        tok
        for tok in sorted(set(_path_tokens(guide.read_text())))
        if not (guide.parent / tok).exists() and not (REPO / tok).exists()
    ]
    assert not missing, f"{guide.relative_to(REPO)} names paths that no longer exist: {missing}"
