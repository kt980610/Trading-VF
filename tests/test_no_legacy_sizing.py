"""Audit: the legacy TotalNetPnL / fixed-%-balance sizing must stay removed.

Position sizing is now driven solely by the MVO portfolio-weight artifact
(`total_equity * weight`). This test fails if any legacy sizing token reappears
in the Python or Rust source / config, guarding against regressions.
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories that hold real source / config we control.
SCAN_DIRS = [
    os.path.join("src"),
    os.path.join("rust_live", "src"),
    os.path.join("rust_live", "config"),
    os.path.join("config"),
]

SCAN_SUFFIXES = (".py", ".rs", ".yaml", ".yml")

# Exact legacy sizing tokens / formula fragments that must never reappear.
FORBIDDEN = [
    "TotalNetPnL",
    "currentBalance",
    "0.1 * currentBalance",
    "total_deploy_capital",
    "use_account_balance",
    "deploy_capital",
]

# This audit file itself necessarily mentions the tokens.
SELF = os.path.basename(__file__)


def _iter_source_files():
    for rel in SCAN_DIRS:
        base = os.path.join(REPO_ROOT, rel)
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for name in files:
                if name == SELF:
                    continue
                if name.endswith(SCAN_SUFFIXES):
                    yield os.path.join(root, name)


def test_no_legacy_sizing_tokens():
    offenders = []
    for path in _iter_source_files():
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{os.path.relpath(path, REPO_ROOT)}: {token}")
    assert not offenders, "legacy sizing references found:\n" + "\n".join(offenders)
