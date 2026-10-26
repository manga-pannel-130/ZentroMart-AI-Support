"""Resolve the supplied course dataset independently of the code checkout."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_FILES = (
    "data/products.csv",
    "data/orders.csv",
    "data/order_items.csv",
    "data/customers.csv",
    "data/holidays.csv",
    "practice/questions.csv",
    "practice/conversations.csv",
)
_REQUIRED_DIRS = ("data", "practice", "tools")


def _missing_parts(root: Path) -> list[str]:
    missing = [name + "/" for name in _REQUIRED_DIRS if not (root / name).is_dir()]
    kb = root / "kb"
    markdown = sorted(kb.glob("*.md")) if kb.is_dir() else []
    if not markdown:
        missing.append("kb/*.md (knowledge-base documents)")
    elif not any("## [" in path.read_text(encoding="utf-8", errors="replace") for path in markdown):
        missing.append("sectioned Markdown headings in kb/*.md")
    missing.extend(name for name in _REQUIRED_FILES if not (root / name).is_file())
    return missing


def _validated(root: Path) -> Path | None:
    root = root.expanduser().resolve()
    return root if not _missing_parts(root) else None


def resolve_data_root(data_dir: str | Path | None = None) -> Path:
    """Return a complete dataset root or raise an actionable path error.

    Explicit arguments and ZENTROMART_DATA_DIR take precedence. Without either,
    search beside the extracted project and the common Desktop/Projects location.
    """
    requested = data_dir if data_dir is not None else os.getenv("ZENTROMART_DATA_DIR")
    if requested is not None and str(requested).strip():
        root = Path(requested).expanduser().resolve()
        missing = _missing_parts(root)
        if not missing:
            return root
        raise FileNotFoundError(
            f"ZentroMart dataset at {root} is incomplete. Missing: {', '.join(missing)}. "
            "Set ZENTROMART_DATA_DIR to the extracted dataset folder containing kb/, data/, practice/, and tools/."
        )

    candidates = (
        ROOT,
        ROOT.parent / "Projects" / "zentromart-capstone-submission",
        ROOT.parent / "Capstone_Project_Dataset_FTTT_2026",
        ROOT.parent.parent / "Projects" / "zentromart-capstone-submission",
    )
    seen: set[Path] = set()
    problems: list[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        found = _validated(candidate)
        if found is not None:
            return found
        missing = _missing_parts(candidate)
        problems.append(f"{candidate} (missing {', '.join(missing)})")

    tried = "\n - ".join(problems)
    raise FileNotFoundError(
        "Could not locate a complete ZentroMart capstone dataset. Checked:\n - " + tried +
        "\nSet ZENTROMART_DATA_DIR to the extracted dataset folder containing kb/, data/, practice/, and tools/."
    )