"""Helpers for writing LaTeX tables and snippets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .io_utils import ensure_dir


def escape_latex(text: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    result = str(text)
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def write_simple_table(
    path: str | Path,
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    caption: str,
    label: str,
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    cols = "l" * len(headers)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join(escape_latex(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(escape_latex(v) for v in row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{escape_latex(caption)}}}",
            rf"\label{{{escape_latex(label)}}}",
            r"\end{table}",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")

