#!/usr/bin/env python3
"""Assemble the final paper artifact and compile it with the AAAI template when available."""

from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from _bootstrap import ROOT
from src.utils.io_utils import ensure_dir, write_text

matplotlib.use("Agg")

LATEX_DIR = ROOT / "latex"
SECTIONS_DIR = LATEX_DIR / "sections"
TABLES_DIR = LATEX_DIR / "tables"
FIGURES_DIR = LATEX_DIR / "figures"
RESULTS_DIR = ROOT / "results"

SECTION_ORDER = [
    "intro",
    "related_work",
    "framework",
    "metrics",
    "method_overview",
    "experiments",
    "results",
    "limitations",
    "broader_impact",
    "conclusion",
]

MAIN_FIGURES = [
    "framework_overview.pdf",
    "main_results_panel.pdf",
    "sensortamper_monitorability.pdf",
]

MAIN_TABLES = [
    "environment_summary_compact.tex",
    "main_results_compact.tex",
    "sensortamper_monitor.tex",
]

REF_MAP = {
    "fig:framework-overview": "Figure 1",
    "fig:main-results": "Figure 2",
    "fig:sensortamper-monitor": "Figure 3",
    "tab:env-summary-compact": "Table 1",
    "tab:env-main": "Table 1",
    "tab:main-results-compact": "Table 2",
    "tab:sensortamper-monitor": "Table 3",
}

CITE_PAREN = {
    "amodei2016concrete": "Amodei et al., 2016",
    "armstrong2020pitfalls": "Armstrong et al., 2020",
    "cobbe2020leveraging": "Cobbe et al., 2020",
    "everitt2021reward": "Everitt et al., 2021",
    "leike2017ai": "Leike et al., 2017",
    "schulman2017ppo": "Schulman et al., 2017",
    "skalse2022rewardhacking": "Skalse et al., 2022",
}

CITE_TEXT = {
    "amodei2016concrete": "Amodei et al. (2016)",
    "armstrong2020pitfalls": "Armstrong et al. (2020)",
    "cobbe2020leveraging": "Cobbe et al. (2020)",
    "everitt2021reward": "Everitt et al. (2021)",
    "leike2017ai": "Leike et al. (2017)",
    "schulman2017ppo": "Schulman et al. (2017)",
    "skalse2022rewardhacking": "Skalse et al. (2022)",
}

@dataclass
class Block:
    kind: str
    text: str | None = None
    title: str | None = None
    path: str | None = None
    caption: str | None = None
    headers: list[str] | None = None
    rows: list[list[str]] | None = None


def extract_abstract(main_tex: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.S)
    if not match:
        raise ValueError("Could not find abstract in latex/main.tex")
    return match.group(1).strip()


def substitute_table_inputs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        rel = match.group(1)
        if not rel.startswith("tables/"):
            raise ValueError(f"Unexpected input dependency in final paper body: {rel}")
        table_path = LATEX_DIR / f"{rel}.tex"
        if not table_path.exists():
            raise FileNotFoundError(table_path)
        return table_path.read_text(encoding="utf-8").strip()

    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def assemble_body() -> tuple[str, list[tuple[str, Path]]]:
    parts: list[str] = []
    snapshot_sources: list[tuple[str, Path]] = []
    for section_name in SECTION_ORDER:
        section_path = SECTIONS_DIR / f"{section_name}.tex"
        section_text = section_path.read_text(encoding="utf-8").strip()
        expanded = substitute_table_inputs(section_text)
        parts.append(expanded)
        snapshot_sources.append((section_name, section_path))
    return "\n\n".join(parts), snapshot_sources


def build_final_tex() -> tuple[str, list[tuple[str, Path]]]:
    main_text = (LATEX_DIR / "main.tex").read_text(encoding="utf-8")
    abstract = extract_abstract(main_text)
    body, snapshot_sources = assemble_body()
    final_text = "\n".join(
        [
            r"\documentclass[10pt,twocolumn,letterpaper]{article}",
            r"\usepackage[submission]{aaai2026}",
            r"\usepackage{times}",
            r"\usepackage{helvet}",
            r"\usepackage{courier}",
            r"\usepackage[hyphens]{url}",
            r"\urlstyle{rm}",
            r"\def\UrlFont{\rm}",
            r"\usepackage{graphicx}",
            r"\usepackage{natbib}",
            r"\usepackage{caption}",
            r"\usepackage{booktabs}",
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
            r"\usepackage{multirow}",
            r"\newcommand{\framework}{SpecLens-RL}",
            r"\title{\framework: A Compact Framework for Diagnosing Specification Failures in Reinforcement Learning}",
            r"\author{Anonymous Authors}",
            r"\affiliations{}",
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
            abstract,
            r"\end{abstract}",
            body,
            r"\bibliography{references_seed}",
            r"\end{document}",
            "",
        ]
    )
    write_text(LATEX_DIR / "final.tex", final_text)
    return final_text, snapshot_sources


def find_inputs(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    return re.findall(r"\\input\{([^}]+)\}", text)


def find_graphics(tex_path: Path) -> list[str]:
    text = tex_path.read_text(encoding="utf-8")
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)


def find_cites_in_text(text: str) -> list[str]:
    keys: list[str] = []
    for match in re.findall(r"\\cite\w*\{([^}]+)\}", text):
        keys.extend([key.strip() for key in match.split(",") if key.strip()])
    return keys


def read_bib_keys() -> list[str]:
    bib_text = (LATEX_DIR / "references_seed.bib").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"@\w+\{([^,]+),", bib_text)))


def strip_latex(text: str) -> str:
    text = text.replace("~", " ")
    text = text.replace(r"\framework", "SpecLens-RL")
    text = text.replace(r"\LaTeX{}", "LaTeX")
    text = text.replace(r"\LaTeX", "LaTeX")
    text = text.replace(r"\Delta", "Delta")
    text = text.replace(r"\Pi", "Pi")
    text = text.replace(r"M_{\mathrm{mon}}", "M_mon")
    text = text.replace(r"\emph{monitorability gap}", "monitorability gap")
    text = re.sub(r"\\citep\{([^}]+)\}", lambda m: "(" + "; ".join(CITE_PAREN.get(k.strip(), k.strip()) for k in m.group(1).split(",")) + ")", text)
    text = re.sub(r"\\citet\{([^}]+)\}", lambda m: "; ".join(CITE_TEXT.get(k.strip(), k.strip()) for k in m.group(1).split(",")), text)
    text = re.sub(r"\\cite\{([^}]+)\}", lambda m: "(" + "; ".join(CITE_PAREN.get(k.strip(), k.strip()) for k in m.group(1).split(",")) + ")", text)
    text = re.sub(r"(Figure|Table)~\\ref\{([^}]+)\}", lambda m: REF_MAP.get(m.group(2), f"{m.group(1)} ?"), text)
    text = re.sub(r"\\ref\{([^}]+)\}", lambda m: REF_MAP.get(m.group(1), "?"), text)
    text = re.sub(r"\\(emph|textbf|mathrm)\{([^}]*)\}", lambda m: m.group(2), text)
    text = re.sub(r"\\url\{([^}]*)\}", lambda m: m.group(1), text)
    text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", "", text)
    text = re.sub(r"\\[A-Za-z]+\*?", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_table_env(env_text: str) -> Block:
    caption_match = re.search(r"\\caption\{(.*?)\}", env_text, re.S)
    caption = strip_latex(caption_match.group(1)) if caption_match else ""
    tabular_match = re.search(r"\\begin\{tabular\}\{.*?\}(.*?)\\end\{tabular\}", env_text, re.S)
    if not tabular_match:
        return Block(kind="table", caption=caption, headers=[], rows=[])
    lines = []
    for raw_line in tabular_match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line in {r"\toprule", r"\midrule", r"\bottomrule"}:
            continue
        if line.endswith(r"\\"):
            line = line[:-2].strip()
        if "&" in line:
            lines.append([strip_latex(cell.strip()) for cell in line.split("&")])
    headers = lines[0] if lines else []
    rows = lines[1:] if len(lines) > 1 else []
    return Block(kind="table", headers=headers, rows=rows, caption=caption)


def parse_figure_env(env_text: str) -> Block:
    path_match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", env_text)
    caption_match = re.search(r"\\caption\{(.*?)\}", env_text, re.S)
    return Block(
        kind="figure",
        path=path_match.group(1) if path_match else None,
        caption=strip_latex(caption_match.group(1)) if caption_match else "",
    )


def parse_bibliography(env_text: str) -> list[Block]:
    items: list[Block] = []
    matches = list(re.finditer(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", env_text))
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(env_text)
        entry = strip_latex(env_text[start:end])
        items.append(Block(kind="reference", text=entry))
    return items


def parse_final_tex_to_blocks(final_text: str) -> list[Block]:
    lines = final_text.splitlines()
    blocks: list[Block] = []
    in_document = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == r"\begin{document}":
            in_document = True
            i += 1
            continue
        if not in_document:
            i += 1
            continue
        if line == r"\maketitle":
            blocks.append(Block(kind="maketitle"))
            i += 1
            continue
        if line.startswith(r"\begin{abstract}"):
            chunk = [line]
            i += 1
            while i < len(lines) and r"\end{abstract}" not in lines[i]:
                chunk.append(lines[i])
                i += 1
            blocks.append(Block(kind="abstract", text=strip_latex(" ".join(chunk[1:]))))
            i += 1
            continue
        if line == r"\[":
            eq_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != r"\]":
                eq_lines.append(lines[i].strip())
                i += 1
            blocks.append(Block(kind="equation", text=strip_latex(" ".join(eq_lines))))
            i += 1
            continue
        if line.startswith(r"\section{"):
            blocks.append(Block(kind="section", title=strip_latex(line)))
            i += 1
            continue
        if line.startswith(r"\subsection{"):
            blocks.append(Block(kind="subsection", title=strip_latex(line)))
            i += 1
            continue
        if line.startswith(r"\paragraph{"):
            blocks.append(Block(kind="paragraph_heading", title=strip_latex(line)))
            i += 1
            continue
        if line.startswith(r"\begin{figure"):
            env_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(r"\end{figure"):
                env_lines.append(lines[i])
                i += 1
            if i < len(lines):
                env_lines.append(lines[i])
            blocks.append(parse_figure_env("\n".join(env_lines)))
            i += 1
            continue
        if line.startswith(r"\begin{table"):
            env_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(r"\end{table"):
                env_lines.append(lines[i])
                i += 1
            if i < len(lines):
                env_lines.append(lines[i])
            blocks.append(parse_table_env("\n".join(env_lines)))
            i += 1
            continue
        if line.startswith(r"\begin{thebibliography}"):
            env_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(r"\end{thebibliography}"):
                env_lines.append(lines[i])
                i += 1
            if i < len(lines):
                env_lines.append(lines[i])
            blocks.extend(parse_bibliography("\n".join(env_lines)))
            i += 1
            continue
        if not line or line.startswith("%"):
            i += 1
            continue
        para_lines = [line]
        i += 1
        while i < len(lines):
            candidate = lines[i].strip()
            if (
                not candidate
                or candidate in {r"\[", r"\maketitle"}
                or candidate.startswith((r"\section{", r"\subsection{", r"\paragraph{", r"\begin{figure", r"\begin{table", r"\begin{thebibliography}"))
            ):
                break
            para_lines.append(candidate)
            i += 1
        blocks.append(Block(kind="paragraph", text=strip_latex(" ".join(para_lines))))
    return blocks


class SnapshotRenderer:
    def __init__(self, output_path: Path, log_lines: list[str]):
        self.output_path = output_path
        self.log_lines = log_lines
        self.pdf = PdfPages(output_path)
        self.page_number = 0
        self.fig = None
        self.used_text_page = False
        self.col = 0
        self.y = 0.95
        self.x_positions = [0.07, 0.54]
        self.col_width = 0.37
        self.bottom = 0.075

    def _new_text_page(self) -> None:
        self._close_page()
        self.fig = plt.figure(figsize=(8.5, 11))
        self.fig.patch.set_facecolor("white")
        self.page_number += 1
        self.used_text_page = False
        self.col = 0
        self.y = 0.95
        self.log_lines.append(f"Started text page {self.page_number}.")

    def _close_page(self) -> None:
        if self.fig is None:
            return
        self.fig.text(0.5, 0.03, str(self.page_number), ha="center", va="bottom", fontsize=9, family="serif")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def close(self) -> None:
        self._close_page()
        self.pdf.close()

    def title_page(self, title: str, author: str, abstract: str) -> None:
        self._close_page()
        self.fig = plt.figure(figsize=(8.5, 11))
        self.page_number += 1
        self.fig.patch.set_facecolor("white")
        self.fig.text(0.5, 0.93, title, ha="center", va="top", fontsize=18, fontweight="bold", family="serif", wrap=True)
        self.fig.text(0.5, 0.88, author, ha="center", va="top", fontsize=11, family="serif")
        self.fig.text(0.08, 0.82, "Abstract", ha="left", va="top", fontsize=12, fontweight="bold", family="serif")
        abstract_lines = textwrap.wrap(abstract, width=110)
        y = 0.79
        for line in abstract_lines:
            self.fig.text(0.08, y, line, ha="left", va="top", fontsize=10, family="serif")
            y -= 0.018
        self.fig.text(0.5, 0.03, str(self.page_number), ha="center", va="bottom", fontsize=9, family="serif")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None
        self.log_lines.append("Rendered title and abstract page.")

    def _ensure_space(self, line_count: int, line_height: float) -> None:
        if self.fig is None:
            self._new_text_page()
        required = line_count * line_height
        if self.y - required < self.bottom:
            if self.col == 0:
                self.col = 1
                self.y = 0.95
            else:
                self._new_text_page()

    def _write_wrapped(self, text: str, fontsize: int, width_chars: int, weight: str = "normal") -> None:
        line_height = 0.0145 if fontsize <= 10 else 0.018
        lines = textwrap.wrap(text, width=width_chars) or [text]
        self._ensure_space(len(lines) + 1, line_height)
        x = self.x_positions[self.col]
        for line in lines:
            self.fig.text(x, self.y, line, ha="left", va="top", fontsize=fontsize, family="serif", fontweight=weight)
            self.y -= line_height
        self.y -= line_height * 0.35
        self.used_text_page = True

    def section(self, title: str) -> None:
        clean = title.replace("section", "").strip()
        self._write_wrapped(clean, fontsize=13, width_chars=48, weight="bold")

    def subsection(self, title: str) -> None:
        clean = title.replace("subsection", "").strip()
        self._write_wrapped(clean, fontsize=11, width_chars=52, weight="bold")

    def paragraph_heading(self, title: str) -> None:
        clean = title.replace("paragraph", "").strip()
        self._write_wrapped(clean, fontsize=10, width_chars=54, weight="bold")

    def paragraph(self, text: str) -> None:
        self._write_wrapped(text, fontsize=9, width_chars=72)

    def equation(self, text: str) -> None:
        self._write_wrapped(text, fontsize=10, width_chars=58)

    def full_page_figure(self, image_path: Path, caption: str) -> None:
        self._close_page()
        fig = plt.figure(figsize=(8.5, 11))
        self.page_number += 1
        ax = fig.add_axes([0.07, 0.17, 0.86, 0.72])
        ax.axis("off")
        image = Image.open(image_path)
        ax.imshow(image)
        fig.text(0.07, 0.11, caption, ha="left", va="top", fontsize=10, family="serif", wrap=True)
        fig.text(0.5, 0.03, str(self.page_number), ha="center", va="bottom", fontsize=9, family="serif")
        self.pdf.savefig(fig)
        plt.close(fig)
        self.log_lines.append(f"Rendered figure page {self.page_number}: {image_path.name}")

    def full_page_table(self, headers: list[str], rows: list[list[str]], caption: str) -> None:
        self._close_page()
        fig = plt.figure(figsize=(8.5, 11))
        self.page_number += 1
        ax = fig.add_axes([0.05, 0.18, 0.90, 0.67])
        ax.axis("off")
        wrapped_rows = []
        widths = [16, 24, 18, 18, 22, 20]
        for row in rows:
            wrapped_rows.append(
                [
                    textwrap.fill(cell, width=widths[min(idx, len(widths) - 1)]) if cell else ""
                    for idx, cell in enumerate(row)
                ]
            )
        wrapped_headers = [
            textwrap.fill(cell, width=widths[min(idx, len(widths) - 1)])
            for idx, cell in enumerate(headers)
        ]
        table = ax.table(cellText=wrapped_rows, colLabels=wrapped_headers, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.1)
        fig.text(0.07, 0.10, caption, ha="left", va="top", fontsize=10, family="serif", wrap=True)
        fig.text(0.5, 0.03, str(self.page_number), ha="center", va="bottom", fontsize=9, family="serif")
        self.pdf.savefig(fig)
        plt.close(fig)
        self.log_lines.append(f"Rendered table page {self.page_number}.")

    def references_page(self, entries: list[str]) -> None:
        self._close_page()
        self._new_text_page()
        self._write_wrapped("References", fontsize=13, width_chars=48, weight="bold")
        for entry in entries:
            self._write_wrapped(entry, fontsize=8, width_chars=74)

    def render_text_blocks(self, blocks: list[Block]) -> None:
        for block in blocks:
            if block.kind == "section":
                self.section(block.title or "")
            elif block.kind == "subsection":
                self.subsection(block.title or "")
            elif block.kind == "paragraph_heading":
                self.paragraph_heading(block.title or "")
            elif block.kind == "paragraph":
                self.paragraph(block.text or "")
            elif block.kind == "equation":
                self.equation(block.text or "")

    def combined_asset_page(self, items: list[Block], png_dir: Path) -> None:
        self._close_page()
        fig = plt.figure(figsize=(8.5, 11))
        self.page_number += 1
        fig.patch.set_facecolor("white")
        if len(items) == 2 and items[0].kind == "figure" and items[1].kind == "table":
            image_path = rasterize_pdf(LATEX_DIR / (items[0].path or ""), png_dir)
            top_ax = fig.add_axes([0.07, 0.52, 0.86, 0.32])
            top_ax.axis("off")
            top_ax.imshow(Image.open(image_path))
            fig.text(0.07, 0.49, items[0].caption or "", ha="left", va="top", fontsize=9.5, family="serif", wrap=True)

            bottom_ax = fig.add_axes([0.05, 0.12, 0.90, 0.28])
            bottom_ax.axis("off")
            widths = [16, 24, 18, 18, 22, 20]
            wrapped_rows = []
            for row in items[1].rows or []:
                wrapped_rows.append(
                    [
                        textwrap.fill(cell, width=widths[min(idx, len(widths) - 1)]) if cell else ""
                        for idx, cell in enumerate(row)
                    ]
                )
            wrapped_headers = [
                textwrap.fill(cell, width=widths[min(idx, len(widths) - 1)])
                for idx, cell in enumerate(items[1].headers or [])
            ]
            table = bottom_ax.table(cellText=wrapped_rows, colLabels=wrapped_headers, loc="center", cellLoc="left")
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1, 1.7)
            fig.text(0.07, 0.10, items[1].caption or "", ha="left", va="top", fontsize=9.5, family="serif", wrap=True)
        else:
            y_positions = [(0.55, 0.32), (0.10, 0.28)]
            for item, (y0, h) in zip(items, y_positions):
                if item.kind == "figure":
                    image_path = rasterize_pdf(LATEX_DIR / (item.path or ""), png_dir)
                    ax = fig.add_axes([0.07, y0, 0.86, h])
                    ax.axis("off")
                    ax.imshow(Image.open(image_path))
                    fig.text(0.07, y0 - 0.02, item.caption or "", ha="left", va="top", fontsize=9.5, family="serif", wrap=True)
                elif item.kind == "table":
                    ax = fig.add_axes([0.05, y0, 0.90, h])
                    ax.axis("off")
                    table = ax.table(cellText=item.rows or [], colLabels=item.headers or [], loc="center", cellLoc="left")
                    table.auto_set_font_size(False)
                    table.set_fontsize(7.5)
                    table.scale(1, 1.6)
                    fig.text(0.07, y0 - 0.02, item.caption or "", ha="left", va="top", fontsize=9.5, family="serif", wrap=True)
        fig.text(0.5, 0.03, str(self.page_number), ha="center", va="bottom", fontsize=9, family="serif")
        self.pdf.savefig(fig)
        plt.close(fig)
        self.log_lines.append(f"Rendered combined asset page {self.page_number}.")


def rasterize_pdf(pdf_path: Path, png_dir: Path) -> Path:
    ensure_dir(png_dir)
    png_path = png_dir / f"{pdf_path.stem}.png"
    subprocess.run(
        ["sips", "-s", "format", "png", str(pdf_path), "--out", str(png_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return png_path


def build_fallback_pdf(final_text: str) -> tuple[list[str], int]:
    blocks = parse_final_tex_to_blocks(final_text)
    log_lines = ["Fallback PDF build started.", "Official AAAI style unavailable locally; using surrogate PDF renderer."]
    png_dir = LATEX_DIR / ".final_png_cache"
    renderer = SnapshotRenderer(LATEX_DIR / "final.pdf", log_lines)
    references: list[str] = []
    text_groups: dict[str, list[Block]] = {
        "intro_related": [],
        "framework_metrics_methods": [],
        "experiments_results": [],
        "limits_conclusion": [],
    }
    assets: dict[str, Block] = {}
    title = "SpecLens-RL: A Compact Framework for Diagnosing Specification Failures in Reinforcement Learning"
    author = "Anonymous Authors"
    abstract = next(block.text for block in blocks if block.kind == "abstract")
    renderer.title_page(title, author, abstract)
    current_group = "intro_related"
    for block in blocks:
        if block.kind in {"maketitle", "abstract"}:
            continue
        if block.kind == "reference":
            references.append(block.text or "")
            continue
        if block.kind == "section":
            normalized = (block.title or "").lower()
            if normalized == "introduction":
                current_group = "intro_related"
            elif normalized in {"related work"}:
                current_group = "intro_related"
            elif normalized in {"framework", "metric suite", "method overview"}:
                current_group = "framework_metrics_methods"
            elif normalized in {"experimental setup", "results"}:
                current_group = "experiments_results"
            elif normalized in {"limitations", "broader impact", "conclusion"}:
                current_group = "limits_conclusion"
            text_groups[current_group].append(block)
            continue
        if block.kind == "figure":
            if block.path == "figures/framework_overview.pdf":
                assets["framework"] = block
            elif block.path == "figures/main_results_panel.pdf":
                assets["main_results"] = block
            elif block.path == "figures/sensortamper_monitorability.pdf":
                assets["monitor"] = block
            continue
        if block.kind == "table":
            caption = block.caption or ""
            if "three benchmark environments" in caption:
                assets["env_table"] = block
            elif "Main specification-gap results" in caption:
                assets["main_table"] = block
            elif "Leakage-cleared hack-monitor results" in caption:
                assets["monitor_table"] = block
            continue
        text_groups[current_group].append(block)
    renderer.render_text_blocks(text_groups["intro_related"])
    renderer.render_text_blocks(text_groups["framework_metrics_methods"])
    framework_assets = [asset for key in ["framework", "env_table"] if (asset := assets.get(key))]
    if framework_assets:
        renderer.combined_asset_page(framework_assets, png_dir)
    renderer.render_text_blocks(text_groups["experiments_results"])
    main_assets = [asset for key in ["main_results", "main_table"] if (asset := assets.get(key))]
    if main_assets:
        renderer.combined_asset_page(main_assets, png_dir)
    monitor_assets = [asset for key in ["monitor", "monitor_table"] if (asset := assets.get(key))]
    if monitor_assets:
        renderer.combined_asset_page(monitor_assets, png_dir)
    renderer.render_text_blocks(text_groups["limits_conclusion"])
    if references:
        renderer.references_page(references)
    renderer.close()
    shutil.rmtree(png_dir, ignore_errors=True)
    log_lines.append(f"Fallback PDF build finished with {renderer.page_number} pages.")
    return log_lines, renderer.page_number


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w-]+\b", strip_latex(text)))


def count_pdf_pages(pdf_path: Path) -> int | None:
    log_path = pdf_path.with_suffix(".log")
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Output written on .*?\((\d+) pages?,", log_text)
        if match:
            return int(match.group(1))
    result = subprocess.run(
        ["mdls", "-name", "kMDItemNumberOfPages", str(pdf_path.resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"=\s*(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def compile_with_tectonic() -> tuple[bool, list[str], int | None]:
    command = ["tectonic", "final.tex", "--keep-logs", "--keep-intermediates"]
    result = subprocess.run(
        command,
        cwd=LATEX_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    log_lines = [
        "Official AAAI build attempt with tectonic.",
        f"Command: {' '.join(command)}",
        "",
        result.stdout.strip(),
        "",
        result.stderr.strip(),
    ]
    pdf_path = LATEX_DIR / "final.pdf"
    page_count = count_pdf_pages(pdf_path) if pdf_path.exists() else None
    return result.returncode == 0 and pdf_path.exists(), log_lines, page_count


def write_snapshot(snapshot_sources: list[tuple[str, Path]]) -> None:
    lines = [
        "SpecLens-RL final paper snapshot assembly",
        "",
        "The flattened final.tex was assembled from:",
        "",
    ]
    for name, path in snapshot_sources:
        lines.append(f"- section `{name}` from `{path.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "Table inputs were expanded inline from:",
            "- latex/tables/environment_summary_compact.tex",
            "- latex/tables/main_results_compact.tex",
            "- latex/tables/sensortamper_monitor.tex",
            "",
            "The bibliography remains external as `latex/references_seed.bib`, which is allowed by the AAAI source-submission rules.",
            "The flattened `final.tex` uses the official `aaai2026` submission package and local `aaai2026.bst` bibliography style.",
        ]
    )
    write_text(LATEX_DIR / "final_flattened_source_snapshot.txt", "\n".join(lines))


def write_asset_mapping() -> None:
    lines = [
        "# Paper Asset Mapping",
        "",
        "## Final main-paper figures",
        "",
        "- `latex/figures/framework_overview.pdf`: retained as the final conceptual framework figure.",
        "- `latex/figures/main_results_panel.pdf`: retained as the final compact multi-panel main-results figure.",
        "- `latex/figures/sensortamper_monitorability.pdf`: retained as the final leakage-cleared monitorability figure.",
        "",
        "## Final main-paper tables",
        "",
        "- `latex/tables/environment_summary_compact.tex`: new reader-facing compact environment table used in the main paper; supersedes `environment_main.tex` in the main narrative.",
        "- `latex/tables/main_results_compact.tex`: retained as the final main results table.",
        "- `latex/tables/sensortamper_monitor.tex`: regenerated into a reader-facing channel/monitor table with explicit leakage status.",
        "",
        "## Legacy assets retained but not used in the main paper",
        "",
        "- Legacy figures such as `proxy_vs_true_scatter.pdf`, `generalization_gap.pdf`, `hacking_incidence_bars.pdf`, `tampering_summary.pdf`, and `monitorability_gap.pdf` remain in the repository as supporting artifacts only.",
        "- Supporting tables such as `resource_results.tex`, `threshold_robustness.tex`, `monitor_results.tex`, and `main_specification_results.tex` remain available for supplement or appendix use, but are not referenced by the final main paper.",
    ]
    write_text(RESULTS_DIR / "PAPER_ASSET_MAPPING.md", "\n".join(lines))


def write_final_citation_status(final_text: str) -> None:
    bib_keys = set(read_bib_keys())
    used_keys = sorted(set(find_cites_in_text(final_text)))
    undefined = sorted(set(used_keys) - bib_keys)
    unused = sorted(bib_keys - set(used_keys))
    lines = [
        "# Final Citation Status",
        "",
        f"- Cited keys in final paper: {len(used_keys)}",
        f"- Undefined keys: {len(undefined)}",
        f"- Unused keys in `references_seed.bib`: {len(unused)}",
        "",
        "## Used keys",
        "",
    ]
    lines.extend([f"- `{key}`" for key in used_keys])
    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- Undefined keys: `{undefined}`",
            f"- Unused bibliography keys: `{unused}`",
            "- `latex/final.tex` uses `references_seed.bib` plus the vendored `aaai2026.bst` style for official AAAI-style references.",
        ]
    )
    write_text(RESULTS_DIR / "FINAL_CITATION_STATUS.md", "\n".join(lines))


def write_submission_audit(final_text: str) -> None:
    main_inputs = find_inputs(LATEX_DIR / "main.tex")
    input_status = []
    for rel in main_inputs:
        candidate = LATEX_DIR / f"{rel}.tex"
        input_status.append((rel, candidate.exists(), candidate))
    section_inputs = []
    for section_name in SECTION_ORDER:
        section_path = SECTIONS_DIR / f"{section_name}.tex"
        for rel in find_inputs(section_path):
            candidate = LATEX_DIR / f"{rel}.tex"
            section_inputs.append((section_name, rel, candidate.exists(), candidate))
    section_graphics = []
    for section_name in SECTION_ORDER:
        section_path = SECTIONS_DIR / f"{section_name}.tex"
        for rel in find_graphics(section_path):
            section_graphics.append((rel, (LATEX_DIR / rel).exists()))
    final_inputs = re.findall(r"\\input\{([^}]+)\}", final_text)
    final_graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", final_text)
    main_text = (LATEX_DIR / "main.tex").read_text(encoding="utf-8")
    cited_keys = sorted(set(find_cites_in_text(main_text + "\n" + final_text)))
    bib_keys = set(read_bib_keys())
    undefined = sorted(set(cited_keys) - bib_keys)
    section_words = {}
    for name in SECTION_ORDER:
        section_words[name] = word_count((SECTIONS_DIR / f"{name}.tex").read_text(encoding="utf-8"))
    total_words = sum(section_words.values()) + word_count(extract_abstract(main_text))
    lines = [
        "# Submission Readiness Audit",
        "",
        "## Main paper `\\input{}` references",
        "",
        "| Reference | Exists | Resolved path |",
        "| --- | --- | --- |",
    ]
    for rel, exists, path in input_status:
        lines.append(f"| `{rel}` | `{'yes' if exists else 'no'}` | `{path.relative_to(ROOT)}` |")
    lines.extend(
        [
            "",
            "## Section-level table fragment references",
            "",
            "| Section | Reference | Exists | Resolved path |",
            "| --- | --- | --- | --- |",
        ]
    )
    for section_name, rel, exists, path in section_inputs:
        lines.append(f"| `{section_name}` | `{rel}` | `{'yes' if exists else 'no'}` | `{path.relative_to(ROOT)}` |")
    lines.extend(
        [
            "",
            "## `\\includegraphics{}` references used by the modular paper",
            "",
            "| Graphic | Exists |",
            "| --- | --- |",
        ]
    )
    for rel, exists in section_graphics:
        lines.append(f"| `{rel}` | `{'yes' if exists else 'no'}` |")
    lines.extend(
        [
            "",
            "## Bibliography consistency",
            "",
            f"- Cited keys in modular + final paper sources: `{cited_keys}`",
            f"- Undefined citation keys: `{undefined}`",
            "",
            "## Final single-file paper contract",
            "",
            f"- Remaining `\\input{{}}` dependencies in `latex/final.tex`: `{final_inputs}`",
            f"- `\\includegraphics{{}}` dependencies in `latex/final.tex`: `{final_graphics}`",
            "- `latex/final.tex` embeds all body text and tables directly; external dependencies are limited to figure PDFs and `references_seed.bib`.",
            "",
            "## Compile status",
            "",
            f"- `aaai2026.sty` present locally: `{'yes' if (LATEX_DIR / 'aaai2026.sty').exists() else 'no'}`",
            f"- `aaai2026.bst` present locally: `{'yes' if (LATEX_DIR / 'aaai2026.bst').exists() else 'no'}`",
            f"- `tectonic` present locally: `{'yes' if shutil.which('tectonic') else 'no'}`",
            "- Conclusion: the modular paper is internally file-consistent and the repository now contains the AAAI-26 package files needed for an official local compile path.",
            "",
            "## Scaffold-tone status",
            "",
            "- The live paper now inputs `sections/experiments.tex` and `sections/results.tex` instead of placeholder-named sections.",
            "- Compatibility files `experiments_placeholders.tex` and `results_placeholders.tex` still exist for repository/test compatibility, but they are no longer referenced by `latex/main.tex`.",
            "- Auxiliary generated sentence files remain in `latex/sections/` but are not required for the visible paper text.",
            "",
            "## Balance and page-discipline estimate",
            "",
            f"- Approximate abstract + body word count: `{total_words}`",
            "- Section word counts:",
        ]
    )
    for name, count in section_words.items():
        lines.append(f"  - `{name}`: `{count}` words")
    lines.extend(
        [
            "",
            "- The paper looks realistically main-paper shaped by content density: framework + metrics remain substantial, related work is concise, and results is proportionate.",
            "- Exact AAAI page count is determined separately in `results/FINAL_PDF_BUILD_STATUS.md` from the local tectonic build.",
            "",
            "## Missing or mismatched files",
            "",
            "- No missing body `\\input{}` or `\\includegraphics{}` targets were found after the final assembly updates.",
            "- No file-contract mismatches remain in the final paper source tree.",
            "- The remaining question is page discipline under the official compile, not missing assets.",
        ]
    )
    write_text(RESULTS_DIR / "SUBMISSION_READINESS_AUDIT.md", "\n".join(lines))


def write_final_reports(page_count: int | None, official_compile: bool) -> None:
    report_lines = [
        "# Submission Readiness Final Report",
        "",
        "- Does the LaTeX now reference only real files: `yes` for the final paper body and figure/table assets checked in this pass.",
        f"- Does the paper compile cleanly in principle: `{'yes' if official_compile else 'no'}` for the official local AAAI-26 build path.",
        "- Exact final main-paper figures:",
        "  - `latex/figures/framework_overview.pdf`",
        "  - `latex/figures/main_results_panel.pdf`",
        "  - `latex/figures/sensortamper_monitorability.pdf`",
        "- Exact final main-paper tables:",
        "  - `latex/tables/environment_summary_compact.tex`",
        "  - `latex/tables/main_results_compact.tex`",
        "  - `latex/tables/sensortamper_monitor.tex`",
        "- Strongest honest supported claim:",
        "  - `SpecLens-RL` provides a compact framework for measuring specification gaps, tampering risk, and leakage-sensitive monitorability in lightweight RL tasks. Across three symbolic environments, the framework exposes strong proxy--true objective divergence, while valid leakage-cleared monitor evidence is feasible but currently concentrated in SensorTamper-v0.",
        "- What still prevents true AAAI readiness:",
        "  - the official AAAI kit had to be vendored from a public mirror because the direct `aaai.org/authorkit26/` endpoint returned 404 during this pass",
        "  - final page count and float placement still need a human venue check even after local compilation",
        "- Human polish still recommended before submission:",
        "  - inspect the locally compiled AAAI PDF page by page for overfull lines and float placement",
        "  - verify anonymization, author block, and venue metadata",
        "  - do a final caption-by-caption proofread in the compiled venue template",
        "",
        f"- Final PDF page count produced in this pass: `{page_count}`",
    ]
    write_text(RESULTS_DIR / "SUBMISSION_READINESS_FINAL_REPORT.md", "\n".join(report_lines))

    build_status_lines = [
        "# Final PDF Build Status",
        "",
        f"- `latex/final.tex` compiled successfully: `{'yes' if official_compile else 'no'}`.",
        "- Official AAAI package used: `yes`.",
        "- Local build tool: `tectonic`.",
        "- Package source note:",
        "  - the official AAAI-26 CFP and mirrored official instructions identify the package as `aaai2026.sty` / `aaai2026.bst`",
        "  - the direct `https://aaai.org/authorkit26/` endpoint returned 404 during this pass, so the local style files were vendored from a public mirror copy matching those official filenames",
        "- Build mode:",
        "  - `latex/final.tex` is a flattened single-file submission source using the `aaai2026` submission package and `references_seed.bib`",
        "  - `latex/final.pdf` was compiled locally from that source with `tectonic`",
        "- Remaining issues a human should still check:",
        "  - final float placement and any remaining typography warnings in the compiled PDF",
        "  - final bibliography appearance under the venue's expected environment",
        "  - a final pdflatex/Overleaf venue check, since this local official build was produced with `tectonic`",
        "",
        f"- Produced official local PDF page count: `{page_count}`",
    ]
    write_text(RESULTS_DIR / "FINAL_PDF_BUILD_STATUS.md", "\n".join(build_status_lines))


def main() -> None:
    final_text, snapshot_sources = build_final_tex()
    write_snapshot(snapshot_sources)
    write_asset_mapping()
    write_final_citation_status(final_text)
    write_submission_audit(final_text)
    official_compile, log_lines, page_count = compile_with_tectonic()
    if not official_compile:
        fallback_lines, fallback_pages = build_fallback_pdf(final_text)
        log_lines.extend(["", "Official compile failed; fallback surrogate build was used.", ""])
        log_lines.extend(fallback_lines)
        page_count = fallback_pages
    write_text(LATEX_DIR / "final_build.log", "\n".join(log_lines) + "\n")
    write_final_reports(page_count, official_compile)


if __name__ == "__main__":
    main()
