"""Build a review-ready, Nature-structured LaTeX package from the Word sources.

This is deliberately a transparent conversion step: the Word documents remain
the scientific-content source, while the generated project is the source for
PDF review and for the eventual Nature production package.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
OUT = MANUSCRIPT / "nature_tex"
PANDOC = shutil.which("pandoc")


MAIN_PREAMBLE = r"""\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{booktabs,longtable,array,multirow,calc}
\usepackage{float}
\usepackage{setspace}
\usepackage{lineno}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\urlstyle{same}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setcounter{secnumdepth}{0}
\newcommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\makeatletter
\newsavebox\pandoc@box
\newcommand*\pandocbounded[1]{%
  \sbox\pandoc@box{#1}%
  \Gscale@div\@tempa{\textheight}{\dimexpr\ht\pandoc@box+\dp\pandoc@box\relax}%
  \Gscale@div\@tempb{\linewidth}{\wd\pandoc@box}%
  \ifdim\@tempb\p@<\@tempa\p@\let\@tempa\@tempb\fi%
  \ifdim\@tempa\p@<\p@\scalebox{\@tempa}{\usebox\pandoc@box}%
  \else\usebox\pandoc@box\fi%
}
\makeatother
"""


def run_pandoc(source: Path, destination: Path, media_dir: Path) -> str:
    if not PANDOC:
        raise RuntimeError("Pandoc is required but was not found on PATH.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            PANDOC,
            str(source),
            "--from",
            "docx",
            "--to",
            "latex",
            "--extract-media",
            str(media_dir),
            "--output",
            str(destination),
        ],
        check=True,
        cwd=OUT,
    )
    return destination.read_text(encoding="utf-8")


def clean_tex(text: str) -> str:
    """Remove Word bookmark wrappers while retaining visible citation numbers."""
    text = text.replace("Writing 每 original draft", "Writing -- original draft")
    text = text.replace("Writing 每 review & editing", "Writing -- review \\& editing")
    text = text.replace("\u00a0", " ")
    text = (
        text.replace("±", r"$\pm$")
        .replace("³", r"\textsuperscript{3}")
        .replace("Ω", r"$\Omega$")
        .replace("−", "-")
        .replace("≥", r"$\geq$")
        .replace("⋅", r"$\cdot$")
    )
    text = text.replace(str(OUT).replace("\\", "/") + "/", "")
    text = text.replace(r"\def\LTcaptype{none}", r"\def\LTcaptype{table}")
    text = text.replace("?utm_source=chatgpt.com", "")
    text = re.sub(
        r"\\href\{(https?://[^{}]+)\}\{[^{}]*\}", r"\\url{\1}", text
    )
    def wrap_bare_url(match: re.Match[str]) -> str:
        url = match.group(0)
        trailing = ""
        while url and url[-1] in ".,;:":
            trailing = url[-1] + trailing
            url = url[:-1]
        return r"\url{" + url + "}" + trailing

    text = re.sub(r"(?<![\\{])https?://[^\s}]+", wrap_bare_url, text)
    # Pandoc represents Word cross-reference fields as hyperlinks to internal
    # Word bookmarks. Nature needs the visible numerical citation only.
    text = re.sub(
        r"\\hyperref\[_Ref[^\]]+\]\{(\\textsuperscript\{[^{}]*\})\}",
        r"\1",
        text,
    )
    text = re.sub(r"\\hyperref\[_Ref[^\]]+\]\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\protect\\hypertarget\{[^}]*\}\{\}", "", text)
    text = re.sub(r"\\protect\\phantomsection\\label\{_Ref[^}]+\}", "", text)
    return text.replace("Fig 1", "Fig. 1").replace("Fig 2", "Fig. 2")


def after_heading(text: str, heading: str) -> str:
    marker = f"\n{heading}\n"
    if marker not in text:
        raise ValueError(f"Could not find the heading {heading!r} in converted text.")
    return text.rsplit(marker, 1)[1].lstrip()


def replace_line_headings(text: str, headings: dict[str, str]) -> str:
    for old, new in headings.items():
        text = re.sub(rf"(?m)^{re.escape(old)}$", lambda _: new, text)
    return text


def split_references(text: str) -> tuple[str, list[str]]:
    marker = "\nReferences\n"
    if marker not in text:
        return text, []
    before, refs = text.rsplit(marker, 1)
    entries = [entry.replace("\n", " ").strip() for entry in re.split(r"\n\s*\n", refs) if entry.strip()]
    return before.rstrip(), entries


def bibliography(entries: list[str]) -> str:
    """Flatten Pandoc's Word-numbered lists into a clean reference list."""
    start = 1
    normalized: list[str] = []
    for entry in entries:
        enumerated = re.search(r"(?s)\\begin\{enumerate\}(.*?)\\end\{enumerate\}", entry)
        if not enumerated:
            normalized.append(entry)
            continue

        content = enumerated.group(1)
        counter = re.search(r"\\setcounter\{enumi\}\{(\d+)\}", content)
        if counter:
            start = int(counter.group(1)) + 1
        for item in re.split(r"\\item\s*", content)[1:]:
            item = re.sub(r"^\{\}\s*", "", item).strip()
            if item:
                normalized.append(item)

    if not normalized:
        return ""
    numbered = [
        f"\\noindent {number}. {entry}\\par"
        for number, entry in enumerate(normalized, start=start)
    ]
    return "\n\n".join(["\\section*{References}", *numbered])


def normalize_extended_data_references(text: str) -> str:
    """Use the visible Extended Data numbering consistently, including wrapped text."""
    text = re.sub(
        r"\bFig(?:s)?\.?\s*S(\d+[a-z]?)",
        r"Extended Data Fig. \1",
        text,
    )
    text = re.sub(
        r"(\bExtended Data Figs?\.\s*\d+[a-z]?\s*(?:,|and)\s*)S(\d+[a-z]?)",
        r"\1\2",
        text,
    )
    text = re.sub(
        r"\bExtended Data Fig\. (\d+[a-z]?)(?:--|–|-)S(\d+[a-z]?)",
        r"Extended Data Figs. \1--\2",
        text,
    )
    return re.sub(
        r"\bExtended Data Fig\. (\d+[a-z]?)(\s*(?:,|and)\s*\d+[a-z]?)",
        r"Extended Data Figs. \1\2",
        text,
    )


def format_correspondence(text: str) -> str:
    """Keep both corresponding authors visible without allowing email overflow."""
    pattern = re.compile(
        r"Correspondence and requests for materials should be addressed to\s+"
        r"Hang\s+Chen\s*\(hchen117@uiowa\.edu\)"
    )
    replacement = (
        r"Correspondence and requests for materials should be addressed to\\"
        "\n"
        r"Rongwen Guo (\href{mailto:rongwenguo@csu.edu.cn}{\nolinkurl{rongwenguo@csu.edu.cn}}) or\\"
        "\n"
        r"Hang Chen (\href{mailto:hchen117@uiowa.edu}{\nolinkurl{hchen117@uiowa.edu}})."
    )
    return pattern.sub(lambda _: replacement, text)


def keep_main_figures_with_captions(text: str) -> str:
    """Start each main figure on a new page and keep its full legend intact."""
    pattern = re.compile(
        r"(?ms)^(\\includegraphics\[[^\n]+\]\{[^\n]+\})\n\n"
        r"(\\textbf\{Fig\. [1-4] .*?)(?=\n\n\\(?:subsection|section)\*\{|"
        r"\n\n\\textbf\{(?:Methods|Data availability|Code availability|Acknowledgements|Funding|Author contributions|Competing interests)|"
        r"\n\n\\includegraphics|\Z)"
    )

    def wrap(match: re.Match[str]) -> str:
        return (
            "\\clearpage\n\\noindent\\begin{minipage}{\\textwidth}\n\\begin{center}\n"
            + match.group(1)
            + "\n\\end{center}\n"
            + match.group(2).rstrip()
            + "\n\\end{minipage}"
        )

    return pattern.sub(wrap, text)


def format_extended_data_figures(text: str) -> str:
    """Put each Extended Data figure and its legend on a centered standalone page."""
    section_pattern = re.compile(
        r"(?ms)(\\section\*\{Extended Data Figures\})\n\n(.*?)(?=\n\n\\section\*\{Extended Data Tables\}|\Z)"
    )
    section = section_pattern.search(text)
    if not section:
        return text

    figure_pattern = re.compile(
        r"(?ms)^(\\includegraphics\[[^\n]+\]\{[^\n]+\})\n\n"
        r"(Extended Data Fig\. \d+ .*?)(?=\n\n\\includegraphics|\Z)"
    )
    figures = list(figure_pattern.finditer(section.group(2)))
    if not figures:
        return text

    pages = []
    for index, figure in enumerate(figures):
        pages.append(
            ("" if index == 0 else "\\clearpage\n")
            + "\\begin{samepage}\n"
            "\\begin{center}\n"
            + figure.group(1)
            + "\n\\end{center}\n"
            + figure.group(2).rstrip()
            + "\n\\end{samepage}"
        )

    replacement = "\\clearpage\n" + section.group(1) + "\n\n" + "\n\n".join(pages) + "\n\n\\clearpage"
    return text[: section.start()] + replacement + text[section.end() :]


def format_extended_data_tables(text: str) -> str:
    """Rebuild the three Extended Data tables with their intended hierarchy."""
    pattern = re.compile(
        r"(?ms)(\\section\*\{Extended Data Tables\}\n\n).*?\s*\Z"
    )
    tables = r"""\begingroup
\fontfamily{ptm}\selectfont
\fontsize{11}{12.4}\selectfont
\singlespacing
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.0}

\noindent Extended Data Table 1 \textbar{} Performance of the GNN models across prediction intervals

\vspace{0.35em}
\begin{center}
\begin{tabular}{@{}
  >{\centering\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.1428}}
  >{\centering\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.1428}}
  >{\centering\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.1429}}
  >{\centering\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.1429}}
  >{\centering\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.1429}}
  >{\centering\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.1429}}
  >{\centering\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.1429}}@{}}
\toprule
\multirow{2}{*}{\emph{H}} & \multicolumn{3}{c}{Temporal test} & \multicolumn{3}{c}{Spatial test} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
& RMSE (m) & \emph{r} & NSE & RMSE (m) & \emph{r} & NSE \\
\midrule
1 & 0.314 & 0.676 & 0.444 & 0.411 & 0.638 & 0.391 \\
3 & 0.584 & 0.773 & 0.575 & 0.783 & 0.663 & 0.430 \\
6 & 0.820 & 0.819 & 0.663 & 0.788 & 0.812 & 0.638 \\
\bottomrule
\end{tabular}
\end{center}

\vspace{1.1em}
\noindent Extended Data Table 2 \textbar{} Target percentile ranks and metric weights used to define the response prototypes. Values are target percentile ranks; values in parentheses are metric weights.

\vspace{0.35em}
\begin{center}
\begin{tabular}{@{}
  >{\raggedright\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.19}}
  >{\centering\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.16}}
  >{\centering\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.15}}
  >{\centering\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.18}}
  >{\centering\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.17}}
  >{\centering\arraybackslash}p{(\linewidth - 10\tabcolsep) * \real{0.15}}@{}}
\toprule
Response\newline group & Maximum\newline decline & Decline\newline rate & Early recovery\newline ratio & Half-recovery\newline time & Long-term\newline deficit \\
\midrule
Fast recovery & 0.78 (0.20) & 0.65 (0.16) & 0.88 (0.32) & 0.12 (0.24) & 0.35 (0.08) \\
Slow recovery & 0.62 (0.15) & 0.55 (0.10) & 0.25 (0.25) & 0.88 (0.35) & 0.80 (0.15) \\
Buffered & 0.10 (0.35) & 0.12 (0.30) & 0.60 (0.08) & 0.28 (0.07) & 0.12 (0.20) \\
\bottomrule
\end{tabular}
\end{center}

\clearpage
\noindent Extended Data Table 3 \textbar{} Drought-response metrics across the three response classes

\vspace{0.35em}
\renewcommand{\arraystretch}{1.14}
\begin{center}
\begin{tabular}{@{}
  >{\raggedright\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.19}}
  >{\raggedright\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.35}}
  >{\centering\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.10}}
  >{\centering\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.17}}
  >{\centering\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.19}}@{}}
\toprule
Response class & Metric & Median & Q25-Q75 & Q5-Q95 \\
\midrule
\multirow{5}{*}{Fast recovery} & Maximum decline (m) & 2.025 & 1.738--2.437 & 1.362--3.763 \\
& Decline rate (m/month) & 0.305 & 0.254--0.388 & 0.218--0.640 \\
& Early recovery ratio & 0.882 & 0.816--0.962 & 0.723--1.081 \\
& Half-recovery time (month) & 3.000 & 2.000--4.000 & 1.000--5.000 \\
& Long-term recovery ratio & 1.052 & 0.765--1.475 & 0.264--2.164 \\
\multirow{5}{*}{Slow recovery} & Maximum decline (m) & 1.347 & 1.110--1.699 & 0.885--2.555 \\
& Decline rate (m/month) & 0.325 & 0.267--0.378 & 0.184--0.466 \\
& Early recovery ratio & 0.508 & 0.358--0.644 & 0.173--0.767 \\
& Half-recovery time (month) & 8.000 & 6.000--58.000 & 4.000--67.000 \\
& Long-term recovery ratio & 1.193 & 0.611--1.689 & -0.837--3.392 \\
\multirow{5}{*}{Buffered} & Maximum decline (m) & 1.035 & 0.885--1.204 & 0.693--1.455 \\
& Decline rate (m/month) & 0.165 & 0.131--0.207 & 0.093--0.282 \\
& Early recovery ratio & 0.660 & 0.527--0.791 & 0.348--1.097 \\
& Half-recovery time (month) & 5.000 & 3.000--7.000 & 2.000--32.000 \\
& Long-term recovery ratio & 2.272 & 1.827--2.781 & 0.582--4.408 \\
\bottomrule
\end{tabular}
\end{center}
\endgroup
\clearpage"""

    return pattern.sub(lambda match: match.group(1) + tables + "\n", text)


def make_main_body(raw: str) -> str:
    text = after_heading(clean_tex(raw), "Abstract")
    summary, separator, remainder = text.partition("\n\n")
    if not separator:
        raise ValueError("Could not isolate the summary paragraph.")
    text = "\\section*{Summary paragraph}\n\n\\noindent\\textbf{" + summary + "}\n\n" + remainder
    before_refs, refs = split_references(text)
    before_refs = replace_line_headings(
        before_refs,
        {
            "Introduction": r"\section*{Introduction}",
            "Results": r"\section*{Results}",
            "Geophysics-informed groundwater reconstruction": r"\subsection*{Geophysics-informed groundwater reconstruction}",
            "Extreme drought reveals uneven groundwater responses": r"\subsection*{Extreme drought reveals uneven groundwater responses}",
            "Response metrics reveal slow-recovery zones": r"\subsection*{Response metrics reveal slow-recovery zones}",
            "Discussion": r"\section*{Discussion}",
            "Subsurface Structure and Pumping Organize Slow Recovery": r"\subsection*{Subsurface structure and pumping organize slow recovery}",
            "Implications for groundwater management": r"\subsection*{Implications for groundwater management}",
            "Methods": r"\section*{Online Methods}",
            "Airborne electromagnetic resistivity": r"\subsection*{Airborne electromagnetic resistivity}",
            "Graph neural-network reconstruction of monthly WTD": r"\subsection*{Graph neural-network reconstruction of monthly WTD}",
            "Prediction of slow-recovery probability": r"\subsection*{Prediction of slow-recovery probability}",
            "Data availability": r"\section*{Data availability}",
            "Code availability": r"\section*{Code availability}",
            "Acknowledgements": r"\section*{Acknowledgements}",
            "Funding": r"\section*{Funding}",
            "Author contributions": r"\section*{Author contributions}",
            "Competing interests": r"\section*{Competing interests}",
            "Supplementary Information statement": r"\section*{Additional information}",
            "Materials and correspondence statement": "",
        },
    )
    before_refs = before_refs.replace(
        r"\textbf{Supplementary Information statement}", r"\section*{Additional information}"
    ).replace(r"\textbf{Materials and correspondence statement}", "")
    before_refs = re.sub(r"\bTables S(\d+)", r"Extended Data Tables \1", before_refs)
    before_refs = re.sub(r"\bTable S(\d+)", r"Extended Data Table \1", before_refs)
    before_refs = normalize_extended_data_references(before_refs)
    before_refs = format_correspondence(before_refs)
    return keep_main_figures_with_captions(before_refs) + "\n\n" + bibliography(refs) + "\n"


def make_supplement_body(raw: str) -> str:
    text = after_heading(clean_tex(raw), "Materials and Methods")
    before_refs, refs = split_references(text)
    before_refs = replace_line_headings(
        before_refs,
        {
            "Figs. S1 to S20": r"\section*{Extended Data Figures}",
            "Tables S1 to S3": r"\section*{Extended Data Tables}",
        },
    )
    before_refs = re.sub(r"\bTables S(\d+)", r"Extended Data Tables \1", before_refs)
    before_refs = re.sub(r"\bTable S(\d+)", r"Extended Data Table \1", before_refs)
    before_refs = re.sub(
        r"(?m)^S1\.(\d+) (.+)$",
        lambda match: rf"\subsection*{{S1.{match.group(1)} {match.group(2)}}}",
        before_refs,
    )
    before_refs = normalize_extended_data_references(before_refs)
    before_refs = format_extended_data_figures(before_refs)
    before_refs = format_extended_data_tables(before_refs)
    return "\\section*{Online Methods and Extended Data}\n\n" + before_refs + "\n\n" + bibliography(refs) + "\n"


def write_project(main_body: str, supplement_body: str) -> None:
    content = OUT / "content"
    content.mkdir(parents=True, exist_ok=True)
    (content / "main_body.tex").write_text(main_body, encoding="utf-8")
    (content / "online_methods_extended_data.tex").write_text(supplement_body, encoding="utf-8")

    title = "Geophysics-informed groundwater reconstruction maps slow-recovery zones after extreme drought"
    author_block = r"""Zhengyang Fang$^{1,2}$, Zhuo Liu$^{1,3}$, Deshan Feng$^{1}$, Rongwen Guo$^{1,*}$, and Hang Chen$^{2,*}$\\
\small $^{1}$School of Geosciences and Info-physics, Central South University, Changsha 410083, China\\
\small $^{2}$School of Earth, Environment, and Sustainability, University of Iowa, Iowa City, IA 52245, USA\\
\small $^{3}$Doerr School of Sustainability, Department of Earth and Planetary Sciences, Stanford University, Stanford, CA 94305, USA\\
\small These authors contributed equally: Zhengyang Fang and Zhuo Liu.\\
\small $^{*}$Correspondence: rongwenguo@csu.edu.cn; hchen117@uiowa.edu"""
    main = MAIN_PREAMBLE + rf"""
\begin{{document}}
\begin{{center}}
{{\Large\bfseries {title}\par}}
\vspace{{0.8em}}
{author_block}
\end{{center}}
\vspace{{0.8em}}
\doublespacing
\linenumbers
\input{{content/main_body.tex}}
\end{{document}}
"""
    supplementary = MAIN_PREAMBLE + rf"""
\begin{{document}}
\begin{{center}}
{{\Large\bfseries Extended Data and Online Methods for\par}}
\vspace{{0.4em}}
{{\large\bfseries {title}\par}}
\vspace{{0.8em}}
{author_block}
\end{{center}}
\vspace{{0.8em}}
\doublespacing
\linenumbers
\input{{content/online_methods_extended_data.tex}}
\end{{document}}
"""
    (OUT / "main.tex").write_text(main, encoding="utf-8")
    (OUT / "extended_data.tex").write_text(supplementary, encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Nature LaTeX review package\n\n"
        "Generated from `manuscript-v3.docx` and `Supplementary Material.docx`. "
        "Compile `main.tex` for the main manuscript and `extended_data.tex` for "
        "the Online Methods and Extended Data review PDF. The `figures/` directory "
        "contains images extracted from the Word files. This source uses standard "
        "LaTeX classes and numerical citations preserved from the Word document.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "_conversion"
    raw.mkdir(exist_ok=True)
    main_raw = run_pandoc(MANUSCRIPT / "manuscript-v3.docx", raw / "main_raw.tex", OUT / "figures" / "main")
    supp_raw = run_pandoc(MANUSCRIPT / "Supplementary Material.docx", raw / "supp_raw.tex", OUT / "figures" / "extended_data")
    write_project(make_main_body(main_raw), make_supplement_body(supp_raw))
    print(f"Wrote Nature LaTeX review project to: {OUT}")


if __name__ == "__main__":
    main()
