"""Build a Nature Portfolio project from the LaTeX conversion."""

from __future__ import annotations

import shutil
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
SOURCE = MANUSCRIPT / "nature_tex"
OUT = MANUSCRIPT / "nature_official_sn_template"

TITLE = (
    "Geophysics-informed groundwater reconstruction maps slow-recovery zones "
    "after extreme drought"
)

PREAMBLE = r"""% Springer Nature LaTeX Template, Nature Portfolio option.
% This manuscript file is generated from the reviewed Word-to-LaTeX conversion.
% Keep sn-jnl.cls in this directory when compiling locally or on Overleaf.
\PassOptionsToPackage{hyphens}{url}
\documentclass[referee,lineno,pdflatex,sn-nature]{sn-jnl}

\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{booktabs}
\usepackage{longtable,array,calc}
\usepackage{float}
\usepackage{xurl}

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

AUTHORS_AND_AFFILIATIONS = r"""\title[Geophysics-informed groundwater reconstruction]{%s}

\author[1,2]{\fnm{Zhengyang} \sur{Fang}}
\author[1,3]{\fnm{Zhuo} \sur{Liu}}
\author[1]{\fnm{Deshan} \sur{Feng}}
\author*[1]{\fnm{Rongwen} \sur{Guo}}\email{rongwenguo@csu.edu.cn}
\author*[2]{\fnm{Hang} \sur{Chen}}\email{hchen117@uiowa.edu}

\equalcont{These authors contributed equally: Zhengyang Fang and Zhuo Liu.}

\affil[1]{\orgdiv{School of Geosciences and Info-physics}, \orgname{Central South University}, \orgaddress{\city{Changsha}, \postcode{410083}, \country{China}}}
\affil[2]{\orgdiv{School of Earth, Environment, and Sustainability}, \orgname{University of Iowa}, \orgaddress{\city{Iowa City}, \state{IA}, \postcode{52245}, \country{USA}}}
\affil[3]{\orgdiv{Doerr School of Sustainability, Department of Earth and Planetary Sciences}, \orgname{Stanford University}, \orgaddress{\city{Stanford}, \state{CA}, \postcode{94305}, \country{USA}}}

""" % TITLE


def body_text(filename: str) -> str:
    text = (SOURCE / "content" / filename).read_text(encoding="utf-8").strip()
    # Scale figures to the available line width.
    figure_pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}")
    return figure_pattern.sub(lambda match: r"\pandocbounded{" + match.group(0) + "}", text)


def document(title_prefix: str, body: str) -> str:
    front_matter = AUTHORS_AND_AFFILIATIONS.replace(
        r"\title[Geophysics-informed groundwater reconstruction]{",
        rf"\title[{title_prefix}]{{",
        1,
    )
    return (
        PREAMBLE
        + front_matter
        + "\\begin{document}\n\n"
        + "\\maketitle\n\n"
        + body
        + "\n\n\\end{document}\n"
    )


def write_readme() -> None:
    (OUT / "README.md").write_text(
        "# Official Springer Nature / Nature Portfolio LaTeX project\n\n"
        "This project uses the official Springer Nature `sn-jnl` document class "
        "with the Nature Portfolio option:\n\n"
        "```tex\n"
        "\\documentclass[referee,lineno,pdflatex,sn-nature]{sn-jnl}\n"
        "```\n\n"
        "Compile `main.tex` for the main manuscript and `extended_data.tex` for "
        "the Online Methods and Extended Data document. Both source files are "
        "self-contained; figures remain in `figures/` as separate upload assets.\n\n"
        "The manuscript content, author list, affiliations, figures and manually "
        "checked numbered reference lists were retained from the reviewed project. "
        "The official class controls the document layout, front matter, reference "
        "style selection and review line numbering.\n\n"
        "Template source: https://www.overleaf.com/latex/templates/"
        "springer-nature-latex-template/gsvvftmrppwq\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE / "figures", OUT / "figures", dirs_exist_ok=True)

    main_body = body_text("main_body.tex")
    extended_body = body_text("online_methods_extended_data.tex")

    (OUT / "main.tex").write_text(
        document("Geophysics-informed groundwater reconstruction", main_body),
        encoding="utf-8",
    )
    (OUT / "extended_data.tex").write_text(
        document("Online Methods and Extended Data", extended_body),
        encoding="utf-8",
    )
    write_readme()
    print(f"Wrote official Nature Portfolio project to: {OUT}")


if __name__ == "__main__":
    main()
