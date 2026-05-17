"""
Per-section word count for the three thesis main.tex files.

Strips LaTeX commands, math, comments, and braces, then counts whitespace-
delimited tokens. Sums per \section* block so we can see where the
word budget goes.

Usage:
    python code/count_words_by_section.py
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def strip_latex(text: str) -> str:
    """Crude but consistent strip-for-counting."""
    # Comments (% to end of line, but not \%)
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
    # Display math
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    # Inline math
    text = re.sub(r"\$[^$]*\$", "", text)
    # Environments we should not count (tables, figures, equations,
    # tabular, longtable, thebibliography)
    SKIP_ENVS = [
        "table", "table*", "figure", "figure*", "equation", "equation*",
        "align", "align*", "tabular", "longtable", "thebibliography",
        "tikzpicture", "lstlisting", "verbatim",
    ]
    for env in SKIP_ENVS:
        text = re.sub(
            rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}",
            "",
            text,
            flags=re.DOTALL,
        )
    # Commands with one mandatory arg (keep arg, drop command): keep words
    text = re.sub(r"\\(?:textbf|textit|emph|underline|texttt|textsuperscript|textsubscript)\{([^{}]*)\}", r"\1", text)
    # All other commands with optional+mandatory args: drop entirely
    text = re.sub(r"\\[a-zA-Z]+\*?\s*(\[[^\]]*\])?\s*(\{[^{}]*\})*", "", text)
    # Stray backslashes
    text = re.sub(r"\\\\", " ", text)
    text = re.sub(r"\\.", "", text)
    # Braces
    text = re.sub(r"[{}]", "", text)
    return text


def count_words(text: str) -> int:
    return len(strip_latex(text).split())


def sectionise(content: str):
    r"""Split on \section{...} and \section*{...} and return [(name, body), ...]."""
    parts = re.split(r"\\section\*?\{([^}]+)\}", content)
    out = [("(preamble / title)", parts[0])]
    for i in range(1, len(parts), 2):
        name = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        out.append((name, body))
    return out


def main() -> None:
    for p in (1, 2, 3):
        path = REPO_ROOT / "papers" / f"paper{p}" / "main.tex"
        content = path.read_text()
        sections = sectionise(content)
        print(f"\n=== Paper {p}: {path.relative_to(REPO_ROOT)} ===")
        total = 0
        for name, body in sections:
            n = count_words(body)
            total += n
            print(f"  {n:>5}  {name}")
        print(f"  -----")
        print(f"  {total:>5}  TOTAL")


if __name__ == "__main__":
    main()
