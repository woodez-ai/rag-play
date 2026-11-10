"""PDF -> words with positions (pdfplumber), grouped into lines.

Parsers work on positioned words rather than plain text so they can split side-by-side
columns and recognise rows by where each field starts.
"""

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass(frozen=True)
class Word:
    x0: float
    x1: float
    top: float
    text: str


@dataclass
class Line:
    page: int  # 1-based
    top: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def x0(self) -> float:
        return self.words[0].x0


def read_pages(path: str | Path) -> list[list[Word]]:
    """Words per page, in reading order."""
    with pdfplumber.open(path) as pdf:
        return [
            [Word(w["x0"], w["x1"], w["top"], w["text"]) for w in page.extract_words()]
            for page in pdf.pages
        ]


def group_lines(words: list[Word], page: int, y_tol: float = 3.0) -> list[Line]:
    """Cluster words whose tops are within y_tol into lines, sorted left to right."""
    lines: list[Line] = []
    for w in sorted(words, key=lambda w: (w.top, w.x0)):
        if lines and w.top - lines[-1].top <= y_tol:
            lines[-1].words.append(w)
        else:
            lines.append(Line(page, w.top, [w]))
    for line in lines:
        line.words.sort(key=lambda w: w.x0)
    return lines
