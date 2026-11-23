"""PDF (pdfplumber) or page image (on-device OCR) -> words with positions, grouped into lines.

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


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}
PAGE_WIDTH = 612.0  # OCR coordinates are scaled to PDF points so parsers see one scale


def read_image(path: str | Path) -> list[Word]:
    """OCR one page photo/scan with Apple's Vision framework (on-device, macOS only).

    Each recognised line is split into words with their own boxes; coordinates are top-left
    origin, scaled so the page is PAGE_WIDTH wide (height keeps the image's aspect ratio).
    """
    try:
        import Quartz
        import Vision
        from Foundation import NSURL
    except ImportError as e:  # pragma: no cover - platform dependent
        raise ValueError("image statements need the 'ocr' extra on macOS: "
                         "uv sync --extra ocr") from e
    url = NSURL.fileURLWithPath_(str(Path(path).resolve()))
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None) if src else None
    if img is None:
        raise ValueError(f"cannot read image {Path(path).name}")
    w_px, h_px = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)  # don't "fix" merchant codes and amounts
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise ValueError(f"OCR failed for {Path(path).name}: {err}")
    scale_w, scale_h = PAGE_WIDTH, PAGE_WIDTH * h_px / w_px
    obs_list = [(o, o.topCandidates_(1)[0]) for o in (req.results() or [])]

    # A photographed page is slightly rotated and curled, so one printed row drifts up or down
    # by more than the gap between rows over the width of the page. Each long OCR line tells us
    # the local text angle; every word is levelled with the median slope of long lines near it.
    def slope(o) -> float:  # dy/dx in page units, y pointing down
        bl, br = o.bottomLeft(), o.bottomRight()
        return -((br.y - bl.y) * scale_h) / max((br.x - bl.x) * scale_w, 1e-6)

    long = [((1 - o.boundingBox().origin.y) * scale_h, slope(o)) for o, _ in obs_list
            if o.boundingBox().size.width > 0.15]
    page_slope = sorted(s for _, s in long)[len(long) // 2] if long else 0.0

    def local_slope(y: float) -> float:
        near = sorted(s for yy, s in long if abs(yy - y) < 80)
        return near[len(near) // 2] if near else page_slope

    words: list[Word] = []
    for obs, cand in obs_list:
        text = str(cand.string())
        pos = 0
        for token in text.split():
            start = text.index(token, pos)
            pos = start + len(token)
            box, _ = cand.boundingBoxForRange_error_((start, len(token)), None)
            bb = box.boundingBox() if box is not None else obs.boundingBox()
            (x, y), (bw, bh) = bb.origin, bb.size  # normalised, bottom-left origin
            x0, x1, top = x * scale_w, (x + bw) * scale_w, (1 - y - bh) * scale_h
            top -= local_slope(top) * ((x0 + x1) / 2 - PAGE_WIDTH / 2)
            words.append(Word(x0, x1, top, token))
    return sorted(words, key=lambda w: (w.top, w.x0))


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
