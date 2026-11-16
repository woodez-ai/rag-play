import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

from django import template
from django.utils.html import escape
from django.utils.safestring import mark_safe

register = template.Library()


@register.filter
def money(v, places: int = 2) -> str:
    """$1,234.56 / −$12.00 (refunds). Empty for None."""
    if v is None or v == "":
        return ""
    try:
        d = Decimal(str(v))
    except InvalidOperation:
        return str(v)
    s = f"${abs(d):,.{int(places)}f}"
    return f"−{s}" if d < 0 else s


@register.filter
def label(v) -> str:
    """Category key -> display label."""
    return "" if v is None else str(v).replace("_", " ").capitalize()


@register.filter
def month_name(v: str) -> str:
    try:
        return datetime.strptime(v, "%Y-%m").strftime("%B %Y")
    except (TypeError, ValueError):
        return v or ""


BOLD = re.compile(r"\*\*(.+?)\*\*")
BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")


@register.filter
def md(text: str) -> str:
    """The little Markdown the local model writes: **bold** and '- ' bullet lists. Everything is
    HTML-escaped first, so model output can never inject markup."""
    out, items = [], []

    def flush():
        if items:
            out.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            items.clear()

    for line in (text or "").splitlines():
        html = BOLD.sub(r"<strong>\1</strong>", escape(line.strip()))
        m = BULLET.match(line)
        if m:
            items.append(BOLD.sub(r"<strong>\1</strong>", escape(m.group(1).strip())))
        elif html:
            flush()
            out.append(f"<p>{html}</p>")
        else:
            flush()
    flush()
    return mark_safe("".join(out))


@register.filter
def cell(v) -> str:
    """Trace row cell (already JSON-plain)."""
    return "" if v is None else str(v)
