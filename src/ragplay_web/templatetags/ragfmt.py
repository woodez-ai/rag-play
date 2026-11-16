from datetime import datetime
from decimal import Decimal, InvalidOperation

from django import template

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


@register.filter
def cell(v) -> str:
    """Trace row cell (already JSON-plain)."""
    return "" if v is None else str(v)
