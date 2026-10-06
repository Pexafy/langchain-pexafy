"""What the model reads of a photo.

The API returns some thirty fields per photo, most of them for a page that renders
the image (blur hash, five sizes, the long AI description). An agent needs far less:
what the photo shows, where to fetch it, and the credit line it has to print next to
it. Everything else stays in the tool's artifact, untouched.

Two fields come from third parties — the photo's alt text and the photographer's
name — and reach the model as text it will read, so they are flattened to one line,
stripped of control and invisible characters, and capped.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from pexafy import Photo

FREE_TEXT_MAX_LENGTH = 300

# Words a source writes where it has no name: "Photo by Unknown on Pixabay", and a
# missing surname stringified, "nympha57 None".
_NO_NAME = {"unknown", "none", "null", "undefined", "n/a", "nan"}
_CREDIT_LINE = re.compile(r"^(Photo) by (.*?)( on .*)$", re.S)


def clean_text(value: Any, limit: int = FREE_TEXT_MAX_LENGTH) -> str:
    """One line, no control or format character, at most `limit` characters."""
    if not isinstance(value, str):
        return ""
    kept = [
        " " if ch.isspace() else ch
        for ch in value
        if ch.isspace() or unicodedata.category(ch) not in {"Cc", "Cf", "Cs"}
    ]
    text = re.sub(r" {2,}", " ", "".join(kept)).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def clean_name(name: Any) -> str:
    """A photographer's name without placeholder words; "" when nothing real is left."""
    words = clean_text(name).split()
    kept = [w for w in words if w.lower() not in _NO_NAME]
    if len(kept) < len(words) and [w.lower() for w in kept] == ["photographer"]:
        kept = []
    cleaned = " ".join(kept)
    return "" if cleaned.isdigit() else cleaned


def clean_credit(line: str) -> str:
    """'Photo by nympha57 None on Pexels' -> 'Photo by nympha57 on Pexels'."""
    line = clean_text(line, limit=400)
    match = _CREDIT_LINE.match(line)
    if not match:
        return line
    lead, who, rest = match.groups()
    who = clean_name(who)
    return f"{lead} by {who}{rest}" if who else f"{lead}{rest}"


def summarize(photo: Photo, rank: int) -> dict[str, Any]:
    """The compact view of one photo that goes into the tool message."""
    return {
        "rank": rank,
        "photo_id": photo.photo_id,
        "alt_text": clean_text(photo.alt_text),
        "url": photo.urls.regular or photo.image_url,
        "thumbnail_url": photo.urls.small or photo.urls.thumb,
        "width": photo.width,
        "height": photo.height,
        "orientation": photo.orientation,
        "dominant_color": photo.color_hex,
        "photographer": clean_name(photo.photographer_full_name)
        or clean_name(photo.photographer_username),
        "source": photo.source,
        "source_page_url": photo.source_image_url,
        "license": photo.license_type,
        "credit": clean_credit(photo.attribution.plain),
    }
