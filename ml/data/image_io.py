"""Image (de)compression helpers for ingest pipelines.

Card photos arrive from PSA's CDN or eBay's image servers at multi-megabyte
resolutions (PSA: 3-4 MB, eBay s-l1600: ~700 KB-1.5 MB). The grader's
training input is 600x840, so storing originals wastes 10x of disk and
network bandwidth. We resize on ingest to a long-edge ceiling (default
1024 px) and re-encode JPEG q=88, which:

- shrinks ~4 MB PSA originals to ~250-400 KB
- still leaves >1.5x headroom over the 600x840 model input
- visually lossless at training scale

Original CDN URLs are kept in `front_url` / `back_url` in the DB so the
full-resolution version can always be re-fetched if needed.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, ImageOps

log = logging.getLogger(__name__)

IMAGE_MAX_LONG_EDGE = 1024
IMAGE_JPEG_QUALITY = 88


def compress_jpeg(
    raw: bytes,
    *,
    max_long_edge: int = IMAGE_MAX_LONG_EDGE,
    quality: int = IMAGE_JPEG_QUALITY,
) -> bytes:
    """Resize to a long-edge cap and re-encode as JPEG.

    Returns the encoded JPEG bytes. If the input cannot be decoded
    (corrupt download, unsupported format), the original bytes are
    returned unchanged so the caller can decide whether to keep them.
    """
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            elif im.mode == "P":
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")

            w, h = im.size
            longest = max(w, h)
            if longest > max_long_edge:
                scale = max_long_edge / longest
                im = im.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    Image.LANCZOS,
                )

            buf = io.BytesIO()
            im.save(
                buf,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        log.warning("compress_jpeg failed (%s); keeping original bytes", exc)
        return raw


def save_compressed_jpeg(
    raw: bytes,
    dest: Path,
    *,
    max_long_edge: int = IMAGE_MAX_LONG_EDGE,
    quality: int = IMAGE_JPEG_QUALITY,
) -> int:
    """Compress `raw` and write the JPEG to `dest`. Returns bytes written."""
    out = compress_jpeg(raw, max_long_edge=max_long_edge, quality=quality)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(out)
    return len(out)
