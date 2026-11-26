"""Normalise an uploaded picture to a 512x512 JPEG.

Shared by `PUT /api/me/picture` and the meeting picture routes (B6 calls `normalise_picture`).

Order of checks, all before the pixels are decoded:

1. Magic bytes: PNG or JPEG only. The client's Content-Type and filename are ignored.
2. `Image.open(..., formats=["PNG", "JPEG"])` reads the header only. The declared size must be
   256 to 4096 pixels per side. `Image.MAX_IMAGE_PIXELS` is 4096 * 4096, and a
   `DecompressionBombWarning` is raised as an error, so a header that declares more never decodes.
3. Only then `load()`, frame 0 only, EXIF orientation applied, then a centre crop to a square and a
   resize to 512x512. The JPEG is written without EXIF, ICC profile or any other metadata and is
   at most `MAX_OUTPUT_BYTES`.

`normalise_picture_sync` is blocking and CPU bound. `normalise_picture` runs it in a thread.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import warnings
from dataclasses import dataclass
from typing import Literal

from PIL import Image, ImageOps

SIDE = 512
MIN_INPUT_SIDE = 256
MAX_INPUT_SIDE = 4096
MAX_OUTPUT_BYTES = 400 * 1024
JPEG_QUALITY = 90
# Lower steps are tried only when a noisy picture does not fit in MAX_OUTPUT_BYTES at the first.
_FALLBACK_QUALITIES = (85, 80, 75, 70, 60, 50)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"

# Process wide, as Pillow keeps them. Pillow only warns between the limit and twice the limit
# and raises above that; the filter turns the warning into an error too.
Image.MAX_IMAGE_PIXELS = MAX_INPUT_SIDE * MAX_INPUT_SIDE
warnings.simplefilter("error", Image.DecompressionBombWarning)


class PictureError(Exception):
    """The upload cannot become an avatar. Routes map `code` to the HTTP status."""

    def __init__(self, code: Literal["unsupported_type", "picture_invalid"], message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class NormalisedPicture:
    data: bytes
    sha256: str
    width: int = SIDE
    height: int = SIDE


def sniff(data: bytes) -> Literal["PNG", "JPEG"]:
    """The format by magic bytes. 415 `unsupported_type` for anything else."""
    if data.startswith(_PNG_MAGIC):
        return "PNG"
    if data.startswith(_JPEG_MAGIC):
        return "JPEG"
    raise PictureError("unsupported_type", "The picture must be a PNG or a JPEG file")


def _invalid(message: str) -> PictureError:
    return PictureError("picture_invalid", message)


def _to_rgb(im: Image.Image) -> Image.Image:
    """RGB with any transparency flattened on white, so a PNG cut-out does not turn black."""
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return im.convert("RGB")


def _encode(im: Image.Image) -> bytes:
    # The JPEG writer falls back to im.info for the comment and the ICC profile, and resizing
    # keeps im.info, so it is emptied here and not only left out of the save arguments.
    im.info.clear()
    for quality in (JPEG_QUALITY, *_FALLBACK_QUALITIES):
        out = io.BytesIO()
        # No exif, icc_profile or comment argument: the output carries no metadata.
        im.save(out, format="JPEG", quality=quality, optimize=True)
        if out.tell() <= MAX_OUTPUT_BYTES:
            return out.getvalue()
    raise _invalid("The picture cannot be compressed under the size limit")


def normalise_picture_sync(data: bytes) -> NormalisedPicture:
    sniff(data)
    try:
        with Image.open(io.BytesIO(data), formats=["PNG", "JPEG"]) as opened:
            width, height = opened.size  # header only, nothing is decoded yet
            if not (MIN_INPUT_SIDE <= width <= MAX_INPUT_SIDE and MIN_INPUT_SIDE <= height <= MAX_INPUT_SIDE):
                raise _invalid(
                    f"The picture must be {MIN_INPUT_SIDE} to {MAX_INPUT_SIDE} pixels per side, got {width}x{height}"
                )
            if width * height > Image.MAX_IMAGE_PIXELS:
                raise _invalid("The picture has too many pixels")
            opened.load()  # frame 0 only, no seek
            upright = ImageOps.exif_transpose(opened)
            flat = _to_rgb(upright)
            square = ImageOps.fit(flat, (SIDE, SIDE), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            encoded = _encode(square)
    except PictureError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise _invalid("The picture has too many pixels") from exc
    except (
        OSError,  # includes UnidentifiedImageError and truncated data
        SyntaxError,
        ValueError,
        EOFError,
    ) as exc:
        raise _invalid("The picture cannot be decoded") from exc
    return NormalisedPicture(data=encoded, sha256=hashlib.sha256(encoded).hexdigest())


async def normalise_picture(data: bytes) -> NormalisedPicture:
    """Same as `normalise_picture_sync`, off the event loop."""
    return await asyncio.to_thread(normalise_picture_sync, data)
