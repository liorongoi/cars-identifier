"""Safe image decoding and encoding."""

from __future__ import annotations

import hashlib
import io

import imagehash
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

RGBImage = NDArray[np.uint8]

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "MPO", "HEIF", "GIF"}

try:  # optional HEIC support
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - optional dependency
    pass


class ImageDecodeError(ValueError):
    """The bytes are not an acceptable image."""


def decode_image(data: bytes, max_pixels: int = 60_000_000) -> RGBImage:
    """Decode bytes to an RGB array, applying EXIF orientation.

    Rejects unknown formats and decompression bombs before decoding pixels.
    """
    if not data:
        raise ImageDecodeError("empty file")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format or ""
            w, h = probe.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError("not a recognised image file") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ImageDecodeError(f"unsupported image format: {fmt or 'unknown'}")
    if w * h > max_pixels:
        raise ImageDecodeError(f"image too large: {w}x{h} pixels")
    if w < 16 or h < 16:
        raise ImageDecodeError(f"image too small: {w}x{h} pixels")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.seek(0)
            oriented = ImageOps.exif_transpose(im)
            rgb = oriented.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8).copy()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"cannot decode image: {exc}") from exc


def encode_jpeg(img: RGBImage, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def perceptual_hash(img: RGBImage) -> str:
    return str(imagehash.phash(Image.fromarray(img)))


def phash_distance(a: str, b: str) -> int:
    return int(imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b))


def resize_max_side(img: RGBImage, max_side: int) -> RGBImage:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    import cv2

    return np.asarray(
        cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA),
        dtype=np.uint8,
    )


def crop(img: RGBImage, bbox: tuple[int, int, int, int]) -> RGBImage:
    x0, y0, x1, y1 = bbox
    h, w = img.shape[:2]
    x0, x1 = max(0, min(x0, w - 1)), max(1, min(x1, w))
    y0, y1 = max(0, min(y0, h - 1)), max(1, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty crop {bbox} for image {w}x{h}")
    return img[y0:y1, x0:x1]
