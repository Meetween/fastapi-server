"""utils.picture: header checks before decoding, orientation, crop, metadata, size cap."""
import io
import struct
import warnings
import zlib

import pytest
from PIL import Image

from utils import picture
from utils.picture import PictureError, normalise_picture_sync


def _encode(im: Image.Image, fmt: str, **kwargs) -> bytes:
    out = io.BytesIO()
    im.save(out, format=fmt, **kwargs)
    return out.getvalue()


def _png_header_only(width: int, height: int) -> bytes:
    """A PNG that declares width x height but holds one tiny IDAT. Decoding it would fail late;
    the header check must refuse it first."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )


def _result_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def test_non_square_jpeg_becomes_512_square_jpeg() -> None:
    # Left half red, right half blue: the centre crop of a 1280x720 frame keeps both halves.
    src = Image.new("RGB", (1280, 720), (255, 0, 0))
    src.paste((0, 0, 255), (640, 0, 1280, 720))
    result = normalise_picture_sync(_encode(src, "JPEG", quality=95))
    out = _result_image(result.data)
    assert out.format == "JPEG"
    assert out.size == (512, 512)
    assert (result.width, result.height) == (512, 512)
    assert len(result.data) <= 400 * 1024
    left, right = out.getpixel((60, 256)), out.getpixel((450, 256))
    assert left[0] > 200 and left[2] < 60
    assert right[2] > 200 and right[0] < 60


def test_centre_crop_drops_the_sides_of_a_wide_picture() -> None:
    src = Image.new("RGB", (1280, 720), (0, 255, 0))
    src.paste((255, 0, 0), (0, 0, 280, 720))  # outside the central 720x720 square
    out = _result_image(normalise_picture_sync(_encode(src, "PNG")).data)
    assert all(out.getpixel((x, 256))[1] > 200 for x in (5, 256, 506))


def test_sha256_matches_the_output() -> None:
    import hashlib

    result = normalise_picture_sync(_encode(Image.new("RGB", (300, 300), "white"), "PNG"))
    assert result.sha256 == hashlib.sha256(result.data).hexdigest()


def test_exif_orientation_is_applied_then_stripped() -> None:
    # 600x300 stored sideways: orientation 6 means "rotate 90 clockwise to display". The red
    # top half of the displayed picture is the left half of the stored pixels.
    src = Image.new("RGB", (600, 300), (0, 0, 255))
    src.paste((255, 0, 0), (0, 0, 300, 300))
    exif = Image.Exif()
    exif[0x0112] = 6
    exif[0x010F] = "Camera maker"
    data = _encode(src, "JPEG", quality=95, exif=exif)
    assert Image.open(io.BytesIO(data)).getexif()[0x0112] == 6

    out = _result_image(normalise_picture_sync(data).data)
    assert out.size == (512, 512)
    assert len(out.getexif()) == 0
    assert "exif" not in out.info
    assert "icc_profile" not in out.info
    # After the rotation the stored left half is on top.
    assert out.getpixel((256, 20))[0] > 200
    assert out.getpixel((256, 490))[2] > 200


def test_icc_profile_and_comment_are_dropped() -> None:
    src = Image.new("RGB", (400, 400), "gray")
    data = _encode(src, "JPEG", icc_profile=b"x" * 128, comment=b"secret note")
    out = _result_image(normalise_picture_sync(data).data)
    assert not out.info.get("icc_profile")
    assert not out.info.get("comment")


def test_png_with_transparency_is_flattened_on_white() -> None:
    src = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    out = _result_image(normalise_picture_sync(_encode(src, "PNG")).data)
    assert out.mode == "RGB"
    assert min(out.getpixel((256, 256))) > 240


def test_palette_png_and_grayscale_are_accepted() -> None:
    for mode in ("P", "L", "1"):
        src = Image.new("RGB", (300, 300), "white").convert(mode)
        assert _result_image(normalise_picture_sync(_encode(src, "PNG")).data).size == (512, 512)


def test_small_picture_is_upscaled() -> None:
    out = _result_image(normalise_picture_sync(_encode(Image.new("RGB", (256, 256), "teal"), "PNG")).data)
    assert out.size == (512, 512)


def test_animated_png_uses_frame_zero() -> None:
    first = Image.new("RGB", (300, 300), (255, 0, 0))
    second = Image.new("RGB", (300, 300), (0, 0, 255))
    data = _encode(first, "PNG", save_all=True, append_images=[second], duration=100)
    out = _result_image(normalise_picture_sync(data).data)
    assert out.getpixel((256, 256))[0] > 200


def test_noisy_picture_is_recompressed_under_the_cap() -> None:
    noise = Image.effect_noise((2000, 2000), 128).convert("RGB")
    result = normalise_picture_sync(_encode(noise, "PNG"))
    assert len(result.data) <= 400 * 1024
    assert _result_image(result.data).size == (512, 512)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"GIF89a" + b"\x00" * 64,
        b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 64,
        b"<svg xmlns='http://www.w3.org/2000/svg'/>",
        b"%PDF-1.7\n",
        b"BM" + b"\x00" * 64,
    ],
    ids=["empty", "gif", "webp", "svg", "pdf", "bmp"],
)
def test_other_formats_are_unsupported_by_magic_bytes(data: bytes) -> None:
    with pytest.raises(PictureError) as err:
        normalise_picture_sync(data)
    assert err.value.code == "unsupported_type"


def test_content_is_judged_not_the_name_or_type() -> None:
    # A GIF is refused even though a client may call it image/png; the function never sees that.
    gif = _encode(Image.new("RGB", (300, 300)), "GIF")
    with pytest.raises(PictureError) as err:
        normalise_picture_sync(gif)
    assert err.value.code == "unsupported_type"


@pytest.mark.parametrize("size", [(255, 600), (600, 255), (100, 100), (1, 1)])
def test_too_small_is_invalid(size) -> None:
    with pytest.raises(PictureError) as err:
        normalise_picture_sync(_encode(Image.new("RGB", size), "PNG"))
    assert err.value.code == "picture_invalid"


@pytest.mark.parametrize("size", [(4097, 300), (300, 4097), (30000, 30000), (65535, 65535)])
def test_declared_size_over_limit_is_refused_from_the_header(size, monkeypatch) -> None:
    """A tiny file that declares a huge frame is refused without any decoding."""
    decoded = []
    real_load = Image.Image.load
    monkeypatch.setattr(Image.Image, "load", lambda self, *a, **k: decoded.append(1) or real_load(self, *a, **k))
    with pytest.raises(PictureError) as err:
        normalise_picture_sync(_png_header_only(*size))
    assert err.value.code == "picture_invalid"
    assert decoded == []


def test_jpeg_header_declaring_a_huge_frame_is_refused() -> None:
    data = bytearray(_encode(Image.new("RGB", (300, 300), "white"), "JPEG"))
    sof = data.index(b"\xff\xc0")
    data[sof + 5 : sof + 9] = struct.pack(">HH", 60000, 60000)  # height, width
    with pytest.raises(PictureError) as err:
        normalise_picture_sync(bytes(data))
    assert err.value.code == "picture_invalid"


def test_pixel_limit_is_set_and_the_bomb_warning_is_an_error() -> None:
    assert Image.MAX_IMAGE_PIXELS == 4096 * 4096
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        picture.warnings.simplefilter("error", Image.DecompressionBombWarning)
        with pytest.raises(Image.DecompressionBombWarning):
            warnings.warn("bomb", Image.DecompressionBombWarning)


def test_max_side_exactly_is_accepted() -> None:
    # 4096x4096 equals MAX_IMAGE_PIXELS, which Pillow accepts without a warning.
    src = Image.new("L", (4096, 4096), 128)
    assert _result_image(normalise_picture_sync(_encode(src, "PNG", compress_level=1)).data).size == (512, 512)


def test_truncated_png_is_invalid() -> None:
    data = _encode(Image.effect_noise((600, 600), 100).convert("RGB"), "PNG")
    with pytest.raises(PictureError) as err:
        normalise_picture_sync(data[: len(data) // 2])
    assert err.value.code == "picture_invalid"


def test_corrupt_body_after_a_valid_header_is_invalid() -> None:
    data = bytearray(_encode(Image.effect_noise((600, 600), 100).convert("RGB"), "PNG"))
    for i in range(200, 400):
        data[i] = 0xAB
    with pytest.raises(PictureError) as err:
        normalise_picture_sync(bytes(data))
    assert err.value.code == "picture_invalid"


def test_truncated_jpeg_is_invalid() -> None:
    data = _encode(Image.effect_noise((600, 600), 100).convert("RGB"), "JPEG")
    with pytest.raises(PictureError) as err:
        normalise_picture_sync(data[: len(data) // 2])
    assert err.value.code == "picture_invalid"


def test_only_png_and_jpeg_decoders_are_tried(monkeypatch) -> None:
    seen = {}
    real_open = Image.open

    def spy(fp, mode="r", formats=None):
        seen["formats"] = formats
        return real_open(fp, mode, formats)

    monkeypatch.setattr(picture.Image, "open", spy)
    normalise_picture_sync(_encode(Image.new("RGB", (300, 300)), "PNG"))
    assert list(seen["formats"]) == ["PNG", "JPEG"]


async def test_async_wrapper_runs_in_a_thread() -> None:
    result = await picture.normalise_picture(_encode(Image.new("RGB", (300, 300), "white"), "PNG"))
    assert _result_image(result.data).size == (512, 512)
