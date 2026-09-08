"""Reading what a picture or a recording is, without decoding it.

The controller installs four pure-Python packages so it can run on a NAS, and
that is not going to change for the sake of knowing that a JPEG is 640 by 480.
Every common image and audio format writes its dimensions or its duration in
the first few dozen bytes, in a layout that has not changed in thirty years,
and reading them is a page of code with no dependency.

What this deliberately does not do is decode anything. A file that is
malformed past its header will still be reported with the size its header
claims -- and the trainer, which does decode it, will say so then. What it
*can* say cheaply is the thing worth saying before training: that the file's
first bytes are not what its name claims, which is how a directory of "PNGs"
that are HTML error pages from a broken download gets caught here rather than
three hours into a run.
"""
from __future__ import annotations

import struct

# What the first bytes of each format look like. The mime a file was stored
# under is what its name said; this is what its bytes say.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"fLaC", "audio/flac"),
    (b"OggS", "audio/ogg"),
    (b"ID3", "audio/mpeg"),
    (b"%PDF", "application/pdf"),
)

# Formats that share a container: RIFF holds WAV and WebP, and the box atoms
# of MP4 and M4A start with a size rather than a signature.
_RIFF = {b"WAVE": "audio/wav", b"WEBP": "image/webp"}

# How much of a file these readers ever need. Reading more is wasted disk.
HEAD_BYTES = 64 * 1024


def sniff(head: bytes) -> str:
    """The mime the bytes say they are, or "" when they say nothing."""
    for magic, mime in _MAGIC:
        if head.startswith(magic):
            return mime
    if head[:4] == b"RIFF" and len(head) >= 12:
        return _RIFF.get(head[8:12], "")
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        return "audio/mp4" if brand in (b"M4A ", b"M4B ") else "video/mp4"
    if head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"          # a bare MP3 frame, no ID3 tag
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    # Text wearing a picture's name. The usual way this happens is a download
    # script that saved the server's error page under the filename it asked
    # for, a thousand times, and nobody opened one.
    probe = head[:512]
    if probe and b"\x00" not in probe:
        try:
            text = probe.decode("utf-8").lstrip().lower()
        except UnicodeDecodeError:
            return ""
        if text.startswith(("<!doctype", "<html", "<?xml", "{", "[")):
            return "text/html" if text.startswith("<") else "application/json"
    return ""


def same_family(claimed: str, sniffed: str) -> bool:
    """Whether the name and the bytes agree closely enough to trust.

    Exact for images. Audio is looser: MP4 and M4A are the same container,
    OGG and Opus the same one, and a name is allowed to be vague about which.
    """
    if not sniffed:
        return True                  # nothing to check against
    if claimed == sniffed:
        return True
    a, b = claimed.split("/")[0], sniffed.split("/")[0]
    return a == b and a in ("audio", "video")


def image_size(head: bytes) -> tuple[int, int] | None:
    """Width and height off the header. None when the format is not read."""
    try:
        if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
        if head.startswith((b"GIF87a", b"GIF89a")) and len(head) >= 10:
            w, h = struct.unpack("<HH", head[6:10])
            return int(w), int(h)
        if head.startswith(b"BM") and len(head) >= 26:
            w, h = struct.unpack("<ii", head[18:26])
            return abs(int(w)), abs(int(h))
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return _webp_size(head)
        if head.startswith(b"\xff\xd8"):
            return _jpeg_size(head)
    except (struct.error, IndexError, ValueError):
        return None
    return None


def _jpeg_size(head: bytes) -> tuple[int, int] | None:
    """Walk the JPEG segments to the first frame header.

    A JPEG's size is not at a fixed offset: it sits in the SOF marker, which
    comes after however many metadata segments the camera felt like writing.
    Progressive files use a different SOF marker from baseline; both carry the
    dimensions in the same place.
    """
    i = 2
    n = len(head)
    while i + 9 < n:
        if head[i] != 0xFF:
            i += 1
            continue
        marker = head[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = struct.unpack(">H", head[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", head[i + 5:i + 9])
            return int(w), int(h)
        i += 2 + length
    return None


def _webp_size(head: bytes) -> tuple[int, int] | None:
    chunk = head[12:16]
    if chunk == b"VP8 " and len(head) >= 30:
        w, h = struct.unpack("<HH", head[26:30])
        return int(w & 0x3FFF), int(h & 0x3FFF)
    if chunk == b"VP8L" and len(head) >= 25:
        b0, b1, b2, b3 = head[21:25]
        w = ((b1 & 0x3F) << 8 | b0) + 1
        h = ((b3 & 0x0F) << 10 | b2 << 2 | (b1 & 0xC0) >> 6) + 1
        return int(w), int(h)
    if chunk == b"VP8X" and len(head) >= 30:
        w = int.from_bytes(head[24:27], "little") + 1
        h = int.from_bytes(head[27:30], "little") + 1
        return w, h
    return None


def wav_duration(head: bytes, size: int) -> float | None:
    """Seconds of audio in a WAV, from the format chunk and the file size.

    Only WAV: it is the one format whose duration is arithmetic on the header
    rather than a walk over every frame, and it is the format speech datasets
    actually arrive in. An MP3's length needs the whole file read, and that is
    the trainer's job.
    """
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    i = 12
    while i + 8 <= len(head):
        ident = head[i:i + 4]
        length = struct.unpack("<I", head[i + 4:i + 8])[0]
        if ident == b"fmt " and i + 24 <= len(head):
            _fmt, channels, rate, byte_rate, _align, bits = struct.unpack(
                "<HHIIHH", head[i + 8:i + 24])
            if byte_rate <= 0:
                byte_rate = rate * channels * max(bits, 8) // 8
            if byte_rate <= 0:
                return None
            # The data chunk is the file minus its headers, near enough. Off
            # by the size of the metadata chunks, which is milliseconds.
            return round(max(size - 44, 0) / byte_rate, 3)
        i += 8 + length + (length & 1)
    return None


def describe(head: bytes, size: int, claimed: str) -> dict:
    """Everything the header will say about a file, in one call."""
    sniffed = sniff(head)
    out: dict = {"sniffed": sniffed,
                 "mismatch": not same_family(claimed, sniffed)}
    if (claimed or sniffed).startswith("image/"):
        if dims := image_size(head):
            out["width"], out["height"] = dims
    elif claimed in ("audio/wav", "audio/x-wav") or sniffed == "audio/wav":
        if (secs := wav_duration(head, size)) is not None:
            out["seconds"] = secs
    return out
