"""Read BED files, skipping whole comment lines only.

A line is skipped when it is blank or starts with "#", "track" or
"browser" (the UCSC header lines).  A "#" anywhere else is data:
ChIP-Atlas peak BEDs keep antibody catalog numbers such as
"Catalog#: MS-699-P" in column 4, and pandas.read_csv(comment="#") cuts
such a line at the "#", so column 5 (the score) and the columns after it
are lost.  Files are read as UTF-8 (a byte order mark is ignored, bytes
that are not UTF-8 become U+FFFD).

The compression is found from the first bytes of the file, not from its
name: gzip, bzip2, xz, zip (an archive that holds exactly one file) and
zstd files are decompressed.  zstd needs Python 3.14 or later
(compression.zstd) or the zstandard package.  A compressed file that
cannot be read (corrupt or truncated data, a zip archive with no file or
more than one, zstd without a decoder) raises BedReadError, an OSError.

pandas is imported by read_bed, not here: the MCP server imports this
module at startup and loads pandas itself before its stdio loop.
"""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import re
import zipfile
import zlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from os import PathLike
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

COMMENT_PREFIXES = ("#", "track", "browser")

# What reading compressed data raises: truncated data gives EOFError, corrupt
# data zlib.error (gzip, zip), lzma.LZMAError (xz), OSError (bzip2, and
# gzip.BadGzipFile), zipfile.BadZipFile, or RuntimeError (an encrypted zip
# member; NotImplementedError, a subclass, for an unsupported zip method).
_DECOMPRESSION_ERRORS = (OSError, EOFError, RuntimeError, zlib.error,
                         lzma.LZMAError, zipfile.BadZipFile)
_CHUNK = 1 << 16


class BedReadError(OSError):
    """A compressed BED file that cannot be read.

    reason says why, without the path: corrupt or truncated data, a zip
    archive that does not hold exactly one file, or zstd data with no
    zstd decoder installed.
    """

    def __init__(self, path: str | PathLike, reason: str):
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


def is_data_line(line: str) -> bool:
    """True unless the line is blank or starts with #, track or browser."""
    return bool(line.strip()) and not line.startswith(COMMENT_PREFIXES)


def compression(path: str | PathLike) -> str | None:
    """"gzip", "bzip2", "xz", "zip" or "zstd" from the first bytes of the
    file; None for a file that is read as plain text."""
    with open(path, "rb") as fh:
        magic = fh.read(6)
    if magic.startswith(b"\x1f\x8b"):
        return "gzip"
    if magic.startswith(b"BZh"):
        return "bzip2"
    if magic == b"\xfd7zXZ\x00":
        return "xz"
    # A local file header, or the end record that is all an empty archive holds.
    if magic.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return "zip"
    if magic.startswith(b"\x28\xb5\x2f\xfd"):
        return "zstd"
    return None


def _open_zip(path):
    with zipfile.ZipFile(path) as archive:
        # Folders and the __MACOSX/ entries that the macOS Finder adds are not files.
        members = [info for info in archive.infolist()
                   if not info.is_dir() and not info.filename.startswith("__MACOSX/")]
        if len(members) != 1:
            names = ", ".join(m.filename for m in members[:5])
            held = ("no file" if not members else
                    f"{len(members)} files ({names}{', ...' if len(members) > 5 else ''})")
            raise BedReadError(path, f"the zip archive holds {held}; "
                                     "a zipped BED must hold exactly one file")
        # The member keeps the archive's file open after the archive is closed.
        return archive.open(members[0])


class _ZstandardFrames(io.RawIOBase):
    """Every frame of a zstd file, decoded by the zstandard package.

    zstandard's stream_reader returns what it has decoded, without an
    error, when the data is corrupt or ends inside a frame; this reader
    raises EOFError then, as compression.zstd does.
    """

    def __init__(self, path, zstandard):
        super().__init__()
        self._fh = open(path, "rb")
        self._dctx = zstandard.ZstdDecompressor()
        self._chunks = self._decoded()
        self._pending = b""

    def _decoded(self) -> Iterator[bytes]:
        frame = None
        data = self._fh.read(_CHUNK)
        while data:
            if frame is None:
                frame = self._dctx.decompressobj()
            out = frame.decompress(data)
            if out:
                yield out
            if frame.eof:
                data = frame.unused_data or self._fh.read(_CHUNK)
                frame = None
            else:
                data = self._fh.read(_CHUNK)
        if frame is not None:
            raise EOFError("Compressed file ended before the end-of-stream marker was reached")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not self._pending:
            self._pending = next(self._chunks, b"")
            if not self._pending:
                return 0
        n = min(len(buffer), len(self._pending))
        buffer[:n] = self._pending[:n]
        self._pending = self._pending[n:]
        return n

    def close(self) -> None:
        self._fh.close()
        super().close()


def _open_zstd_stdlib(path):
    try:
        from compression import zstd  # Python 3.14 and later
    except ImportError:
        return None
    return zstd.open(path, "rb"), (zstd.ZstdError,)


def _open_zstd_zstandard(path):
    try:
        import zstandard
    except ImportError:
        return None
    return io.BufferedReader(_ZstandardFrames(path, zstandard)), (zstandard.ZstdError,)


# Tried in this order; each returns None when its module is missing.
ZSTD_OPENERS = [_open_zstd_stdlib, _open_zstd_zstandard]


def _open_zstd(path):
    for opener in ZSTD_OPENERS:
        opened = opener(path)
        if opened is not None:
            return opened
    raise BedReadError(path, "the file is zstd compressed, and this Python has no zstd "
                             "decoder (compression.zstd needs Python 3.14 or later; "
                             "otherwise install the zstandard package)")


# Each returns (binary stream, extra error types its decoder raises).
_OPENERS = {
    "gzip": lambda path: (gzip.open(path, "rb"), ()),
    "bzip2": lambda path: (bz2.open(path, "rb"), ()),
    "xz": lambda path: (lzma.open(path, "rb"), ()),
    "zip": lambda path: (_open_zip(path), ()),
    "zstd": _open_zstd,
}


def _checked_lines(fh, path, kind: str, errors: tuple) -> Iterator[str]:
    try:
        for line in fh:
            yield line
    except errors as exc:
        raise BedReadError(path, f"the {kind} data cannot be decompressed ({exc})") from exc


@contextmanager
def _open_text(path: str | PathLike):
    """The lines of the file as text, decompressed when it is compressed."""
    kind = compression(path)
    if kind is None:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            yield fh
        return
    try:
        raw, extra_errors = _OPENERS[kind](path)
    except BedReadError:
        raise
    except _DECOMPRESSION_ERRORS as exc:
        raise BedReadError(path, f"the {kind} data cannot be decompressed ({exc})") from exc
    with io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace") as fh:
        yield _checked_lines(fh, path, kind, _DECOMPRESSION_ERRORS + extra_errors)


def _data_lines(fh) -> Iterator[tuple[int, str]]:
    for number, line in enumerate(fh, 1):
        if is_data_line(line):
            yield number, line.rstrip("\r\n")


def numbered_bed_lines(path: str | PathLike) -> Iterator[tuple[int, str]]:
    """(line number in the file, 1-based; line without its ending) per data line."""
    with _open_text(path) as fh:
        yield from _data_lines(fh)


def bed_lines(path: str | PathLike) -> Iterator[str]:
    """The data lines of a BED file, without their line endings."""
    for _number, line in numbered_bed_lines(path):
        yield line


class _LineStream(io.TextIOBase):
    """A read-only text stream that gives each line a "\\n" ending."""

    def __init__(self, lines: Iterable[str]):
        super().__init__()
        self._lines = iter(lines)
        self._buffer = ""

    def readable(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> str:
        parts = [self._buffer]
        have = len(self._buffer)
        while size is None or size < 0 or have < size:
            line = next(self._lines, None)
            if line is None:
                break
            parts.append(line + "\n")
            have += len(line) + 1
        text = "".join(parts)
        if size is None or size < 0:
            size = len(text)
        self._buffer = text[size:]
        return text[:size]

    def readline(self, size: int | None = -1) -> str:
        if not self._buffer:
            line = next(self._lines, None)
            return "" if line is None else line + "\n"
        end = self._buffer.find("\n") + 1 or len(self._buffer)
        line, self._buffer = self._buffer[:end], self._buffer[end:]
        return line


_POSITION = re.compile(r"\b(line|row) (\d+)\b")


def _with_file_lines(message: str, path: str | PathLike) -> str:
    """A pandas message with its line (1-based) or row (0-based) numbers,
    which count data lines only, replaced by line numbers in the file."""
    wanted = {}
    for word, number in _POSITION.findall(message):
        wanted[int(number) - 1 if word == "line" else int(number)] = None
    if not wanted:
        return message
    for k, (number, _line) in enumerate(numbered_bed_lines(path)):
        if k in wanted:
            wanted[k] = number
            if all(v is not None for v in wanted.values()):
                break

    def file_line(m: re.Match) -> str:
        k = int(m.group(2)) - 1 if m.group(1) == "line" else int(m.group(2))
        return m.group(0) if wanted.get(k) is None else f"line {wanted[k]}"

    return _POSITION.sub(file_line, message)


def read_bed(path: str | PathLike, **read_csv_kwargs) -> pd.DataFrame:
    """A BED file as a DataFrame, from its data lines only (see bed_lines).

    The data lines go to pandas.read_csv with sep="\\t" and header=None
    unless the caller passes them; names=, dtype=, usecols= and the
    other read_csv arguments pass through.  comment= is refused, because
    it would cut lines at an inline "#" again.  Raises what read_csv
    raises: pandas.errors.EmptyDataError when no data line remains and
    no names are given, pandas.errors.ParserError for a line with more
    fields than the first (its line number counts every line of the
    file, as pandas does for a file it reads itself).  Raises
    BedReadError for a compressed file that cannot be read.
    """
    import pandas as pd

    if "comment" in read_csv_kwargs:
        raise TypeError("read_bed skips whole comment lines itself; comment= would "
                        "also cut lines at an inline '#'")
    kwargs = {"sep": "\t", "header": None, **read_csv_kwargs}
    with _open_text(path) as fh:
        stream = _LineStream(line for _number, line in _data_lines(fh))
        try:
            return pd.read_csv(stream, **kwargs)
        except pd.errors.ParserError as exc:
            raise pd.errors.ParserError(_with_file_lines(str(exc), path)) from exc
