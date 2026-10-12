"""BED reading: only whole lines starting with #, track or browser (and blank
lines) are comments; a "#" inside a line is data (ChIP-Atlas "Catalog#: ...").
Compressed files are decompressed, and a damaged one raises BedReadError."""
import asyncio
import bz2
import gzip
import json
import lzma
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import torch

from tfbs import bedio

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "tfbs" / "data" / "demo"
GENOME = DEMO / "demo_genome.fa"
CATALOG = "ID=SRX{k};Name=NKX2-1;Title=lung ChIP-seq;Catalog#: MS-699-P"
HEADER = ["track name=NKX2-1 description=\"ChIP-Atlas\"",
          "browser position demo_chr8:1-10001",
          "# chrom\tstart\tend\tname\tscore",
          ""]


def run(coro):
    return json.loads(asyncio.run(coro))


def _demo_rows():
    """The four demo peaks with a ChIP-Atlas catalog number in column 4."""
    rows = [line.split("\t") for line in (DEMO / "demo_peaks.bed").read_text().splitlines()]
    for k, row in enumerate(rows):
        row[3] = CATALOG.format(k=k)
    return rows


def _write(path, lines, *, bom=False, newline="\n"):
    text = newline.join(lines) + newline
    path.write_bytes((("﻿" if bom else "") + text).encode("utf-8"))
    return path


def test_inline_hash_keeps_every_row_and_score(tmp_path):
    rows = _demo_rows()
    bed = _write(tmp_path / "p.bed", ["\t".join(r) for r in rows])
    df = bedio.read_bed(bed)
    assert df.shape == (4, 6)
    assert df[3].tolist() == [CATALOG.format(k=k) for k in range(4)]
    assert df[4].tolist() == [10.37, 12.06, 10.36, 8.99]
    # What the old reader did: the line is cut at the "#", the score is lost.
    old = pd.read_csv(bed, sep="\t", header=None, comment="#")
    assert old.shape == (4, 4)


def test_comment_track_browser_and_blank_lines_are_skipped(tmp_path):
    lines = HEADER + ["chr1\t10\t20\tx#1\t5", "", "   ", "# mid-file comment",
                      "chr1\t30\t40\ty\t6"]
    bed = _write(tmp_path / "p.bed", lines)
    assert list(bedio.bed_lines(bed)) == ["chr1\t10\t20\tx#1\t5", "chr1\t30\t40\ty\t6"]
    assert [n for n, _ in bedio.numbered_bed_lines(bed)] == [5, 9]
    df = bedio.read_bed(bed)
    assert df.values.tolist() == [["chr1", 10, 20, "x#1", 5], ["chr1", 30, 40, "y", 6]]


def test_bom_and_crlf_line_endings(tmp_path):
    bed = _write(tmp_path / "p.bed", HEADER + ["chr1\t10\t20\tCatalog#: 1\t5.5"],
                 bom=True, newline="\r\n")
    assert list(bedio.bed_lines(bed)) == ["chr1\t10\t20\tCatalog#: 1\t5.5"]
    df = bedio.read_bed(bed)
    assert df.values.tolist() == [["chr1", 10, 20, "Catalog#: 1", 5.5]]
    # The BOM is not part of the first chrom, even without header lines.
    bare = _write(tmp_path / "bare.bed", ["chr2\t1\t2"], bom=True, newline="\r\n")
    assert bedio.read_bed(bare).values.tolist() == [["chr2", 1, 2]]


def _zip(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _zstd_compress(data: bytes) -> bytes:
    try:
        from compression import zstd
    except ImportError:
        return pytest.importorskip("zstandard").ZstdCompressor().compress(data)
    return zstd.compress(data)


def test_compressed_files_read_like_plain_ones(tmp_path):
    lines = HEADER + ["\t".join(r) for r in _demo_rows()]
    plain = _write(tmp_path / "p.bed", lines)
    data = plain.read_bytes()
    gz = tmp_path / "p.bed.gz"
    gz.write_bytes(gzip.compress(data))
    bz = tmp_path / "p.bed.bz2"
    bz.write_bytes(bz2.compress(data))
    xz = tmp_path / "p.bed.xz"
    xz.write_bytes(lzma.compress(data))
    zp = _zip(tmp_path / "p.bed.zip", {"p.bed": data})
    # The macOS Finder adds a __MACOSX/ entry; folders are not files either.
    mac = _zip(tmp_path / "mac.zip", {"peaks/": b"", "peaks/p.bed": data,
                                      "__MACOSX/peaks/._p.bed": b"\x00\x05"})
    # The name does not matter, the first bytes do.
    renamed = tmp_path / "gzip_without_suffix.bed"
    renamed.write_bytes(gzip.compress(data))
    expected = bedio.read_bed(plain)
    assert expected.shape == (4, 6)
    assert [bedio.compression(f) for f in (plain, gz, bz, xz, zp, mac, renamed)] == [
        None, "gzip", "bzip2", "xz", "zip", "zip", "gzip"]
    for packed in (gz, bz, xz, zp, mac, renamed):
        assert bedio.read_bed(packed).equals(expected), packed.name
        assert list(bedio.bed_lines(packed)) == list(bedio.bed_lines(plain))


ZSTD_DECODERS = {"compression.zstd": "_open_zstd_stdlib", "zstandard": "_open_zstd_zstandard"}


@pytest.fixture(params=sorted(ZSTD_DECODERS))
def zstd_decoder(request, monkeypatch):
    """Only one zstd decoder: Python 3.14's compression.zstd, or the zstandard package."""
    pytest.importorskip(request.param)
    monkeypatch.setattr(bedio, "ZSTD_OPENERS", [getattr(bedio, ZSTD_DECODERS[request.param])])
    return request.param


def test_zstd_files_read_like_plain_ones_with_every_frame(zstd_decoder, tmp_path):
    lines = HEADER + ["\t".join(r) for r in _demo_rows()]
    plain = _write(tmp_path / "p.bed", lines)
    data = plain.read_bytes()
    zst = tmp_path / "p.bed.zst"
    # Two frames, as concatenated .zst files give.
    zst.write_bytes(_zstd_compress(data[:100]) + _zstd_compress(data[100:]))
    assert bedio.compression(zst) == "zstd"
    assert bedio.read_bed(zst).equals(bedio.read_bed(plain))
    assert list(bedio.bed_lines(zst)) == list(bedio.bed_lines(plain))


def test_truncated_or_corrupt_zstd_raises_bed_read_error(zstd_decoder, tmp_path):
    packed = _zstd_compress("".join("\t".join(r) + "\n" for r in _demo_rows()).encode())
    cut = tmp_path / "cut.bed.zst"
    cut.write_bytes(packed[:-10])
    garbage = tmp_path / "garbage.bed.zst"
    garbage.write_bytes(packed[:4] + bytes(40))
    for bed in (cut, garbage):
        with pytest.raises(bedio.BedReadError) as caught:
            bedio.read_bed(bed)
        assert caught.value.reason.startswith("the zstd data cannot be decompressed ("), bed.name


def test_zstd_without_a_decoder_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(bedio, "ZSTD_OPENERS", [])
    bed = tmp_path / "p.bed.zst"
    bed.write_bytes(b"\x28\xb5\x2f\xfd" + bytes(20))
    with pytest.raises(bedio.BedReadError, match="no zstd decoder"):
        list(bedio.bed_lines(bed))


def test_a_zip_archive_must_hold_exactly_one_file(tmp_path):
    two = _zip(tmp_path / "two.zip", {"a.bed": b"chr1\t1\t2\n", "b.bed": b"chr1\t3\t4\n"})
    empty = _zip(tmp_path / "empty.zip", {})
    assert bedio.compression(empty) == "zip"
    for bed, held in ((two, "2 files (a.bed, b.bed)"), (empty, "no file")):
        with pytest.raises(bedio.BedReadError) as caught:
            bedio.read_bed(bed)
        assert caught.value.reason == (f"the zip archive holds {held}; "
                                       "a zipped BED must hold exactly one file")
        assert isinstance(caught.value, OSError)


def _damaged(data: bytes, start: int) -> bytes:
    """data with 40 bytes from start inverted."""
    return data[:start] + bytes(b ^ 0xFF for b in data[start:start + 40]) + data[start + 40:]


def _damaged_files(tmp_path):
    data = "".join("\t".join(r) + "\n" for r in _demo_rows()).encode() * 200
    gz, bz, xz = gzip.compress(data), bz2.compress(data), lzma.compress(data)
    zp = _zip(tmp_path / "whole.zip", {"p.bed": data}).read_bytes()
    files = {
        "gzip corrupt": ("gzip", _damaged(gz, 20)),
        "gzip truncated": ("gzip", gz[: len(gz) // 2]),
        "bzip2 corrupt": ("bzip2", _damaged(bz, 20)),
        "xz corrupt": ("xz", _damaged(xz, 30)),
        "xz truncated": ("xz", xz[: len(xz) // 2]),
        "zip member corrupt": ("zip", _damaged(zp, 45)),
        "zip truncated": ("zip", zp[: len(zp) // 2]),
    }
    paths = {}
    for label, (kind, blob) in files.items():
        path = tmp_path / (label.replace(" ", "_") + ".bed")
        path.write_bytes(blob)
        paths[label] = (kind, path)
    return paths


def test_damaged_compressed_files_raise_bed_read_error(tmp_path):
    for label, (kind, bed) in _damaged_files(tmp_path).items():
        for read in (bedio.read_bed, lambda path: list(bedio.bed_lines(path))):
            with pytest.raises(bedio.BedReadError) as caught:
                read(bed)
            assert caught.value.reason.startswith(f"the {kind} data cannot be decompressed ("), label


def test_a_large_file_crosses_pandas_read_chunks(tmp_path):
    lines = [f"chr1\t{k}\t{k + 5}\t{CATALOG.format(k=k)}\t{k}.5" for k in range(8000)]
    bed = _write(tmp_path / "big.bed", HEADER + lines, newline="\r\n")
    assert bed.stat().st_size > 2 * 262144
    for engine in ("c", "python"):
        df = bedio.read_bed(bed, engine=engine)
        assert df.shape == (8000, 5)
        assert df[4].isna().sum() == 0
        assert df[4].iloc[-1] == 7999.5


def test_only_comment_lines_raise_empty_data_error(tmp_path):
    bed = _write(tmp_path / "p.bed", HEADER + ["# nothing else"])
    with pytest.raises(pd.errors.EmptyDataError):
        bedio.read_bed(bed)
    assert bedio.read_bed(bed, names=["chrom", "start", "end"]).shape == (0, 3)


def test_parser_error_names_the_line_of_the_file(tmp_path):
    bed = _write(tmp_path / "p.bed", HEADER + ["chr1\t1\t2", "# c", "chr1\t3\t4\tx\t5"])
    with pytest.raises(pd.errors.ParserError, match=r"Expected 3 fields in line 7, saw 5"):
        bedio.read_bed(bed)


def test_comment_argument_is_refused(tmp_path):
    bed = _write(tmp_path / "p.bed", ["chr1\t1\t2"])
    with pytest.raises(TypeError, match="comment"):
        bedio.read_bed(bed, comment="#")


def test_names_usecols_and_dtype_pass_through(tmp_path):
    bed = _write(tmp_path / "p.bed", HEADER + ["chr1\t1\t2\tCatalog#: 3\t4"])
    df = bedio.read_bed(bed, names=["chrom", "start", "end", "name", "score"],
                        usecols=["chrom", "score"], dtype={"chrom": str, "score": float})
    assert df.to_dict("records") == [{"chrom": "chr1", "score": 4.0}]


# ---------------------------------------------------------------------------
# The readers that use it
# ---------------------------------------------------------------------------

@pytest.fixture
def server(monkeypatch, tmp_path):
    from tfbs.mcp import server
    monkeypatch.setattr(server, "_results_root", lambda: tmp_path / "results")
    return server


def test_extract_loci_keeps_scores_after_an_inline_hash(server, tmp_path):
    bed = tmp_path / "peaks" / "chipatlas.bed"
    bed.parent.mkdir()
    _write(bed, HEADER + ["\t".join(r) for r in _demo_rows()], bom=True, newline="\r\n")
    result = run(server.tfbs_extract_loci(bed_path=str(bed), genome_fasta=str(GENOME),
                                          score_column=4))
    assert "error" not in result, result
    assert result["num_peaks_input"] == 4
    assert result["num_peaks_extracted"] == 4
    assert result["num_filtered_score"] == 0
    assert result["has_scores"] is True
    scores = torch.load(result["files"]["scores"], weights_only=True)
    assert scores.shape == (4, 1)
    assert scores[:, 0].tolist() == pytest.approx([10.37, 12.06, 10.36, 8.99])


def test_extract_loci_with_only_comment_lines_reports_the_empty_file(server, tmp_path):
    bed = _write(tmp_path / "empty.bed", HEADER)
    result = run(server.tfbs_extract_loci(bed_path=str(bed), genome_fasta=str(GENOME)))
    assert result["error"].startswith(f"Could not read BED file {bed.resolve()}: ")
    assert "No columns to parse from file" in result["error"]
    assert "chrom, start and end" in result["fix"]


def test_extract_loci_reports_a_damaged_compressed_bed(server, tmp_path):
    for label, (kind, bed) in _damaged_files(tmp_path).items():
        result = run(server.tfbs_extract_loci(bed_path=str(bed), genome_fasta=str(GENOME)))
        assert result["error"].startswith(
            f"Could not read BED file {bed.resolve()}: the {kind} data cannot be decompressed ("
        ), label
        assert result["fix"].startswith("Decompress the file yourself"), label


def test_extract_loci_reads_zip_and_zstd_beds(server, tmp_path):
    data = "\n".join(HEADER + ["\t".join(r) for r in _demo_rows()]).encode() + b"\n"
    zp = _zip(tmp_path / "chipatlas.bed.zip", {"chipatlas.bed": data})
    zst = tmp_path / "chipatlas.bed.zst"
    zst.write_bytes(_zstd_compress(data))
    for bed in (zp, zst):
        result = run(server.tfbs_extract_loci(bed_path=str(bed), genome_fasta=str(GENOME),
                                              score_column=4,
                                              output_dir=str(tmp_path / bed.suffix[1:])))
        assert "error" not in result, result
        assert result["num_peaks_extracted"] == 4
        scores = torch.load(result["files"]["scores"], weights_only=True)
        assert scores[:, 0].tolist() == pytest.approx([10.37, 12.06, 10.36, 8.99])


def test_extract_loci_without_a_zstd_decoder_says_so(server, monkeypatch, tmp_path):
    monkeypatch.setattr(bedio, "ZSTD_OPENERS", [])
    bed = tmp_path / "p.bed.zst"
    bed.write_bytes(b"\x28\xb5\x2f\xfd" + bytes(20))
    result = run(server.tfbs_extract_loci(bed_path=str(bed), genome_fasta=str(GENOME)))
    assert result["error"] == (
        f"Could not read BED file {bed.resolve()}: the file is zstd compressed, and this "
        "Python has no zstd decoder (compression.zstd needs Python 3.14 or later; "
        "otherwise install the zstandard package)")


def test_extract_loci_header_hint_names_the_comment_rule(server, tmp_path):
    bed = _write(tmp_path / "h.bed", ["chrom\tstart\tend", "demo_chr8\t310\t510"])
    result = run(server.tfbs_extract_loci(bed_path=str(bed), genome_fasta=str(GENOME)))
    assert result["error"].startswith("Non-integer start or end")
    assert "#, track or browser" in result["fix"]


def test_remap_reader_keeps_a_hash_in_the_name_column(tmp_path):
    from tfbs.pipelines import remap_validation as rv

    bed = _write(tmp_path / "remap.bed", [
        "track name=ReMap2022",
        "chr1\t100\t300\tNKX2-1:A549#rep1\t3\t.\t200\t201\t0,0,255",
        "chr1\t500\t700\tNKX2-1:H441\t1\t.\t600\t601\t0,0,255",
    ], newline="\r\n")
    df = rv.read_remap_bed(bed)
    assert df.columns.tolist() == rv.REMAP_COLUMNS
    assert df["name"].tolist() == ["NKX2-1:A549#rep1", "NKX2-1:H441"]
    assert df["score"].tolist() == [3, 1]
    assert df["thickStart"].tolist() == [200, 600]


def test_disease_peaks_reader_reads_a_gzipped_bed_like_a_plain_one(tmp_path):
    from tfbs.pipelines import disease_variant_validation as dv

    lines = HEADER + ["chr1\t10\t20\t3", "chr1\t30\t40\t2"]
    plain = _write(tmp_path / "p.bed", lines, newline="\r\n")
    gz = tmp_path / "p.bed.gz"
    gz.write_bytes(gzip.compress(plain.read_bytes()))
    expected = dv.read_peaks_bed(str(plain))
    assert expected.to_dict("records") == [
        {"chrom": "chr1", "start": 10, "end": 20, "num": 3},
        {"chrom": "chr1", "start": 30, "end": 40, "num": 2},
    ]
    assert dv.read_peaks_bed(str(gz)).equals(expected)


def test_disease_peaks_reader_names_the_line_of_the_file(tmp_path):
    from tfbs.pipelines import disease_variant_validation as dv

    bed = _write(tmp_path / "p.bed", HEADER + ["chr1\t10\t20", "chr1\tx\t40"])
    with pytest.raises(ValueError, match=r"line 6: start and end must be integers"):
        dv.read_peaks_bed(str(bed))
