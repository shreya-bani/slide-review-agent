#!/usr/bin/env python3
"""
pptx_to_md.py
Convert PowerPoint (.pptx) files to Markdown using pptx2md.

Usage examples:
  # Convert a single file; outputs next to the PPTX
  python pptx_to_md.py /path/to/slides.pptx

  # Convert many files with a glob
  python pptx_to_md.py "/path/to/deck/*.pptx"

  # Choose an output directory for .md files
  python pptx_to_md.py slides.pptx --out-dir ./md_out

  # Customize image folder name and enable slide delimiters
  python pptx_to_md.py slides.pptx --img-subdir img --enable-slides

  # Set max image width (in pixels) and disable notes extraction
  python pptx_to_md.py slides.pptx --image-width 900 --no-notes

  # Convert only a specific slide
  python pptx_to_md.py slides.pptx --page 7
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Iterable, List, Optional

# Ensure pptx2md is available
try:
    from pptx2md import convert, ConversionConfig
except Exception as e:
    sys.stderr.write(
        "Error: pptx2md is not installed or failed to import.\n"
        "Install it with:\n\n"
        "    pip install pptx2md\n\n"
        f"Details: {e}\n"
    )
    sys.exit(1)


def expand_inputs(inputs: List[str]) -> List[Path]:
    """Expand files and globs into concrete Paths; keep .pptx only."""
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        # Handle glob patterns explicitly
        if any(ch in item for ch in ["*", "?", "["]):
            for match in p.parent.glob(p.name):
                if match.is_file() and match.suffix.lower() == ".pptx":
                    paths.append(match.resolve())
        else:
            if p.is_file() and p.suffix.lower() == ".pptx":
                paths.append(p.resolve())
            elif p.is_dir():
                # If a directory is passed, take all .pptx within it (non-recursive by default)
                paths.extend(sorted((p.glob("*.pptx"))))
            else:
                sys.stderr.write(f"Warning: Skipping non-existent path: {item}\n")
    # De-duplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def destination_paths(
    src: Path,
    out_dir: Optional[Path],
    img_subdir: str,
) -> tuple[Path, Path]:
    """
    Compute the Markdown output path and image directory for a given source file.
    If out_dir is provided, place the .md there; otherwise, next to the source.
    Images go to <base>_img by default (or to out_dir/<base>/<img_subdir> if out_dir is set).
    """
    base = src.stem
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / f"{base}.md"
        img_dir = out_dir / base / img_subdir
    else:
        md_path = src.with_suffix(".md")
        img_dir = src.parent / f"{base}_img"
    img_dir.mkdir(parents=True, exist_ok=True)
    return md_path, img_dir


def convert_one(
    src: Path,
    md_path: Path,
    img_dir: Path,
    enable_slides: bool,
    image_width: Optional[int],
    include_notes: bool,
    titles_file: Optional[Path],
    page: Optional[int],
    disable_wmf: bool,
) -> None:
    """Run pptx2md conversion for a single file."""
    cfg_kwargs = dict(
        pptx_path=src,
        output_path=md_path,
        image_dir=img_dir,
        enable_slides=enable_slides,
        disable_notes=not include_notes,
    )

    if image_width is not None:
        cfg_kwargs["image_width"] = image_width
    if titles_file is not None:
        cfg_kwargs["titles_path"] = titles_file
    if page is not None:
        cfg_kwargs["page"] = page
    if disable_wmf:
        cfg_kwargs["disable_wmf"] = True

    cfg = ConversionConfig(**cfg_kwargs)
    convert(cfg)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert .pptx to Markdown using pptx2md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="One or more .pptx files, directories, or glob patterns.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write .md files. Defaults to writing next to each source .pptx.",
    )
    p.add_argument(
        "--img-subdir",
        type=str,
        default="img",
        help="Image folder name under each deck's folder (when using --out-dir). If --out-dir is not set, images go to <deck>_img next to the source.",
    )
    p.add_argument(
        "--enable-slides",
        action="store_true",
        help="Insert slide delimiters into the Markdown (useful for slide frameworks).",
    )
    p.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Max width (px) for exported images. If omitted, uses pptx2md default.",
    )
    p.add_argument(
        "--notes",
        dest="include_notes",
        action="store_true",
        help="Include speaker notes in the Markdown.",
    )
    p.add_argument(
        "--no-notes",
        dest="include_notes",
        action="store_false",
        help="Exclude speaker notes from the Markdown.",
    )
    p.set_defaults(include_notes=False)
    p.add_argument(
        "--titles",
        type=Path,
        default=None,
        help="Path to a titles outline file to guide heading levels.",
    )
    p.add_argument(
        "--page",
        type=int,
        default=None,
        help="Convert only a specific slide number (1-based).",
    )
    p.add_argument(
        "--disable-wmf",
        action="store_true",
        help="Disable WMF image conversion (workaround for certain decks).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    files = expand_inputs(args.inputs)
    if not files:
        sys.stderr.write("No .pptx files found to convert.\n")
        return 2

    converted = 0
    failed: List[tuple[Path, str]] = []

    for src in files:
        try:
            md_path, img_dir = destination_paths(src, args.out_dir, args.img_subdir)
            convert_one(
                src=src,
                md_path=md_path,
                img_dir=img_dir,
                enable_slides=args.enable_slides,
                image_width=args.image_width,
                include_notes=args.include_notes,
                titles_file=args.titles,
                page=args.page,
                disable_wmf=args.disable_wmf,
            )
            print(f"✓ {src.name} → {md_path} (images: {img_dir})")
            converted += 1
        except KeyboardInterrupt:
            sys.stderr.write("\nAborted by user.\n")
            return 130
        except Exception as e:
            failed.append((src, str(e)))
            sys.stderr.write(f"✗ Failed: {src} — {e}\n")

    if failed:
        sys.stderr.write("\nSummary of failures:\n")
        for src, msg in failed:
            sys.stderr.write(f"  - {src}: {msg}\n")

    print(f"\nDone. Converted: {converted}, Failed: {len(failed)}")
    return 0 if converted > 0 and not failed else (1 if converted == 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
