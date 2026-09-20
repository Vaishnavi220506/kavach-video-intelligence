"""Download and self-host the website's typefaces.

The site self-hosts its faces rather than linking a font CDN: it removes a
third-party request from every page load, keeps the site working offline
alongside the local backend, and means the exact font binaries are recorded in
the repository instead of resolved at runtime.

All three families are licensed under the SIL Open Font License 1.1, which
permits redistribution. ``frontend/public/fonts/OFL.txt`` carries the licence
and the attribution required alongside the committed files.

Only the ``latin`` and ``latin-ext`` subsets are kept. The site's copy is
English, and the Cyrillic, Greek and Vietnamese subsets would roughly triple
the payload for glyphs that never render.

Usage::

    python scripts/fetch_fonts.py
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "frontend" / "public" / "fonts"

# A browser user agent is required: the Google Fonts CSS endpoint serves
# older formats (ttf) to clients it does not recognise, and only returns
# woff2 to a modern browser.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

KEPT_SUBSETS = ("latin", "latin-ext")

# family label -> Google Fonts css2 query.
#
# Spectral is the document face: a serif built for screen reading with enough
# character to carry the finding at display size.
# Archivo is the label and interface grotesque, the report's caption voice.
# JetBrains Mono is reserved for measured quantities and has true tabular
# figures, so columns of scores and timecodes align.
FAMILIES = {
    "spectral": "Spectral:ital,wght@0,400;0,600;0,700;1,400",
    "archivo": "Archivo:wght@400..700",
    "jetbrains-mono": "JetBrains+Mono:wght@400..700",
}

BLOCK_PATTERN = re.compile(
    r"/\*\s*(?P<subset>[a-z0-9\-]+)\s*\*/\s*(?P<block>@font-face\s*\{.*?\})",
    re.DOTALL,
)
URL_PATTERN = re.compile(r"url\((?P<url>https://[^)]+\.woff2)\)")
PROPERTY_PATTERN = re.compile(r"(?P<name>[a-z\-]+)\s*:\s*(?P<value>[^;]+);")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def build_family(label: str, query: str, output: Path) -> list[str]:
    """Download one family's kept subsets and return local @font-face rules."""

    css = fetch(f"https://fonts.googleapis.com/css2?family={query}&display=swap")
    css_text = css.decode("utf-8")

    rules: list[str] = []
    index = 0
    for match in BLOCK_PATTERN.finditer(css_text):
        if match.group("subset") not in KEPT_SUBSETS:
            continue
        block = match.group("block")
        url_match = URL_PATTERN.search(block)
        if url_match is None:
            continue

        properties = {
            item.group("name"): item.group("value").strip()
            for item in PROPERTY_PATTERN.finditer(block)
        }

        style = properties.get("font-style", "normal")
        # A variable family reports a range such as "400 700". Keep it intact
        # for the declaration and only flatten it for the filename, or the
        # rule collapses to a single nonsense weight of 400700.
        weight = properties.get("font-weight", "400").strip()
        weight_slug = weight.replace(" ", "-")
        suffix = "italic" if style == "italic" else "normal"
        filename = f"{label}-{weight_slug}-{suffix}-{match.group('subset')}.woff2"
        index += 1

        payload = fetch(url_match.group("url"))
        (output / filename).write_bytes(payload)

        declarations = [
            f"  font-family: {properties.get('font-family', label)};",
            f"  font-style: {style};",
            f"  font-weight: {weight};",
            "  font-display: swap;",
            f"  src: url('/fonts/{filename}') format('woff2');",
        ]
        if "unicode-range" in properties:
            declarations.append(f"  unicode-range: {properties['unicode-range']};")
        rules.append("@font-face {\n" + "\n".join(declarations) + "\n}")

        print(f"  {filename}  {len(payload) / 1024:.1f} KB")

    if not rules:
        raise RuntimeError(f"no {KEPT_SUBSETS} subsets found for {label}")
    return rules


OFL_NOTICE = """\
The font files in this directory are redistributed under the SIL Open Font
License, Version 1.1. They were downloaded by scripts/fetch_fonts.py.

  Spectral         Copyright 2017 Production Type
                   https://fonts.google.com/specimen/Spectral
  Archivo          Copyright 2011 Omnibus-Type
                   https://fonts.google.com/specimen/Archivo
  JetBrains Mono   Copyright 2020 JetBrains
                   https://fonts.google.com/specimen/JetBrains+Mono

Each family is licensed under the SIL Open Font License, Version 1.1, which
is available with a FAQ at https://openfontlicense.org.

The licence permits redistribution of the font files, bundled with software,
provided this notice is retained. The fonts are not sold on their own, and
the KAVACH MIT licence in the repository root applies to the project's own
source, not to these font binaries.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    all_rules: list[str] = []
    for label, query in FAMILIES.items():
        print(f"{label}:")
        try:
            all_rules.extend(build_family(label, query, args.output))
        except Exception as error:
            print(f"  failed: {error}", file=sys.stderr)
            raise SystemExit(1) from error

    header = (
        "/* Generated by scripts/fetch_fonts.py -- do not edit by hand.\n"
        "   Self-hosted latin and latin-ext subsets. See OFL.txt for licensing. */\n\n"
    )
    (args.output / "fonts.css").write_text(
        header + "\n\n".join(all_rules) + "\n", encoding="utf-8"
    )
    (args.output / "OFL.txt").write_text(OFL_NOTICE, encoding="utf-8")

    total = sum(path.stat().st_size for path in args.output.glob("*.woff2"))
    print(f"\n{len(all_rules)} faces, {total / 1024:.1f} KB total")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
