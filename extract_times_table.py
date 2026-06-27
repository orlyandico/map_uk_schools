#!/usr/bin/env python3
"""Extract a Times / Sunday Times "Parent Power" best-schools league table to CSV.

The public page at https://www.thetimes.com/best-schools-league-table only embeds
a top-10 teaser. The full ranked tables are served, fully server-rendered (no JS,
no auth, no device gate), by the Parent Power app:

    https://dlv.tnl-parent-power.gcpp.io/2025?filterId=<FILTER_ID>

Pass a filterId to pick which list. Examples seen in the app's category dropdown:

    the-top-combined-secondary-schools          (joint state + private secondary)
    the-top-state-secondary-schools
    the-top-independent-secondary-schools
    the-top-500-english-state-primary-schools

The table markup uses class `pp-main-table-2023`. Column count and headers vary by
filter (secondary has A-level/GCSE columns; primary has Reading/Grammar/Maths), so
this script reads the header row from the page and adapts.

Usage:
    python3 extract_times_table.py <filterId> <output.csv> [--year 2025]

Requires only the standard library plus a way to fetch the page. By default it
shells out to curl; pass --html <file> to parse an already-saved page instead.
"""
import argparse
import csv
import html as ihtml
import re
import subprocess
import sys

BASE = "https://dlv.tnl-parent-power.gcpp.io"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def fetch(filter_id: str, year: str) -> str:
    url = f"{BASE}/{year}?filterId={filter_id}"
    out = subprocess.run(
        ["curl", "-s", "--max-time", "60", "-A", UA, url],
        capture_output=True, text=True, check=True,
    )
    if len(out.stdout) < 10000:
        sys.exit(f"Unexpectedly small response from {url} ({len(out.stdout)} bytes)")
    return out.stdout


def clean(s: str) -> str:
    s = ihtml.unescape(re.sub(r"<[^>]+>", "", s)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse(src: str):
    hh = re.search(r"pp-main-table-2023__table--header.*?</tr>", src, re.S)
    if not hh:
        sys.exit("No pp-main-table-2023 header found - is the filterId valid?")
    headers = [clean(x) for x in re.findall(r"<th[^>]*>(.*?)</th>", hh.group(0), re.S)]
    ncol = len(headers)

    rows = re.findall(r'<tr class="pp-main-table-2023__table--row".*?</tr>', src, re.S)
    out = []
    for r in rows:
        cells = re.findall(
            r'<td class="pp-main-table-2023__table--data[^"]*"[^>]*>(.*?)</td>', r, re.S
        )
        rec = []
        for idx, c in enumerate(cells):
            if idx == 0:  # rank cell: keep only the rank-number span (drop movement marker)
                rn = re.search(r"rank-number[^>]*>(.*?)</span>", c, re.S)
                rec.append(clean(rn.group(1)) if rn else clean(c))
                continue
            # entry-gender is an icon cell: male-icon and/or female-icon images
            alts = re.findall(r'alt="([^"]+)"', c)
            if alts and not clean(c):
                boys, girls = "Boys" in alts, "Girls" in alts
                rec.append("Mixed" if boys and girls else ("Boys" if boys else "Girls"))
            else:
                rec.append(clean(c))
        if len(rec) >= ncol:
            out.append(rec[:ncol])
    return headers, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter_id", help="e.g. the-top-combined-secondary-schools")
    ap.add_argument("output", help="output CSV path")
    ap.add_argument("--year", default="2025")
    ap.add_argument("--html", help="parse this saved HTML file instead of fetching")
    args = ap.parse_args()

    src = open(args.html).read() if args.html else fetch(args.filter_id, args.year)
    headers, rows = parse(src)

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    print(f"Wrote {len(rows)} schools, {len(headers)} columns -> {args.output}")
    print("Columns:", ", ".join(headers))


if __name__ == "__main__":
    main()
