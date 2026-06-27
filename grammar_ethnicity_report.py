#!/usr/bin/env python3
"""
Generate a per-school ethnicity report for the Times / Sunday Times Parent Power
state-secondary league table, using the DfE "Schools, pupils and their
characteristics" school-level census for the ethnicity figures.

The cohort is read from the Parent Power state-secondary CSV (every school by
default) and each school is matched to a census URN by name + location at runtime
(see build_cohort). The report is sorted by Parent Power rank, rank 1 first.

Data sources:
  - Cohort + rank: Times Parent Power state-secondary table (best_schools_2025_
    state_secondary.csv; produce it with extract_times_table.py).
  - Ethnicity: DfE, Schools, pupils and their characteristics (school-level
    underlying data, one row per school, unsuppressed). The default URL points at
    the Jan 2025 (2024/25) release. Update CSV_URL when a new release is published.

Usage:
    python3 grammar_ethnicity_report.py                 # download census + report
    python3 grammar_ethnicity_report.py --csv data.csv  # use a local census CSV
    python3 grammar_ethnicity_report.py --top 100        # exactly 100 matched schools
    python3 grammar_ethnicity_report.py -o out.md        # choose output path

Notes: only state-funded schools appear in the census. Independent schools submit
an aggregate return with no ethnicity breakdown. The census is England-only, so
Northern Irish and Welsh schools in the Times table cannot be matched.
"""

import argparse
import csv
import http.client
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import date

# --- Configuration -----------------------------------------------------------

CSV_URL = (
    "https://content.explore-education-statistics.service.gov.uk/api/releases/"
    "63491b17-2037-4533-b719-d3656aaf6ed5/files/3dc88c32-da52-4aff-b6d0-0126de016844"
)
CENSUS_LABEL = "DfE Schools, Pupils and their Characteristics — January 2025 census"
DEFAULT_CACHE = "spc_school_level.csv"
DEFAULT_OUTPUT = "grammar_ethnicity_report.md"

# The cohort is the top N of the Times / Sunday Times Parent Power state-secondary
# league table, matched to census URNs at runtime (see build_cohort). Produce the
# CSV with: python3 extract_times_table.py the-top-state-secondary-schools \
#           best_schools_2025_state_secondary.csv
TIMES_SECONDARY_CSV = "best_schools_2025_state_secondary.csv"
COHORT_SIZE = None  # None = every school in the Times table
MATCH_THRESHOLD = 0.6  # min name-token Jaccard (+0.2 location bonus) to accept

# Column headers in the DfE file. Keys are short names used internally.
def _c(s):
    return "% of pupils classified as " + s

PCT_COLS = {
    "WhiteBrit": _c("white British ethnic origin"),
    "OthWhite": _c("any other white background ethnic origin"),
    "Indian": _c("Indian ethnic origin"),
    "Pak": _c("Pakistani ethnic origin"),
    "Bang": _c("Bangladeshi ethnic origin"),
    "OthAsian": _c("any other Asian background ethnic origin"),
    "Chinese": _c("Chinese ethnic origin"),
    "African": _c("African ethnic origin"),
    "Carib": _c("Caribbean ethnic origin"),
    "OthBlack": _c("any other black background ethnic origin"),
    "WhAsian": _c("white and Asian ethnic origin"),
    "WhBAfr": _c("white and black African ethnic origin"),
    "WhBCar": _c("white and black Caribbean ethnic origin"),
    "OthMixed": _c("any other mixed background ethnic origin"),
    "OthEth": _c("any other ethnic group ethnic origin"),
    "EAL": "% of pupils whose first language is known or believed to be other than English",
}
HEADCOUNT_COL = "headcount of pupils"
NAME_COL = "school_name"
URN_COL = "urn"
PHASE_COL = "phase_type_grouping"
SECONDARY_PHASE = "State-funded secondary"
LA_COL = "la_name"
DISTRICT_COL = "district_administrative_name"

# Detailed-group columns shown in the breakdown table (in order).
DETAIL_GROUPS = [
    ("WhiteBrit", "White Brit"),
    ("OthWhite", "Oth White"),
    ("Indian", "Indian"),
    ("Pak", "Pakistani"),
    ("Bang", "Bangladeshi"),
    ("OthAsian", "Oth Asian"),
    ("Chinese", "Chinese"),
    ("African", "Black African"),
    ("Mixed", "Mixed"),
    ("Other", "Other"),
]

# --- Helpers -----------------------------------------------------------------

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def download(url, dest):
    """Download url to dest.

    The EES content API serves this file over HTTP/2 and advertises no
    Accept-Ranges, so a dropped connection can't be resumed. Python's stdlib
    speaks HTTP/1.1 only and the CDN frequently truncates those responses
    (IncompleteRead / short reads). curl negotiates HTTP/2 and fetches the file
    reliably, so we use it when present and fall back to a stdlib retry loop
    otherwise.
    """
    print(f"Downloading census file -> {dest} ...", file=sys.stderr)
    tmp = dest + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    if shutil.which("curl"):
        try:
            _download_curl(url, tmp)
            os.replace(tmp, dest)
            print(f"  saved {os.path.getsize(dest):,} bytes", file=sys.stderr)
            return
        except Exception as e:  # fall back to stdlib
            print(f"  curl download failed ({e}); falling back to urllib",
                  file=sys.stderr)
            if os.path.exists(tmp):
                os.remove(tmp)

    _download_urllib(url, tmp)
    os.replace(tmp, dest)
    print(f"  saved {os.path.getsize(dest):,} bytes", file=sys.stderr)


def _download_curl(url, tmp):
    """Fetch url to tmp with curl (HTTP/2, retries). Validates size if known."""
    subprocess.run(
        ["curl", "-sS", "--fail", "--http2", "--retry", "5",
         "--retry-all-errors", "--max-time", "300",
         "-A", "Mozilla/5.0", "-o", tmp, url],
        check=True,
    )
    expected = _content_length(url)
    got = os.path.getsize(tmp)
    if expected is not None and got < expected:
        raise RuntimeError(f"curl got {got:,} of {expected:,} bytes")


def _content_length(url):
    """Best-effort Content-Length via a HEAD request; None if unavailable."""
    try:
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            clen = r.getheader("Content-Length")
            return int(clen) if clen and clen.isdigit() else None
    except Exception:
        return None


def _download_urllib(url, tmp, attempts=5, chunk_size=65536):
    """Stdlib fallback: stream to tmp, resuming with Range when the server
    allows it, otherwise retrying the whole transfer."""
    downloaded = 0
    total = None

    for attempt in range(1, attempts + 1):
        headers = {"User-Agent": "Mozilla/5.0"}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                # If we asked to resume but the server ignored Range (status 200
                # rather than 206), start over from the beginning.
                if downloaded and getattr(r, "status", None) != 206:
                    downloaded = 0
                mode = "ab" if downloaded else "wb"

                if total is None:
                    crange = r.getheader("Content-Range")
                    clen = r.getheader("Content-Length")
                    if crange and "/" in crange and crange.rsplit("/", 1)[-1].isdigit():
                        total = int(crange.rsplit("/", 1)[-1])
                    elif clen and clen.isdigit():
                        total = int(clen) + downloaded

                with open(tmp, mode) as f:
                    try:
                        while True:
                            chunk = r.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                    except http.client.IncompleteRead as e:
                        if e.partial:
                            f.write(e.partial)
                            downloaded += len(e.partial)
                        raise
        except (http.client.IncompleteRead, urllib.error.URLError,
                ConnectionError, TimeoutError) as e:
            downloaded = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if attempt == attempts:
                raise RuntimeError(
                    f"Download failed after {attempts} attempts: {e}"
                ) from e
            print(
                f"  interrupted ({e}); retry {attempt}/{attempts - 1} "
                f"resuming from byte {downloaded:,}",
                file=sys.stderr,
            )
            time.sleep(2 * attempt)
            continue

        # Completed a response without exception.
        if total is None or downloaded >= total:
            break
        # Server closed cleanly but short — resume.
        print(
            f"  short read ({downloaded:,}/{total:,} bytes); "
            f"resuming (attempt {attempt}/{attempts})",
            file=sys.stderr,
        )

    if total is not None and downloaded < total:
        raise RuntimeError(
            f"Incomplete download: got {downloaded:,} of {total:,} bytes"
        )


def read_secondary_rows(csv_path):
    """Read the state-funded secondary rows from the census (cp1252-encoded)."""
    rows = []
    with open(csv_path, newline="", encoding="cp1252") as fh:
        for row in csv.DictReader(fh):
            if (row.get(PHASE_COL) or "").strip() == SECONDARY_PHASE:
                rows.append(row)
    return rows


# --- Times -> census name matching -------------------------------------------

_MATCH_STOP = {
    "the", "a", "an", "and", "of", "for", "academy", "trust", "sixth", "form",
    "co", "operative", "foundation",
}
_RELIGIOUS = [
    (" roman catholic ", " catholic "), (" church of england ", " ce "),
    (" c of e ", " ce "), (" cofe ", " ce "), (" r c ", " catholic "),
    (" rc ", " catholic "),
]


def _match_base(s):
    s = unicodedata.normalize("NFKD", str(s))
    for a, b in [("’", "'"), ("‘", "'"), ("–", "-"), ("—", "-")]:
        s = s.replace(a, b)
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = " " + s + " "
    for a, b in _RELIGIOUS:
        s = s.replace(a, b)
    return s.strip()


def _match_tokens(s):
    out = set()
    for w in _match_base(s).split():
        if w == "saint":
            w = "st"
        if w not in _MATCH_STOP:
            out.add(w)
    return out


def _jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def build_cohort(sec_rows, times_csv, top_n, threshold):
    """Match Times state secondaries to census URNs, in rank order.

    top_n is the number of *matched* schools wanted: the Times table is walked
    from rank 1 and matching continues until top_n schools have matched (so the
    unmatched Northern Irish / Welsh schools encountered along the way do not eat
    into the count). top_n falsy (0/None) means match the whole table.

    Returns (schools_map {urn: Times label}, ranks {urn: rank}, unmatched list).
    The unmatched list holds only the rows passed over within the span consumed.
    """
    for r in sec_rows:
        r["_tok"] = _match_tokens(r.get(NAME_COL, ""))
        r["_loc"] = " ".join(_match_base(r.get(c, "")) for c in
                             (LA_COL, DISTRICT_COL, NAME_COL))

    with open(times_csv, encoding="utf-8", newline="") as f:
        cohort = list(csv.DictReader(f))

    schools, ranks, unmatched, used = {}, {}, [], set()
    for t in cohort:
        name, loc = t.get("School", ""), _match_base(t.get("Location", ""))
        loc_words = [w for w in loc.split() if len(w) > 3]
        tt = _match_tokens(name)
        best, best_sc = None, 0.0
        for r in sec_rows:
            sc = _jaccard(tt, r["_tok"])
            if sc == 0:
                continue
            if loc_words and any(w in r["_loc"] for w in loc_words):
                sc += 0.2
            if sc > best_sc:
                best, best_sc = r, sc
        urn = (best.get(URN_COL) or "").strip() if best else ""
        if best and best_sc >= threshold and urn not in used:
            schools[urn] = name.replace("’", "'")
            ranks[urn] = (t.get("Rank") or "").strip()
            used.add(urn)
            if top_n and len(schools) >= top_n:
                break
        else:
            unmatched.append((t.get("Rank", ""), name, t.get("Location", "")))
    return schools, ranks, unmatched


def load_schools(sec_rows, schools_map, ranks):
    """Return {urn: data dict} for the cohort, from in-memory census rows."""
    found = {}
    for row in sec_rows:
        urn = (row.get(URN_COL) or "").strip()
        if urn not in schools_map:
            continue
        d = {k: fnum(row.get(col, "")) for k, col in PCT_COLS.items()}
        d["name"] = (row.get(NAME_COL) or "").strip().strip('"')
        d["N"] = (row.get(HEADCOUNT_COL) or "").strip()
        d["label"] = schools_map[urn]
        d["rank"] = ranks.get(urn, "")
        d["urn"] = urn
        # Derived measures
        d["minority"] = round(100 - d["WhiteBrit"], 1)
        d["AsianTot"] = round(
            d["Indian"] + d["Pak"] + d["Bang"] + d["OthAsian"] + d["Chinese"], 1
        )
        d["BlackTot"] = round(d["African"] + d["Carib"] + d["OthBlack"], 1)
        d["Mixed"] = round(d["WhAsian"] + d["WhBAfr"] + d["WhBCar"] + d["OthMixed"], 1)
        d["Other"] = d["OthEth"]
        found[urn] = d
    return found


def fmt(x):
    return f"{x:.1f}" if isinstance(x, float) else str(x)


def rank_key(r):
    """Sort key for a Times rank string like '6=' or '46='. Unranked sorts last."""
    m = re.match(r"\d+", str(r or ""))
    return int(m.group()) if m else 10**9


def build_report(schools, unmatched):
    rows = sorted(schools.values(), key=lambda d: rank_key(d["rank"]))

    out = []
    out.append("# Ethnic background of top English state secondary schools\n")
    out.append("*Cohort: the Times / Sunday Times Parent Power state-secondary league "
               "table, sorted by Parent Power rank (rank 1 first).*  ")
    out.append(f"*Source: {CENSUS_LABEL} (school-level underlying data, unsuppressed).*  ")
    out.append(f"*Generated: {date.today().isoformat()}. Whole-school headcount; "
               "percentages are of all pupils. \"% minority\" = 100 − % White British.*\n")

    # Summary table
    out.append("## Summary — ethnic minority share\n")
    out.append("| PP rank | School | Pupils | % minority | % White British | % EAL |")
    out.append("|--|--|--|--|--|--|")
    for d in rows:
        out.append(
            f"| {d['rank']} | {d['label']} | {d['N']} | {fmt(d['minority'])} | "
            f"{fmt(d['WhiteBrit'])} | {fmt(d['EAL'])} |"
        )
    out.append("")

    # Detailed group table
    out.append("## Group-by-group breakdown (% of pupils)\n")
    out.append('*"Mixed" = all mixed categories combined; "Other" = any other ethnic '
               "group. Small residual groups (White Irish, Gypsy/Roma, Traveller) are "
               "omitted from these columns but counted in % minority above.*\n")
    header = "| PP rank | School | " + " | ".join(lbl for _, lbl in DETAIL_GROUPS) + " |"
    out.append(header)
    out.append("|--" * (len(DETAIL_GROUPS) + 2) + "|")
    for d in rows:
        cells = " | ".join(fmt(d[key]) for key, _ in DETAIL_GROUPS)
        out.append(f"| {d['rank']} | {d['label']} | {cells} |")
    out.append("")

    # Auto-generated observation: largest single minority group per school
    out.append("## Largest single minority group, by school\n")
    out.append("| PP rank | School | Largest group | % |")
    out.append("|--|--|--|--|")
    group_labels = {
        "Indian": "Indian", "Pak": "Pakistani", "Bang": "Bangladeshi",
        "OthAsian": "Other Asian", "Chinese": "Chinese", "African": "Black African",
        "OthWhite": "Other White", "Mixed": "Mixed", "Other": "Other ethnic",
    }
    for d in rows:
        best_key = max(group_labels, key=lambda k: d.get(k, 0))
        out.append(f"| {d['rank']} | {d['label']} | {group_labels[best_key]} | "
                   f"{fmt(d[best_key])} |")
    out.append("")

    if unmatched:
        listed = ", ".join(
            f"{name} (#{rank})" for rank, name, _loc in
            sorted(unmatched, key=lambda u: rank_key(u[0]))
        )
        out.append(f"> **Not matched ({len(unmatched)}):** these Parent Power schools "
                   "were not found in the England census — almost all are in Northern "
                   "Ireland or Wales, which the England-only census does not cover: "
                   + listed + "\n")

    out.append("---")
    out.append("*Independent schools are excluded by design: they submit an aggregate "
               "census with no ethnicity breakdown, so per-school data does not exist "
               "for them.*")
    return "\n".join(out)


# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Path to a local school-level CSV (skips download).")
    ap.add_argument("--cache", default=DEFAULT_CACHE,
                    help=f"Where to cache the downloaded CSV (default {DEFAULT_CACHE}).")
    ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                    help=f"Output Markdown path (default {DEFAULT_OUTPUT}).")
    ap.add_argument("--times-csv", default=TIMES_SECONDARY_CSV,
                    help=f"Parent Power state-secondary CSV (default {TIMES_SECONDARY_CSV}).")
    ap.add_argument("--top", type=int, default=0,
                    help="Report exactly N matched schools, over-fetching down the "
                         "Times table to skip unmatched ones (0 = whole table).")
    ap.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = ap.parse_args()

    csv_path = args.csv or args.cache
    if not args.csv and (args.force or not os.path.exists(csv_path)):
        download(CSV_URL, csv_path)

    if not os.path.exists(csv_path):
        sys.exit(f"CSV not found: {csv_path}")
    if not os.path.exists(args.times_csv):
        sys.exit(f"Times table not found: {args.times_csv}\n"
                 "Generate it with: python3 extract_times_table.py "
                 "the-top-state-secondary-schools " + TIMES_SECONDARY_CSV)

    sec_rows = read_secondary_rows(csv_path)
    top_n = args.top or COHORT_SIZE
    schools_map, ranks, unmatched = build_cohort(
        sec_rows, args.times_csv, top_n, MATCH_THRESHOLD)
    if not schools_map:
        sys.exit("No schools matched — check the Times CSV / census file format.")

    schools = load_schools(sec_rows, schools_map, ranks)
    report = build_report(schools, unmatched)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    total = len(schools) + len(unmatched)
    print(f"Wrote {args.output} ({len(schools)}/{total} Times schools matched to "
          f"census; {len(unmatched)} unmatched).", file=sys.stderr)


if __name__ == "__main__":
    main()
