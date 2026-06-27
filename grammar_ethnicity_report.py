#!/usr/bin/env python3
"""
Generate a per-school ethnicity report for selected English state grammar schools
from the DfE "Schools, pupils and their characteristics" school-level census file.

Data source: DfE, Schools, pupils and their characteristics (school-level underlying
data, one row per school, unsuppressed). The default URL points at the Jan 2025
(2024/25) release. Update CSV_URL when a new release is published.

Usage:
    python3 grammar_ethnicity_report.py                 # download + generate the report
    python3 grammar_ethnicity_report.py --csv data.csv  # use a local CSV instead
    python3 grammar_ethnicity_report.py -o out.md        # choose output path

Note: only state-funded schools appear in this census. Independent schools submit an
aggregate return with no ethnicity breakdown, so they cannot be reported this way.
"""

import argparse
import csv
import os
import sys
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

# Schools to report: URN -> display label. Edit this list to change the cohort.
SCHOOLS = {
    "136290": "Queen Elizabeth's, Barnet",
    "138051": "Henrietta Barnett",
    "136621": "Wilson's",
    "137044": "KE Camp Hill Girls (B'ham)",
    "136795": "Nonsuch HS Girls",
    "137045": "KE Camp Hill Boys (B'ham)",
    "136789": "Wallington HS Girls",
    "101676": "St Olave's",
    "136448": "Kendrick (Reading)",
    "136615": "Tiffin Girls'",
    "136798": "Wallington County GS",
    "136449": "Reading School",
    "136551": "Newstead Wood",
    "136787": "Sutton GS",
    "102055": "Latymer, Edmonton",
    "137289": "Altrincham GS Girls",
    "136458": "Altrincham GS Boys",
    "136910": "Tiffin",
    "136412": "Chelmsford County HS",
    "149899": "Colchester Royal GS",
    "136353": "Pate's (Cheltenham)",
    "118843": "Judd (Tonbridge)",
    "140595": "Skinners' (T. Wells)",
    "136366": "Colyton (Devon)",
    "137739": "Cranbrook (Kent)",
    "136501": "Sir Roger Manwood's (Kent)",
}

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
    print(f"Downloading census file -> {dest} ...", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"  saved {os.path.getsize(dest):,} bytes", file=sys.stderr)


def load_schools(csv_path):
    """Return {urn: data dict} for the configured schools."""
    found = {}
    # DfE file is Windows-1252 encoded.
    with open(csv_path, newline="", encoding="cp1252") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            urn = (row.get(URN_COL) or "").strip()
            if urn not in SCHOOLS:
                continue
            d = {k: fnum(row.get(col, "")) for k, col in PCT_COLS.items()}
            d["name"] = (row.get(NAME_COL) or "").strip().strip('"')
            d["N"] = (row.get(HEADCOUNT_COL) or "").strip()
            d["label"] = SCHOOLS[urn]
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


def build_report(schools):
    rows = sorted(schools.values(), key=lambda d: -d["minority"])
    missing = [u for u in SCHOOLS if u not in schools]

    out = []
    out.append("# Ethnic background of top English state grammar schools\n")
    out.append(f"*Source: {CENSUS_LABEL} (school-level underlying data, unsuppressed).*  ")
    out.append(f"*Generated: {date.today().isoformat()}. Whole-school headcount; "
               "percentages are of all pupils. \"% minority\" = 100 − % White British.*\n")

    # Summary table
    out.append("## Summary — ethnic minority share\n")
    out.append("| School | Pupils | % minority | % White British | % EAL |")
    out.append("|--|--|--|--|--|")
    for d in rows:
        out.append(
            f"| {d['label']} | {d['N']} | {fmt(d['minority'])} | "
            f"{fmt(d['WhiteBrit'])} | {fmt(d['EAL'])} |"
        )
    out.append("")

    # Detailed group table
    out.append("## Group-by-group breakdown (% of pupils)\n")
    out.append('*"Mixed" = all mixed categories combined; "Other" = any other ethnic '
               "group. Small residual groups (White Irish, Gypsy/Roma, Traveller) are "
               "omitted from these columns but counted in % minority above.*\n")
    header = "| School | " + " | ".join(lbl for _, lbl in DETAIL_GROUPS) + " |"
    out.append(header)
    out.append("|--" * (len(DETAIL_GROUPS) + 1) + "|")
    for d in rows:
        cells = " | ".join(fmt(d[key]) for key, _ in DETAIL_GROUPS)
        out.append(f"| {d['label']} | {cells} |")
    out.append("")

    # Auto-generated observation: largest single minority group per school
    out.append("## Largest single minority group, by school\n")
    out.append("| School | Largest group | % |")
    out.append("|--|--|--|")
    group_labels = {
        "Indian": "Indian", "Pak": "Pakistani", "Bang": "Bangladeshi",
        "OthAsian": "Other Asian", "Chinese": "Chinese", "African": "Black African",
        "OthWhite": "Other White", "Mixed": "Mixed", "Other": "Other ethnic",
    }
    for d in rows:
        best_key = max(group_labels, key=lambda k: d.get(k, 0))
        out.append(f"| {d['label']} | {group_labels[best_key]} | {fmt(d[best_key])} |")
    out.append("")

    if missing:
        out.append("> **Note:** the following URNs were not found in the census file "
                   "(check the URN or that the school is state-funded): "
                   + ", ".join(missing) + "\n")

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
    ap.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = ap.parse_args()

    csv_path = args.csv or args.cache
    if not args.csv and (args.force or not os.path.exists(csv_path)):
        download(CSV_URL, csv_path)

    if not os.path.exists(csv_path):
        sys.exit(f"CSV not found: {csv_path}")

    schools = load_schools(csv_path)
    if not schools:
        sys.exit("No matching schools found — check URNs / file format.")

    report = build_report(schools)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"Wrote {args.output} ({len(schools)}/{len(SCHOOLS)} schools matched).",
          file=sys.stderr)


if __name__ == "__main__":
    main()
