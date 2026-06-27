"""
Download KS5 (16-18 A-level) school performance data from Explore Education Statistics.

Data source: DfE "A level and other 16 to 18 results" publication on EES
  https://explore-education-statistics.service.gov.uk/find-statistics/a-level-and-other-16-to-18-results

EES availability: the school-level summary file (publication_file_rectype_13.csv)
was introduced in the 2023/24 release.  Earlier years (2022/23, 2021/22, …) do not
include this file in their EES ZIPs, so automatic download is not possible for them.
For those years, download manually from:
  https://www.compare-school-performance.service.gov.uk/download-data/
  Select year → "All of England" → "16-18 results (final)" → CSV, then gzip.

For each supported academic year, this script downloads the EES release ZIP, joins
the two institution-level files, and saves a gzip-compressed CSV matching the column
names expected by school_data_lib.py:
  {year}_england_ks5final.csv.gz

Column mapping from EES → school_data_lib.py:
  school_name            → SCHNAME
  address                → ADDRESS1
  lad_name               → TOWN
  postcode               → PCODE
  phone_number           → TELNUM
  admissions_policy      → ADMPOL_PT
  sex_policy             → GEND1618
  age_range              → AGERANGE
  cohort                 → TPUP1618
  points_per_entry       → TALLPPE_ALEV_1618
  best_three_alevels_ppe → TB3PTSE

Run once before generate_school_data.py.
"""

import argparse
import gzip
import io
import os
import random
import re
import sys
import time
import zipfile

import httpx

import ees_download_lib as ees

try:
    import pandas as pd
except ImportError:
    print("pandas is required: pip install pandas", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EES_PUBLICATION_SLUG = "a-level-and-other-16-to-18-results"

# Files inside each release ZIP that we need
PERF_FILE = "data/publication_file_rectype_13.csv"   # school-level metrics
INFO_FILE = "data/inst_information.csv"               # addresses / metadata

# Rows to keep from the performance file
DISADV_FILTER = "All students"

# Column rename map: EES name → school_data_lib.py name
COLUMN_MAP = {
    "school_name": "SCHNAME",
    "address": "ADDRESS1",
    "lad_name": "TOWN",
    "postcode": "PCODE",
    "phone_number": "TELNUM",
    "admissions_policy": "ADMPOL_PT",
    "sex_policy": "GEND1618",
    "age_range": "AGERANGE",
    "cohort": "TPUP1618",
    "points_per_entry": "TALLPPE_ALEV_1618",
    "best_three_alevels_ppe": "TB3PTSE",
}

MIN_FILE_BYTES = 10_000  # reject files smaller than this as corrupt


# ---------------------------------------------------------------------------
# Data transformation
# ---------------------------------------------------------------------------

def _transform_zip(zip_bytes: bytes, year_long: str) -> pd.DataFrame | None:
    """
    Extract performance + info CSVs from the ZIP, join, and rename columns.
    Returns a DataFrame with school_data_lib.py-compatible column names.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()

            # Locate files (names may vary slightly across releases)
            def find(pattern):
                matches = [n for n in names if re.search(pattern, n, re.I)]
                return matches[0] if matches else None

            perf_name = find(r"publication_file_rectype_13") or find(r"inst_data")
            info_name = find(r"inst_information")

            if not perf_name:
                print(f"  Could not find performance file in ZIP", file=sys.stderr)
                print(f"  ZIP contents: {names[:20]}", file=sys.stderr)
                return None
            if not info_name:
                print(f"  Could not find inst_information in ZIP", file=sys.stderr)
                return None

            print(f"  Reading: {perf_name}")
            with zf.open(perf_name) as f:
                perf_df = pd.read_csv(f, low_memory=False, dtype=str)

            print(f"  Reading: {info_name}")
            with zf.open(info_name) as f:
                info_df = pd.read_csv(f, low_memory=False, dtype=str)

    except Exception as e:
        print(f"  ZIP error: {e}", file=sys.stderr)
        return None

    # Strip whitespace from column names
    perf_df.columns = perf_df.columns.str.strip().str.strip('"')
    info_df.columns = info_df.columns.str.strip().str.strip('"')

    # Filter performance to "All students" rows only
    if "disadvantaged_status" in perf_df.columns:
        perf_df = perf_df[perf_df["disadvantaged_status"].str.strip() == DISADV_FILTER]

    # Keep only the columns we need from each file
    perf_cols = ["school_urn", "cohort", "points_per_entry", "best_three_alevels_ppe"]
    info_cols = [
        "school_urn", "school_name", "address", "postcode", "phone_number",
        "admissions_policy", "sex_policy", "age_range", "establishment_type",
        "lad_name",
    ]
    perf_df = perf_df[[c for c in perf_cols if c in perf_df.columns]].copy()
    info_df = info_df[[c for c in info_cols if c in info_df.columns]].copy()

    # Drop duplicates in info (it can have multiple time_period rows per school)
    info_df = info_df.drop_duplicates(subset=["school_urn"])

    # Join
    df = perf_df.merge(info_df, on="school_urn", how="left")

    # Rename to school_data_lib.py column names
    df = df.rename(columns=COLUMN_MAP)

    print(f"  Joined data: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Save as gzip CSV
# ---------------------------------------------------------------------------

def _save_gzip(df: pd.DataFrame, dest_path: str) -> bool:
    tmp_path = dest_path + ".tmp"
    try:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        with gzip.open(tmp_path, "wb") as f:
            f.write(csv_bytes)
        os.replace(tmp_path, dest_path)
        print(f"  Saved {dest_path} ({os.path.getsize(dest_path) / 1e6:.1f} MB compressed, "
              f"{len(df):,} schools)")
        return True
    except Exception as e:
        print(f"  Save error: {e}", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


# ---------------------------------------------------------------------------
# Per-year download orchestration
# ---------------------------------------------------------------------------

def download_ks5_year(client: httpx.Client, year_long: str,
                      dest_path: str, releases: list[dict]) -> bool:
    """Download and process one academic year's data."""
    release = ees.pick_release(year_long, releases)
    if not release:
        return False

    release_id = release["id"]
    print(f"  Release: {release.get('yearTitle')} — {release.get('slug')} (id={release_id})")

    zip_bytes = ees.download_release_zip(client, release_id)
    if not zip_bytes:
        return False

    df = _transform_zip(zip_bytes, year_long)
    if df is None or len(df) == 0:
        return False

    return _save_gzip(df, dest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download KS5 16-18 A-level results from EES",
        epilog=(
            "Extra dependencies: pip install httpx pandas\n\n"
            "EES availability:\n"
            "  School-level data is available from EES for 2023-2024 onwards.\n"
            "  Earlier years require manual download from:\n"
            "    https://www.compare-school-performance.service.gov.uk/download-data/\n"
            "  Select year → 'All of England' → '16-18 results (final)' → CSV, then gzip.\n\n"
            "Data source:\n"
            f"  https://explore-education-statistics.service.gov.uk"
            f"/find-statistics/{EES_PUBLICATION_SLUG}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2023-2024", "2022-2023", "2021-2022"],
        metavar="YEAR",
        help="Academic years to download, e.g. 2023-2024 (default: last 3 years)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write files (default: current directory)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ees.clean_stale_tmp(args.output_dir)

    results = {}

    print(f"Fetching EES release list for '{EES_PUBLICATION_SLUG}'...")
    with ees.make_client(timeout=180.0) as client:
        releases = ees.get_ees_releases(client, EES_PUBLICATION_SLUG)

    if not releases:
        print("Could not retrieve EES release list. Check connectivity.")
        sys.exit(1)

    with ees.make_client(timeout=180.0) as client:
        for year in args.years:
            dest = os.path.join(args.output_dir, f"{year}_england_ks5final.csv.gz")

            if os.path.exists(dest) and os.path.getsize(dest) > MIN_FILE_BYTES:
                print(f"Already exists: {dest}")
                results[year] = True
                continue
            elif os.path.exists(dest):
                print(f"Found incomplete file, re-downloading: {dest}")

            print(f"\nDownloading KS5 data for {year}...")
            ok = download_ks5_year(client, year, dest, releases)
            results[year] = ok

            if not ok:
                print(f"\n  *** Manual download for {year}: ***")
                print(f"  EES does not publish school-level data for years before 2023-24.")
                print(f"  1. Go to: https://www.compare-school-performance.service.gov.uk/download-data/")
                print(f"  2. Select year {year}, 'All of England', '16-18 results (final)', CSV")
                print(f"  3. Gzip the downloaded file:")
                print(f"       gzip {year}_england_ks5final.csv")
                print(f"  4. Move to: {dest}")

            if year != args.years[-1]:
                time.sleep(random.uniform(1, 2))

    all_ok = ees.print_summary(results)
    if all_ok:
        print("\nAll files ready. Next steps:")
        print("  python3 generate_school_data.py")
        print("  python3 combined_create_standalone_app.py  (after the KS2 pipeline too)")
    else:
        print("\nSome files need manual download. See instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
