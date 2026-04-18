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

Run once before generate_school_data.py / plot_schools.py.
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

try:
    import pandas as pd
except ImportError:
    print("pandas is required: pip install pandas", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EES_CONTENT_API = "https://content.explore-education-statistics.service.gov.uk/api"
EES_PUBLICATION_SLUG = "a-level-and-other-16-to-18-results"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

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
# HTTP helpers
# ---------------------------------------------------------------------------

def _make_client() -> httpx.Client:
    return httpx.Client(
        headers=BROWSER_HEADERS,
        timeout=180.0,
        follow_redirects=True,
    )


def _ees_get(client: httpx.Client, path: str):
    url = f"{EES_CONTENT_API}/{path.lstrip('/')}"
    try:
        resp = client.get(url, headers={**BROWSER_HEADERS, "Accept": "application/json"})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  EES API error ({path}): {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# EES release discovery
# ---------------------------------------------------------------------------

def get_ees_releases(client: httpx.Client) -> list[dict]:
    """Return all releases for the 16-18 results publication."""
    data = _ees_get(client, f"publications/{EES_PUBLICATION_SLUG}/releases")
    if not data:
        return []
    releases = data if isinstance(data, list) else data.get("releases", [])
    print(f"  Found {len(releases)} releases for '{EES_PUBLICATION_SLUG}'")
    for r in releases:
        print(f"    {r.get('yearTitle')} — {r.get('slug')} (id={r.get('id')})")
    return releases


def _normalise_year(year_title: str) -> str:
    """Normalise any year format to the short 'YYYY/YY' form used by EES.

    Handles: '2023-2024', '2023/2024', '2023-24', '2023/24' → '2023/24'
    """
    s = year_title.replace("-", "/")
    parts = s.split("/")
    if len(parts) == 2 and len(parts[1]) == 4:
        return f"{parts[0]}/{parts[1][2:]}"
    return s


def _pick_release(year_title: str, releases: list[dict]) -> dict | None:
    """Select the best release for a given academic year string."""
    normalised = _normalise_year(year_title)
    candidates = [
        r for r in releases
        if _normalise_year(r.get("yearTitle") or "") == normalised
    ]
    if not candidates:
        print(f"  No release found for year '{year_title}'.")
        print(f"  Available: {[r.get('yearTitle') for r in releases]}")
        return None

    def preference(r):
        slug = r.get("slug", "")
        if "revised" in slug:
            return 0
        if "provisional" in slug:
            return 2
        return 1

    return sorted(candidates, key=preference)[0]


# ---------------------------------------------------------------------------
# ZIP download and data transformation
# ---------------------------------------------------------------------------

def _download_zip(client: httpx.Client, release_id: str) -> bytes | None:
    url = f"{EES_CONTENT_API}/releases/{release_id}/files"
    print(f"  Downloading release ZIP...")
    try:
        resp = client.get(url, headers={**BROWSER_HEADERS, "Accept": "application/zip"})
        resp.raise_for_status()
        data = resp.content
        print(f"  ZIP size: {len(data) / 1e6:.1f} MB")
        return data
    except Exception as e:
        print(f"  Failed to download ZIP: {e}")
        return None


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
    release = _pick_release(year_long, releases)
    if not release:
        return False

    release_id = release["id"]
    print(f"  Release: {release.get('yearTitle')} — {release.get('slug')} (id={release_id})")

    zip_bytes = _download_zip(client, release_id)
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

    # Remove stale .tmp files from interrupted previous runs
    for fname in os.listdir(args.output_dir):
        if fname.endswith(".tmp"):
            tmp = os.path.join(args.output_dir, fname)
            print(f"Removing stale temp file: {tmp}")
            os.remove(tmp)

    results = {}

    print(f"Fetching EES release list for '{EES_PUBLICATION_SLUG}'...")
    with _make_client() as client:
        releases = get_ees_releases(client)

    if not releases:
        print("Could not retrieve EES release list. Check connectivity.")
        sys.exit(1)

    with _make_client() as client:
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

    # Summary
    print("\n" + "=" * 50)
    print("Download summary:")
    all_ok = True
    for year, ok in results.items():
        status = "OK" if ok else "FAILED (manual download required)"
        print(f"  {year}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll files ready. Next steps:")
        print("  python3 generate_school_data.py")
        print("  python3 create_standalone_app.py")
    else:
        print("\nSome files need manual download. See instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
