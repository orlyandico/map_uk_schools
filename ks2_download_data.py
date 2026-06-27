"""
Download KS2 performance data and GIAS school information.

Data sources:
  - KS2 school-level attainment: Explore Education Statistics (EES) content API
  - School addresses/metadata: Get Information About Schools (GIAS)

Uses httpx with browser-like headers (User-Agent + Referer) to download from
government sites that reject plain requests. No extra dependencies needed
beyond httpx (already in the base project requirements).

Additional dependency:
    pip install httpx

Run once before ks2_generate_school_data.py.
"""

import argparse
import io
import os
import random
import re
import sys
import time
import zipfile

import httpx

import ees_download_lib as ees

EES_PUBLICATION_SLUG = "key-stage-2-attainment"

# GIAS direct download base (daily extracts, date-stamped)
GIAS_DOWNLOAD_BASE = (
    "https://ea-edubase-api-prod.azurewebsites.net"
    "/edubase/downloads/public/edubasealldata{date}.csv"
)

# Filename pattern to identify the school-level performance CSV inside release ZIPs
SCHOOL_PERF_PATTERN = re.compile(
    r"(institution|school).*(performance|attainment).*\.csv$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------

def _extract_school_perf_csv(zip_bytes: bytes, dest_path: str):
    """
    Extract the school-level performance CSV from a release ZIP and save it.

    Selects the CSV that carries a per-school URN column. Writes to a .tmp file
    first; renames to dest_path only on success.

    Returns:
        True  — a school-level CSV was saved.
        None  — the release has no school-level file (expected for pre-2022/23,
                where DfE publishes only aggregated KS2 data, not per-school).
        False — a genuine error (no CSVs in the ZIP, or an extraction failure).
    """
    tmp_path = dest_path + ".tmp"
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                print("  No CSVs found in ZIP.", file=sys.stderr)
                return False

            def has_urn(name):
                """True if the CSV's header carries a per-school URN column."""
                with zf.open(name) as f:
                    header = f.readline().decode("utf-8", errors="replace")
                cols = [c.strip().strip('"') for c in header.split(",")]
                return any(c in cols for c in ("school_urn", "urn", "URN"))

            # The school-level file is the one with a URN column. Confirm by
            # header rather than picking the largest CSV, so we never grab an
            # aggregated (national/LA/regional) file that happens to be bigger.
            # A name hint just decides which order we probe in.
            hinted = [n for n in csv_names if SCHOOL_PERF_PATTERN.search(n)]
            ordered = hinted + [n for n in csv_names if n not in hinted]
            chosen = next((n for n in ordered if has_urn(n)), None)
            if chosen is None:
                print(
                    "  No school-level (per-URN) file in this release — EES only "
                    "publishes school-level KS2 attainment from 2022/23 onwards.",
                    file=sys.stderr,
                )
                return None

            print(f"  Extracting: {chosen}")
            with zf.open(chosen) as src:
                raw = src.read()
            with open(tmp_path, "wb") as dst:
                dst.write(raw)
        os.replace(tmp_path, dest_path)
        print(f"  Saved to {dest_path} ({os.path.getsize(dest_path) / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ZIP extraction error: {e}", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def download_ees_year(client: httpx.Client, year_title: str,
                      dest_path: str, releases: list[dict]):
    """
    Download the school-level performance CSV for a specific academic year.

    year_title examples: "2023-24", "2022-23", "2021-22"
    Normalises dash/slash so "2023-24" matches yearTitle "2023/24".
    When multiple releases exist for the same year (e.g. provisional +
    revised), prefers "revised".

    Returns the tri-state of _extract_school_perf_csv (True saved / None no
    school-level file in the release / False genuine error), or False if the
    year has no EES release at all or the ZIP could not be downloaded.
    """
    matched = ees.pick_release(year_title, releases)
    if not matched:
        return False

    release_id = matched["id"]  # use "id", not "releaseId"
    year_display = matched.get("yearTitle") or matched.get("title")
    print(f"  Matched release: {year_display} — {matched.get('slug')} (id={release_id})")

    zip_bytes = ees.download_release_zip(client, release_id)
    if not zip_bytes:
        return False

    return _extract_school_perf_csv(zip_bytes, dest_path)


# ---------------------------------------------------------------------------
# GIAS download
# ---------------------------------------------------------------------------

def download_gias(dest_path: str) -> bool:
    """
    Download the GIAS all-establishments CSV.

    The server rejects HEAD requests (returns 500) but serves GET correctly,
    so we stream-download directly without probing first.
    A Referer header pointing at the GIAS site is required.
    """
    import datetime

    today = datetime.date.today()
    client_headers = {
        **ees.BROWSER_HEADERS,
        "Referer": "https://get-information-schools.service.gov.uk/",
    }

    tmp_path = dest_path + ".tmp"
    with httpx.Client(headers=client_headers, timeout=180.0, follow_redirects=True) as client:
        for delta in range(60):
            date_str = (today - datetime.timedelta(days=delta)).strftime("%Y%m%d")
            url = GIAS_DOWNLOAD_BASE.format(date=date_str)
            try:
                with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        continue  # try the next date
                    total = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    print(f"  Downloading GIAS ({date_str})...")
                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                print(
                                    f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB "
                                    f"({downloaded / total * 100:.0f}%)",
                                    end="", flush=True,
                                )
                os.replace(tmp_path, dest_path)
                print(f"\n  Saved to {dest_path} ({os.path.getsize(dest_path) / 1e6:.1f} MB)")
                return True
            except Exception as e:
                print(f"  {date_str}: {e}", file=sys.stderr)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                continue

    print()
    print("  *** GIAS automatic download failed. Manual steps: ***")
    print("  1. Go to: https://get-information-schools.service.gov.uk/Downloads")
    print("  2. Download 'All establishments' → CSV format")
    print(f"  3. Save as: {dest_path}")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download KS2 performance data (EES) and GIAS school list",
        epilog=(
            "Extra dependencies: pip install httpx\n\n"
            "Data sources:\n"
            "  EES: explore-education-statistics.service.gov.uk\n"
            "  GIAS: get-information-schools.service.gov.uk"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2023-24"],
        metavar="YEAR",
        help="Academic years to download (default: 2023-24). The latest release "
             "embeds prior years' school-level rows, so one year is usually "
             "enough. Pre-2022/23 years have no school-level data and are skipped.",
    )
    parser.add_argument(
        "--gias-file",
        default="gias_establishments.csv",
        help="Output path for GIAS establishments CSV (default: gias_establishments.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write downloaded files (default: current directory)",
    )
    parser.add_argument(
        "--skip-ees", action="store_true",
        help="Skip EES download (if files already present)",
    )
    parser.add_argument(
        "--skip-gias", action="store_true",
        help="Skip GIAS download (if already present)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ees.clean_stale_tmp(args.output_dir)

    results = {}
    skipped = []  # years with no school-level data — expected, not a failure

    # --- EES downloads ---
    if not args.skip_ees:
        print(f"Fetching EES release list for '{EES_PUBLICATION_SLUG}'...")
        with ees.make_client() as client:
            releases = ees.get_ees_releases(client, EES_PUBLICATION_SLUG)

        if not releases:
            print("Could not retrieve EES release list. Check connectivity.")
            for year in args.years:
                results[f"ees_{year}"] = False
        else:
            for year in args.years:
                # Output filename: e.g. ks2_school_attainment_2324.csv
                safe_year = year.replace("-", "").replace("/", "")
                dest = os.path.join(args.output_dir, f"ks2_school_attainment_{safe_year}.csv")

                if os.path.exists(dest) and os.path.getsize(dest) > 1024:
                    print(f"Already exists: {dest}")
                    results[f"ees_{year}"] = True
                    continue
                elif os.path.exists(dest):
                    print(f"Found incomplete file (too small), re-downloading: {dest}")

                print(f"\nDownloading KS2 data for {year}...")
                with ees.make_client() as client:
                    status = download_ees_year(client, year, dest, releases)

                if status is None:
                    # Release exists but has no school-level file: expected for
                    # pre-2022/23. Don't count it as a failure or print manual
                    # steps — there is nothing to download by hand.
                    print(f"  Skipping {year}: no school-level data published "
                          f"(the latest release already covers prior years).")
                    skipped.append(f"ees_{year}")
                else:
                    results[f"ees_{year}"] = status
                    if not status:
                        print(f"\n  *** Manual download for {year}: ***")
                        print(f"  1. Go to: https://explore-education-statistics.service.gov.uk"
                              f"/find-statistics/key-stage-2-attainment/{year}")
                        print(f"  2. Click 'Download all data (ZIP)'")
                        print(f"  3. Extract the school-level performance CSV")
                        print(f"  4. Save as: {dest}")

                time.sleep(random.uniform(1, 2))  # polite delay between releases
    else:
        for year in args.years:
            safe_year = year.replace("-", "").replace("/", "")
            dest = os.path.join(args.output_dir, f"ks2_school_attainment_{safe_year}.csv")
            results[f"ees_{year}"] = os.path.exists(dest)

    # --- GIAS download ---
    gias_dest = os.path.join(args.output_dir, args.gias_file)
    if not args.skip_gias:
        if os.path.exists(gias_dest) and os.path.getsize(gias_dest) > 1024:
            print(f"\nAlready exists: {gias_dest}")
            results["gias"] = True
        else:
            if os.path.exists(gias_dest):
                print(f"\nFound incomplete GIAS file (too small), re-downloading.")
            print("\nDownloading GIAS establishments...")
            results["gias"] = download_gias(gias_dest)
    else:
        results["gias"] = os.path.exists(gias_dest)

    # --- Summary ---
    all_ok = ees.print_summary(results)
    for key in skipped:
        print(f"  {key}: SKIPPED (no school-level data for this year)")
    if all_ok:
        print("\nAll files ready. Next step:")
        print("  python3 ks2_generate_school_data.py")
    else:
        print("\nSome files need manual download. See instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
