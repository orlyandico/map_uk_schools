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
import json
import os
import random
import re
import sys
import time
import zipfile

import httpx

# EES content API — stable REST endpoints, no JS rendering needed
EES_CONTENT_API = "https://content.explore-education-statistics.service.gov.uk/api"
EES_PUBLICATION_SLUG = "key-stage-2-attainment"

# GIAS direct download base (daily extracts, date-stamped)
GIAS_DOWNLOAD_BASE = (
    "https://ea-edubase-api-prod.azurewebsites.net"
    "/edubase/downloads/public/edubasealldata{date}.csv"
)
GIAS_ALT_URL = (
    "https://get-information-schools.service.gov.uk"
    "/Downloads/GetDownload/edubasealldata"
)

# Browser-like headers that avoid trivial bot-detection
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

# Target academic years we want (time_period codes as they appear in EES)
TARGET_YEARS = {"2021-22", "2022-23", "2023-24"}

# Filename pattern to identify the school-level performance CSV inside release ZIPs
SCHOOL_PERF_PATTERN = re.compile(
    r"(institution|school).*(performance|attainment).*\.csv$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# HTTP helpers — adapted from adzuna_client.py
# ---------------------------------------------------------------------------

def _make_client():
    """Return an httpx client with browser-like headers and generous timeout."""
    return httpx.Client(
        headers=BROWSER_HEADERS,
        timeout=120.0,
        follow_redirects=True,
    )



def _download_stream(client: httpx.Client, url: str, dest_path: str,
                     description: str, max_retries: int = 3) -> bool:
    """Stream-download a file to disk with retries and progress output."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {description} (attempt {attempt})...")
            print(f"  URL: {url}")
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(
                                f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB "
                                f"({downloaded / total * 100:.0f}%)",
                                end="", flush=True,
                            )
            print(f"\n  Saved to {dest_path} ({os.path.getsize(dest_path) / 1e6:.1f} MB)")
            return True
        except httpx.HTTPStatusError as e:
            print(f"\n  HTTP {e.response.status_code}")
        except Exception as e:
            print(f"\n  Error: {e}")

        if attempt < max_retries:
            delay = random.uniform(2, 5)
            print(f"  Retrying in {delay:.1f}s...")
            time.sleep(delay)

    return False


# ---------------------------------------------------------------------------
# EES content API helpers
# ---------------------------------------------------------------------------

def _ees_get(client: httpx.Client, path: str) -> dict | None:
    """GET a JSON endpoint from the EES content API."""
    url = f"{EES_CONTENT_API}/{path.lstrip('/')}"
    try:
        resp = client.get(url, headers={**BROWSER_HEADERS, "Accept": "application/json"})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  EES API error ({path}): {e}", file=sys.stderr)
        return None


def get_ees_releases(client: httpx.Client) -> list[dict]:
    """
    Return all releases for the KS2 attainment publication.

    Uses /publications/{slug}/releases (not /publications/{slug} which
    returns an empty releases list).  Prefers the "revised" release when
    multiple releases exist for the same year.
    """
    data = _ees_get(client, f"publications/{EES_PUBLICATION_SLUG}/releases")
    if not data:
        return []
    releases = data if isinstance(data, list) else data.get("releases", [])
    print(f"  Found {len(releases)} releases for '{EES_PUBLICATION_SLUG}'")
    for r in releases:
        print(f"    {r.get('yearTitle')} — {r.get('slug')} (id={r.get('id')})")
    return releases


def get_release_zip_url(release_id: str) -> str:
    """Return the ZIP download URL for a given release id (not releaseId)."""
    return f"{EES_CONTENT_API}/releases/{release_id}/files"


def _extract_school_perf_csv(zip_bytes: bytes, dest_path: str) -> bool:
    """
    Extract the school-level performance CSV from a release ZIP and save it.

    Tries to match files by pattern; falls back to the largest CSV.
    Writes to a .tmp file first; renames to dest_path only on success.
    """
    tmp_path = dest_path + ".tmp"
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            # Prefer files matching our pattern
            candidates = [n for n in names if SCHOOL_PERF_PATTERN.search(n)]
            if not candidates:
                # Fall back to largest CSV
                csv_names = [n for n in names if n.endswith(".csv")]
                if not csv_names:
                    print("  No CSVs found in ZIP.", file=sys.stderr)
                    return False
                candidates = [
                    max(csv_names, key=lambda n: zf.getinfo(n).file_size)
                ]
            chosen = candidates[0]
            print(f"  Extracting: {chosen}")
            with zf.open(chosen) as src, open(tmp_path, "wb") as dst:
                dst.write(src.read())
        os.replace(tmp_path, dest_path)
        print(f"  Saved to {dest_path} ({os.path.getsize(dest_path) / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ZIP extraction error: {e}", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def download_ees_year(client: httpx.Client, year_title: str,
                      dest_path: str, releases: list[dict]) -> bool:
    """
    Download the school-level performance CSV for a specific academic year.

    year_title examples: "2023-24", "2022-23", "2021-22"
    Normalises dash/slash so "2023-24" matches yearTitle "2023/24".
    When multiple releases exist for the same year (e.g. provisional +
    revised), prefers "revised".
    """
    normalised = year_title.replace("-", "/")  # "2023-24" → "2023/24"

    candidates = [
        r for r in releases
        if (r.get("yearTitle") or "").replace("-", "/") == normalised
    ]

    if not candidates:
        print(f"  No release found for year '{year_title}'.")
        print(f"  Available: {[r.get('yearTitle') for r in releases]}")
        return False

    # Prefer "revised" over "provisional" over anything else
    def preference(r):
        slug = r.get("slug", "")
        if "revised" in slug:
            return 0
        if "provisional" in slug:
            return 2
        return 1

    matched = sorted(candidates, key=preference)[0]
    release_id = matched["id"]  # use "id", not "releaseId"
    year_display = matched.get("yearTitle") or matched.get("title")
    print(f"  Matched release: {year_display} — {matched.get('slug')} (id={release_id})")

    zip_url = get_release_zip_url(release_id)

    # Download ZIP into memory (these are typically 5-30 MB)
    print(f"  Downloading release ZIP...")
    try:
        resp = client.get(zip_url, headers={**BROWSER_HEADERS, "Accept": "application/zip"})
        resp.raise_for_status()
        zip_bytes = resp.content
        print(f"  ZIP size: {len(zip_bytes) / 1e6:.1f} MB")
    except Exception as e:
        print(f"  Failed to download ZIP: {e}")
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
        **BROWSER_HEADERS,
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
        default=["2023-24", "2022-23", "2021-22"],
        metavar="YEAR",
        help="Academic years to download, e.g. 2023-24 2022-23 (default: last 3 years)",
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

    # Remove any stale .tmp files left by interrupted previous runs
    for fname in os.listdir(args.output_dir):
        if fname.endswith(".tmp"):
            tmp = os.path.join(args.output_dir, fname)
            print(f"Removing stale temp file: {tmp}")
            os.remove(tmp)

    results = {}

    # --- EES downloads ---
    if not args.skip_ees:
        print(f"Fetching EES release list for '{EES_PUBLICATION_SLUG}'...")
        with _make_client() as client:
            releases = get_ees_releases(client)

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
                with _make_client() as client:
                    ok = download_ees_year(client, year, dest, releases)
                results[f"ees_{year}"] = ok

                if not ok:
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
    print("\n" + "=" * 50)
    print("Download summary:")
    all_ok = True
    for key, ok in results.items():
        status = "OK" if ok else "FAILED (manual download required)"
        print(f"  {key}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll files ready. Next step:")
        print("  python3 ks2_generate_school_data.py")
    else:
        print("\nSome files need manual download. See instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
