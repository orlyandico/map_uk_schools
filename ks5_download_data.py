"""
Download KS5 (16-18 A-level) school performance data.

Data source: DfE Compare School Performance
  https://www.compare-school-performance.service.gov.uk/download-data/

Each year's file is downloaded as a plain CSV, then gzip-compressed and saved
with the naming convention expected by school_data_lib.py:
  {year}_england_ks5final.csv.gz

The site serves files via a parameterised download endpoint discovered by
parsing the download page.  Plain httpx with browser-like headers is
sufficient — no JS rendering needed.

Run once before generate_school_data.py.
"""

import argparse
import gzip
import os
import random
import re
import sys
import time

import httpx

# --- Constants ---

DOWNLOAD_BASE = "https://www.compare-school-performance.service.gov.uk"
DOWNLOAD_PAGE = f"{DOWNLOAD_BASE}/download-data/"

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
    "Referer": DOWNLOAD_BASE + "/",
}

# URL templates to try in order.  The site uses "year" as the first 4 digits
# of the academic year (e.g. 2023 for 2023-2024).
# We try several known patterns because the endpoint has changed over time.
DOWNLOAD_URL_TEMPLATES = [
    # Current format (observed 2024)
    (
        f"{DOWNLOAD_BASE}/download-data/download"
        "?fileType=csv&years={{short_year}}&type=ks5&schoolType=state-funded"
        "&phase=16-18-results-final&area=england"
    ),
    # Variant with "final" in type
    (
        f"{DOWNLOAD_BASE}/download-data/download"
        "?fileType=csv&years={{short_year}}&type=ks5final&area=england"
    ),
    # Older format
    (
        f"{DOWNLOAD_BASE}/download-data/download"
        "?downloadYear={{long_year}}&downloadType=ks5final"
        "&area=england&fileType=csv"
    ),
]

# Min size for a "real" CSV — reject anything smaller as a corrupt/error file
MIN_FILE_BYTES = 10_000  # 10 KB


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _make_client() -> httpx.Client:
    return httpx.Client(
        headers=BROWSER_HEADERS,
        timeout=180.0,
        follow_redirects=True,
    )


def _is_csv_response(resp: httpx.Response) -> bool:
    """Return True if the response looks like a real CSV file."""
    ct = resp.headers.get("content-type", "")
    cd = resp.headers.get("content-disposition", "")
    return (
        "text/csv" in ct
        or "application/csv" in ct
        or "octet-stream" in ct
        or ".csv" in cd
        or "attachment" in cd
    )


# ---------------------------------------------------------------------------
# Download page scraping — fallback URL discovery
# ---------------------------------------------------------------------------

def _scrape_download_links(client: httpx.Client, year_long: str) -> list[str]:
    """
    Fetch the download page and look for ks5 CSV download hrefs.

    Returns a list of candidate URLs (may be empty).
    """
    try:
        resp = client.get(DOWNLOAD_PAGE)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"  Could not fetch download page: {e}", file=sys.stderr)
        return []

    # Look for href values that contain year and ks5 hints
    short_year = year_long.split("-")[0]  # "2023-2024" → "2023"
    candidates = []

    # Patterns like href="...ks5...2023...csv..." or action="...download..."
    for m in re.finditer(r'href=["\']([^"\']*(?:ks5|16.18)[^"\']*\.csv[^"\']*)["\']',
                         html, re.IGNORECASE):
        url = m.group(1)
        if not url.startswith("http"):
            url = DOWNLOAD_BASE + ("" if url.startswith("/") else "/") + url
        candidates.append(url)

    # Also look for a generic download endpoint used with query params
    for m in re.finditer(
        r'href=["\']([^"\']*download-data/download[^"\']*)["\']', html, re.IGNORECASE
    ):
        url = m.group(1)
        if not url.startswith("http"):
            url = DOWNLOAD_BASE + url
        # Inject year into the URL if it isn't already there
        if short_year not in url and year_long not in url:
            sep = "&" if "?" in url else "?"
            url += f"{sep}years={short_year}&fileType=csv"
        candidates.append(url)

    return candidates


# ---------------------------------------------------------------------------
# Core download function for one year
# ---------------------------------------------------------------------------

def download_ks5_year(year_long: str, dest_path: str,
                      client: httpx.Client) -> bool:
    """
    Download the 16-18 final results CSV for one academic year and gzip it.

    year_long: "2023-2024", "2022-2023", "2021-2022"
    dest_path: destination .csv.gz path
    Returns True on success.
    """
    short_year = year_long.split("-")[0]  # "2023-2024" → "2023"
    tmp_csv = dest_path + ".csv.tmp"
    tmp_gz = dest_path + ".tmp"

    # Build candidate URL list: templates first, then scraped links
    candidate_urls = []
    for template in DOWNLOAD_URL_TEMPLATES:
        candidate_urls.append(
            template
            .replace("{short_year}", short_year)
            .replace("{long_year}", year_long)
        )
    candidate_urls += _scrape_download_links(client, year_long)

    for url in candidate_urls:
        print(f"  Trying: {url}")
        try:
            with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    print(f"    HTTP {resp.status_code} — skipping")
                    continue
                if not _is_csv_response(resp):
                    print(f"    Not a CSV response (Content-Type: "
                          f"{resp.headers.get('content-type', '?')}) — skipping")
                    continue

                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(tmp_csv, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(
                                f"\r    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB "
                                f"({downloaded / total * 100:.0f}%)",
                                end="", flush=True,
                            )
                print()

            if os.path.getsize(tmp_csv) < MIN_FILE_BYTES:
                print(f"    Response too small ({os.path.getsize(tmp_csv)} bytes) — skipping")
                os.remove(tmp_csv)
                continue

            # Gzip the CSV
            print(f"    Compressing → {dest_path}")
            with open(tmp_csv, "rb") as f_in, gzip.open(tmp_gz, "wb") as f_out:
                while True:
                    chunk = f_in.read(65536)
                    if not chunk:
                        break
                    f_out.write(chunk)
            os.remove(tmp_csv)
            os.replace(tmp_gz, dest_path)
            print(f"  Saved {dest_path} ({os.path.getsize(dest_path) / 1e6:.1f} MB compressed)")
            return True

        except Exception as e:
            print(f"    Error: {e}")
            for p in (tmp_csv, tmp_gz):
                if os.path.exists(p):
                    os.remove(p)
            continue

    # Tidy up any leftover temps
    for p in (tmp_csv, tmp_gz):
        if os.path.exists(p):
            os.remove(p)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download KS5 16-18 final results CSVs from DfE",
        epilog=(
            "Extra dependencies: pip install httpx\n\n"
            "Data source:\n"
            "  https://www.compare-school-performance.service.gov.uk/download-data/"
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
        help="Directory to write downloaded files (default: current directory)",
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

    with _make_client() as client:
        for year in args.years:
            # e.g. 2023-2024_england_ks5final.csv.gz
            dest = os.path.join(args.output_dir, f"{year}_england_ks5final.csv.gz")

            if os.path.exists(dest) and os.path.getsize(dest) > MIN_FILE_BYTES:
                print(f"Already exists: {dest}")
                results[year] = True
                continue
            elif os.path.exists(dest):
                print(f"Found incomplete file (too small), re-downloading: {dest}")

            print(f"\nDownloading KS5 data for {year}...")
            ok = download_ks5_year(year, dest, client)
            results[year] = ok

            if not ok:
                print(f"\n  *** Manual download for {year}: ***")
                print(f"  1. Go to: {DOWNLOAD_PAGE}")
                print(f"  2. Select year {year}, 'All of England', "
                      f"'16-18 results (final)', CSV format")
                print(f"  3. Download and gzip the file:")
                print(f"       gzip {year}_england_ks5final.csv")
                print(f"  4. Move to: {dest}")

            if year != args.years[-1]:
                time.sleep(random.uniform(1, 2))

    # --- Summary ---
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
