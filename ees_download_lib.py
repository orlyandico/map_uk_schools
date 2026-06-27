"""
Shared helpers for downloading from the DfE Explore Education Statistics (EES)
content API, used by ks5_download_data.py and ks2_download_data.py.

Covers the parts both downloaders share: a browser-like httpx client, the EES
release listing, academic-year matching with release preference, the release
ZIP download, stale-temp cleanup, and the run summary. Each downloader keeps its
own publication-specific transform (column joins, CSV extraction, GIAS, etc.).
"""

import os
import sys

import httpx

# EES content API — stable REST endpoints, no JS rendering needed.
EES_CONTENT_API = "https://content.explore-education-statistics.service.gov.uk/api"

# Browser-like headers that avoid trivial bot-detection on government sites.
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


def make_client(timeout: float = 120.0) -> httpx.Client:
    """Return an httpx client with browser-like headers and a generous timeout."""
    return httpx.Client(headers=BROWSER_HEADERS, timeout=timeout, follow_redirects=True)


def ees_get(client: httpx.Client, path: str):
    """GET a JSON endpoint from the EES content API; None on error."""
    url = f"{EES_CONTENT_API}/{path.lstrip('/')}"
    try:
        resp = client.get(url, headers={**BROWSER_HEADERS, "Accept": "application/json"})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  EES API error ({path}): {e}", file=sys.stderr)
        return None


def get_ees_releases(client: httpx.Client, slug: str) -> list[dict]:
    """Return all releases for a publication slug.

    Uses /publications/{slug}/releases (not /publications/{slug}, which returns
    an empty releases list).
    """
    data = ees_get(client, f"publications/{slug}/releases")
    if not data:
        return []
    releases = data if isinstance(data, list) else data.get("releases", [])
    print(f"  Found {len(releases)} releases for '{slug}'")
    for r in releases:
        print(f"    {r.get('yearTitle')} — {r.get('slug')} (id={r.get('id')})")
    return releases


def normalise_year(year_title: str) -> str:
    """Normalise any year format to the short 'YYYY/YY' form used by EES.

    Handles '2023-2024', '2023/2024', '2023-24', '2023/24' → '2023/24'.
    """
    s = (year_title or "").replace("-", "/")
    parts = s.split("/")
    if len(parts) == 2 and len(parts[1]) == 4:
        return f"{parts[0]}/{parts[1][2:]}"
    return s


def _release_preference(r: dict) -> int:
    """Lower is preferred: revised < default < provisional."""
    slug = r.get("slug", "")
    if "revised" in slug:
        return 0
    if "provisional" in slug:
        return 2
    return 1


def pick_release(year_title: str, releases: list[dict]) -> dict | None:
    """Select the best release for an academic year, preferring 'revised'.

    Prints the available years and returns None when no release matches.
    """
    normalised = normalise_year(year_title)
    candidates = [
        r for r in releases if normalise_year(r.get("yearTitle") or "") == normalised
    ]
    if not candidates:
        print(f"  No release found for year '{year_title}'.")
        print(f"  Available: {[r.get('yearTitle') for r in releases]}")
        return None
    return sorted(candidates, key=_release_preference)[0]


def download_release_zip(client: httpx.Client, release_id: str) -> bytes | None:
    """Download a release's data ZIP into memory (typically 5-30 MB)."""
    url = f"{EES_CONTENT_API}/releases/{release_id}/files"
    print("  Downloading release ZIP...")
    try:
        resp = client.get(url, headers={**BROWSER_HEADERS, "Accept": "application/zip"})
        resp.raise_for_status()
        data = resp.content
        print(f"  ZIP size: {len(data) / 1e6:.1f} MB")
        return data
    except Exception as e:
        print(f"  Failed to download ZIP: {e}")
        return None


def clean_stale_tmp(output_dir: str) -> None:
    """Remove any .tmp files left behind by an interrupted previous run."""
    for fname in os.listdir(output_dir):
        if fname.endswith(".tmp"):
            tmp = os.path.join(output_dir, fname)
            print(f"Removing stale temp file: {tmp}")
            os.remove(tmp)


def print_summary(results: dict) -> bool:
    """Print the per-item OK/FAILED table; return True if everything succeeded."""
    print("\n" + "=" * 50)
    print("Download summary:")
    all_ok = True
    for key, ok in results.items():
        status = "OK" if ok else "FAILED (manual download required)"
        print(f"  {key}: {status}")
        if not ok:
            all_ok = False
    return all_ok
