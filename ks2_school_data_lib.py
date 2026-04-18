"""
Shared library for KS2 primary school data processing.

Handles loading long-format EES data, joining with GIAS for addresses,
pivoting/consolidating across years, and feeding into the geocoding/crime
pipeline from school_data_lib.
"""

import os
import json
import logging

import pandas as pd
import numpy as np

import school_data_lib  # reuse geocoding, crime, caching utilities


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path="ks2_config.json"):
    """Load KS2 configuration, falling back to defaults."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found. Using defaults.")
        return get_default_config()
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing {config_path}: {e}. Using defaults.")
        return get_default_config()


def get_default_config():
    return {
        "filtering": {
            "percentile": 0.75,
            "subject": "Reading, writing and maths",
            "breakdown": "All pupils",
        },
        "grading": {
            "high_threshold": 85,
            "expected_threshold": 75,
        },
        "colors": {
            "high": "#005C29",
            "expected": "#228B22",
            "below_expected": "#8FBC8F",
        },
        "crime": {
            "school_crime_radius_km": 3,
            "crime_data_file": "combined_crimes.csv.gz",
            "excluded_crime_types": [
                "Shoplifting", "Bicycle theft", "Other theft", "Other crime",
                "Drugs", "Anti-social behaviour", "Criminal damage and arson",
            ],
        },
        "geocoding": {"index_name": "your-place-index-name", "region_name": "eu-west-2"},
        "caching": {
            "geocoding_cache_file": "ks2_geocoding_cache.json",
            "crime_cache_file": "ks2_crime_cache.json",
        },
        "data": {
            "ks2_performance_files": [
                "ks2_school_attainment_data.csv",
                "ks2_school_attainment_2122.csv",
            ],
            "gias_file": "gias_establishments.csv",
        },
        "output": {
            "processed_csv": "ks2_processed_school_data.csv",
            "map_filename": "ks2_schools_map.html",
        },
        "clustering": {"default_cluster_radius_km": 5, "default_min_schools": 2},
        "map": {
            "default_zoom": 8,
            "marker_size": 30,
            "cluster_marker_size": 20,
            "popup_max_width": 350,
        },
    }


def validate_config(config):
    """Basic config validation; raises ValueError on problems."""
    errors = []
    p = config["filtering"]["percentile"]
    if not 0 <= p <= 1:
        errors.append(f"percentile must be 0-1, got {p}")
    ht = config["grading"]["high_threshold"]
    et = config["grading"]["expected_threshold"]
    if ht <= et:
        errors.append(f"high_threshold ({ht}) must be > expected_threshold ({et})")
    if errors:
        raise ValueError("KS2 config validation failed:\n  - " + "\n  - ".join(errors))
    return True


# ---------------------------------------------------------------------------
# KS2 performance data loading
# ---------------------------------------------------------------------------

# Possible column names for the subject filter across different EES releases
_SUBJECT_COL_CANDIDATES = ["subject", "Subject"]
_BREAKDOWN_COL_CANDIDATES = ["breakdown", "Breakdown", "breakdown_topic"]
_URN_COL_CANDIDATES = ["school_urn", "urn", "URN"]
_NAME_COL_CANDIDATES = ["school_name", "institution_name"]
_PERIOD_COL_CANDIDATES = ["time_period", "academic_year"]
_EXPECTED_COL_CANDIDATES = [
    "expected_standard_pupil_percent",
    "percent_meeting_expected_standard",
    "pct_expected_standard",
]
_HIGHER_COL_CANDIDATES = [
    "higher_standard_pupil_percent",
    "percent_meeting_higher_standard",
    "pct_higher_standard",
]


def _find_col(df, candidates, required=True):
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"Could not find any of {candidates} in columns: {list(df.columns)[:20]}"
        )
    return None


def load_ks2_performance_files(file_paths, config):
    """
    Load one or more EES KS2 school-level CSV files and concatenate them.

    The EES format is long: one row per school × subject × breakdown × year.
    We filter to the configured subject and breakdown.

    Returns a DataFrame with standardised column names:
        school_urn, school_name, time_period,
        expected_pct, higher_pct
    """
    target_subject = config["filtering"]["subject"]
    target_breakdown = config["filtering"]["breakdown"]

    all_dfs = []
    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"KS2 file not found, skipping: {path}")
            continue

        logger.info(f"Loading {path} ...")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            continue

        logger.info(f"  Loaded {len(df):,} rows, columns: {list(df.columns[:8])}...")

        # Identify key columns — skip file entirely if no school-level URN column
        urn_col = _find_col(df, _URN_COL_CANDIDATES, required=False)
        if not urn_col:
            logger.warning(
                f"  No school_urn column in {path} "
                f"(columns: {list(df.columns[:8])}…) — "
                "this file does not contain school-level data, skipping."
            )
            continue
        name_col = _find_col(df, _NAME_COL_CANDIDATES, required=False)
        period_col = _find_col(df, _PERIOD_COL_CANDIDATES)
        subject_col = _find_col(df, _SUBJECT_COL_CANDIDATES)
        expected_col = _find_col(df, _EXPECTED_COL_CANDIDATES)
        higher_col = _find_col(df, _HIGHER_COL_CANDIDATES, required=False)

        # Filter to the configured subject
        original_len = len(df)
        df = df[df[subject_col].astype(str).str.strip() == target_subject]
        logger.info(
            f"  After subject filter ('{target_subject}'): {len(df):,} rows "
            f"(from {original_len:,})"
        )

        if len(df) == 0:
            # List available subjects to help debugging
            all_subjects = pd.read_csv(path, usecols=[subject_col], low_memory=False)[subject_col].unique()
            logger.warning(f"  Available subjects: {list(all_subjects)}")
            continue

        # Filter to the target breakdown (e.g. "All pupils") if a breakdown column exists
        breakdown_col = _find_col(df, _BREAKDOWN_COL_CANDIDATES, required=False)
        if breakdown_col:
            before = len(df)
            df_filtered = df[df[breakdown_col].astype(str).str.strip() == target_breakdown]
            if len(df_filtered) > 0:
                df = df_filtered
                logger.info(f"  After breakdown filter ('{target_breakdown}'): {len(df):,} rows")
            else:
                logger.warning(
                    f"  Breakdown '{target_breakdown}' not found; "
                    f"available: {df[breakdown_col].unique()[:10]}. Skipping breakdown filter."
                )

        # Standardise to common column names
        rename = {urn_col: "school_urn", period_col: "time_period", expected_col: "expected_pct"}
        if name_col:
            rename[name_col] = "school_name"
        if higher_col:
            rename[higher_col] = "higher_pct"
        df = df.rename(columns=rename)

        keep = ["school_urn", "time_period", "expected_pct"]
        if "school_name" in df.columns:
            keep.append("school_name")
        if "higher_pct" in df.columns:
            keep.append("higher_pct")
        df = df[keep].copy()

        # Normalise types
        df["school_urn"] = df["school_urn"].astype(str).str.strip()
        df["time_period"] = df["time_period"].astype(str).str.strip()
        df["expected_pct"] = pd.to_numeric(df["expected_pct"], errors="coerce")

        all_dfs.append(df)
        logger.info(f"  Kept {len(df):,} usable rows from {path}")

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined KS2 performance data: {len(combined):,} rows")
    return combined


# ---------------------------------------------------------------------------
# GIAS loading
# ---------------------------------------------------------------------------

# GIAS can use different encodings depending on when it was exported
_GIAS_ENCODINGS = ["utf-8-sig", "latin-1", "cp1252"]

_STATE_TYPE_GROUPS = {
    "Local authority maintained schools",
    "Free Schools",
    "Academies",
    "Special schools",  # included so special primaries are optionally retained
}


def load_gias(file_path, state_only=True):
    """
    Load GIAS establishment CSV and return open primary schools.

    Args:
        file_path: Path to gias_establishments.csv
        state_only: If True, exclude independent schools

    Returns:
        DataFrame with columns: URN, EstablishmentName, Street, Town, Postcode,
        TelephoneNum, Gender, NumberOfPupils, StatutoryLowAge, StatutoryHighAge,
        TypeGroup, EstablishmentStatus
    """
    if not os.path.exists(file_path):
        logger.error(f"GIAS file not found: {file_path}")
        return None

    df = None
    for enc in _GIAS_ENCODINGS:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            logger.info(f"Loaded GIAS with encoding {enc}: {len(df):,} rows")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        logger.error("Could not read GIAS file with any known encoding.")
        return None

    # Normalise column names (GIAS uses names with spaces and parentheses)
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)

    def find_gias_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    status_col = find_gias_col(["EstablishmentStatus (name)", "EstablishmentStatus"])
    phase_col = find_gias_col(["PhaseOfEducation (name)", "PhaseOfEducation"])
    type_group_col = find_gias_col(["EstablishmentTypeGroup (name)", "EstablishmentTypeGroup"])
    gender_col = find_gias_col(["Gender (name)", "Gender"])

    # Filter: open schools
    if status_col:
        df = df[df[status_col].astype(str).str.strip() == "Open"]
        logger.info(f"After Open filter: {len(df):,}")

    # Filter: primary phase
    if phase_col:
        df = df[df[phase_col].astype(str).str.strip() == "Primary"]
        logger.info(f"After Primary phase filter: {len(df):,}")

    # Filter: state schools only
    if state_only and type_group_col:
        df = df[df[type_group_col].isin(_STATE_TYPE_GROUPS)]
        logger.info(f"After state-only filter: {len(df):,}")

    # Standardise output columns
    keep = {}
    for col_candidates, out_name in [
        (["URN"], "URN"),
        (["EstablishmentName"], "EstablishmentName"),
        (["Street"], "Street"),
        (["Locality"], "Locality"),
        (["Town"], "Town"),
        (["County (name)", "County"], "County"),
        (["Postcode"], "Postcode"),
        (["TelephoneNum"], "TelephoneNum"),
        (["NumberOfPupils"], "NumberOfPupils"),
        (["StatutoryLowAge"], "StatutoryLowAge"),
        (["StatutoryHighAge"], "StatutoryHighAge"),
    ]:
        col = find_gias_col(col_candidates)
        if col:
            keep[col] = out_name

    if gender_col:
        keep[gender_col] = "Gender"

    df = df.rename(columns=keep)[[c for c in keep.values() if c in [keep[k] for k in keep]]]
    df["URN"] = df["URN"].astype(str).str.strip()

    return df


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def _format_time_period(tp):
    """Convert '202223' → '2022/23', leave other formats unchanged."""
    tp = str(tp).strip()
    if len(tp) == 6 and tp.isdigit():
        return f"20{tp[2:4]}/{tp[4:6]}"
    return tp


def consolidate_ks2_schools(perf_df, gias_df, config):
    """
    Merge performance data with GIAS and consolidate to one row per school.

    Steps:
    1. Join performance data with GIAS on URN
    2. For each school, average expected_pct across years (non-zero only)
    3. Build year-by-year score string for popup display

    Returns a DataFrame ready for geocoding.
    """
    # Join
    perf_df = perf_df.copy()
    perf_df["school_urn"] = perf_df["school_urn"].astype(str).str.strip()
    gias_df = gias_df.copy()
    gias_df["URN"] = gias_df["URN"].astype(str).str.strip()

    merged = perf_df.merge(gias_df, left_on="school_urn", right_on="URN", how="inner")
    logger.info(f"After GIAS join: {len(merged):,} rows ({merged['school_urn'].nunique():,} schools)")

    if len(merged) == 0:
        logger.error("No rows after joining KS2 data with GIAS. Check that URNs match.")
        return None

    # Exclude zero/invalid scores
    merged = merged[merged["expected_pct"] > 0]

    # Aggregate per school
    consolidated = []
    for urn, group in merged.groupby("school_urn"):
        avg_pct = group["expected_pct"].mean()
        avg_higher = pd.to_numeric(group["higher_pct"], errors="coerce").mean() if "higher_pct" in group.columns else None

        # Take the most recent row for metadata
        latest = group.iloc[-1].copy()

        year_scores = []
        for _, row in group.sort_values("time_period").iterrows():
            year_label = _format_time_period(row["time_period"])
            pct = row["expected_pct"]
            if pd.notna(pct) and pct > 0:
                year_scores.append(f"{year_label}: {pct:.0f}%")

        school = {
            "school_urn": urn,
            "school_name": latest.get("school_name", latest.get("EstablishmentName", "")),
            "EstablishmentName": latest.get("EstablishmentName", ""),
            "Street": latest.get("Street", ""),
            "Locality": latest.get("Locality", ""),
            "Town": latest.get("Town", ""),
            "County": latest.get("County", ""),
            "Postcode": latest.get("Postcode", ""),
            "TelephoneNum": latest.get("TelephoneNum", ""),
            "Gender": latest.get("Gender", ""),
            "NumberOfPupils": latest.get("NumberOfPupils", ""),
            "StatutoryLowAge": latest.get("StatutoryLowAge", ""),
            "StatutoryHighAge": latest.get("StatutoryHighAge", ""),
            "expected_pct": round(avg_pct, 2),
            "higher_pct": round(float(avg_higher), 2) if avg_higher is not None and pd.notna(avg_higher) else None,
            "year_scores": "<br>".join(year_scores),
        }
        consolidated.append(school)

    df_out = pd.DataFrame(consolidated)
    logger.info(f"Consolidated to {len(df_out):,} unique schools")
    return df_out


# ---------------------------------------------------------------------------
# Percentile filtering
# ---------------------------------------------------------------------------

def filter_by_percentile(df, config):
    """Filter schools to those at or above the configured percentile of expected_pct."""
    percentile = config["filtering"]["percentile"]
    threshold = df["expected_pct"].quantile(percentile)

    logger.info(f"\nKS2 expected_pct distribution:")
    for p in [0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
        score = df["expected_pct"].quantile(p)
        logger.info(f"  P{int(p*100):3d}: {score:.1f}%")

    df_filtered = df[df["expected_pct"] >= threshold]
    logger.info(
        f"\nFiltering at P{percentile*100:.0f} (≥{threshold:.1f}%): "
        f"{len(df_filtered):,} of {len(df):,} schools selected"
    )
    return df_filtered


# ---------------------------------------------------------------------------
# Address building for geocoding
# ---------------------------------------------------------------------------

def build_address(row):
    """Build a geocodeable address string from GIAS fields."""
    parts = []
    for field in ["Street", "Locality", "Town", "County", "Postcode"]:
        val = row.get(field, "")
        if pd.notna(val) and str(val).strip() and str(val).lower() != "nan":
            parts.append(str(val).strip())
    return ", ".join(parts) if parts else None


def geocode_and_enrich(df, crime_df, config):
    """
    Geocode school addresses and attach crime statistics.

    Wraps school_data_lib.geocode_address, using KS2 GIAS address fields.
    """
    from tqdm import tqdm

    df = df.copy()
    df["full_address"] = df.apply(build_address, axis=1)

    total = len(df)
    cache_hits = [0]
    cache_misses = [0]
    crime_calcs = [0]

    logger.info(f"Geocoding {total:,} schools...")
    with tqdm(total=total, desc="Geocoding", unit="school") as pbar:
        geocoded = df["full_address"].apply(
            lambda addr: school_data_lib.geocode_address(
                addr,
                config,
                pbar=pbar,
                hits_counter=cache_hits,
                misses_counter=cache_misses,
                crime_df=crime_df,
                crime_calc_counter=crime_calcs,
            )
        )

    df["Latitude"] = geocoded["Latitude"]
    df["Longitude"] = geocoded["Longitude"]
    df["crime_stats"] = geocoded["crime_stats"]

    logger.info(
        f"Geocoding complete — cached: {cache_hits[0]}, new: {cache_misses[0]}, "
        f"crime stats: {crime_calcs[0]}"
    )

    school_data_lib.save_geocoding_cache(config)
    school_data_lib.save_crime_cache(config)

    df = df.dropna(subset=["Latitude", "Longitude"])
    logger.info(f"Schools with valid coordinates: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Grade label helper
# ---------------------------------------------------------------------------

def expected_pct_to_label(pct, config):
    """Return a human-readable grade label for a given % meeting expected."""
    ht = config["grading"]["high_threshold"]
    et = config["grading"]["expected_threshold"]
    if pd.isna(pct) or pct == 0:
        return "N/A"
    if pct >= ht:
        return f"High (≥{ht}%)"
    if pct >= et:
        return f"Good (≥{et}%)"
    return "Above average"


def get_marker_color(pct, config):
    """Return hex colour for a school marker based on expected_pct."""
    ht = config["grading"]["high_threshold"]
    et = config["grading"]["expected_threshold"]
    colors = config["colors"]
    if pct >= ht:
        return colors["high"]
    if pct >= et:
        return colors["expected"]
    return colors["below_expected"]
