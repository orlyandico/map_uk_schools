#!/usr/bin/env python3
"""
Look up crime counts around one or more free-form UK addresses.

Reads addresses either from --input <file> (one per line) or, if no input file
is given, by prompting interactively (type STOP or END to finish). Each
address is geocoded via Amazon Location Service (using the place index
configured in config.json). All crimes from combined_crimes.csv.gz within
500 m of each point (or the radii given via --radius 250,500,1000) are
aggregated and written to a side-by-side CSV with one column per address
(short code) and one row per (radius, crime type).
"""

import argparse
import csv
import sys
from dataclasses import dataclass

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from school_data_lib import (
    calculate_circle_bounding_box,
    haversine_vectorized,
    load_config,
    load_crime_data,
)

DEFAULT_RADII_M = [500]
STOP_TOKENS = {"STOP", "END"}


def parse_radii(value):
    try:
        radii = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--radius must be a comma-separated list of integer metres (got {value!r})"
        )
    if not radii or any(r <= 0 for r in radii):
        raise argparse.ArgumentTypeError(
            f"--radius must contain at least one positive integer (got {value!r})"
        )
    return radii


@dataclass
class Address:
    code: str
    raw: str
    label: str
    lat: float
    lon: float


def geocode(address, config):
    geo = config["geocoding"]
    client = boto3.client("location", region_name=geo["region_name"])
    response = client.search_place_index_for_text(
        IndexName=geo["index_name"],
        Text=address,
        FilterCountries=["GBR"],
        MaxResults=1,
    )
    if not response.get("Results"):
        return None
    place = response["Results"][0]["Place"]
    lon, lat = place["Geometry"]["Point"]
    label = place.get("Label", address)
    return lat, lon, label


def crimes_within(crime_df, lat, lon, radius_km):
    min_lat, max_lat, min_lon, max_lon = calculate_circle_bounding_box(
        lat, lon, radius_km
    )
    bbox = crime_df[
        crime_df["Latitude"].between(min_lat, max_lat)
        & crime_df["Longitude"].between(min_lon, max_lon)
    ]
    if bbox.empty:
        return bbox
    distances = haversine_vectorized(
        lat, lon, bbox["Latitude"].values, bbox["Longitude"].values
    )
    return bbox[distances <= radius_km]


def print_aggregate(radius_m, hits, all_crime_types):
    total = len(hits)
    print(f"\nWithin {radius_m} m: {total} crimes")
    by_type = (
        hits["Crime type"]
        .value_counts()
        .reindex(all_crime_types, fill_value=0)
    )
    width = max(len(t) for t in all_crime_types)
    for crime_type, count in by_type.items():
        print(f"  {crime_type.ljust(width)}  {count}")


def prompt_address_lines():
    print("Enter UK addresses, one per line. Type STOP or END to finish.")
    while True:
        try:
            yield input("Address: ")
        except EOFError:
            print()
            return


def file_address_lines(path):
    with open(path) as f:
        for line in f:
            yield line


def geocode_addresses(raw_lines, config):
    addresses = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.upper() in STOP_TOKENS:
            break
        try:
            result = geocode(raw, config)
        except (BotoCoreError, ClientError) as e:
            print(f"  Geocoding failed for '{raw}': {e}", file=sys.stderr)
            continue
        if result is None:
            print(f"  No geocoding result for '{raw}', skipping.", file=sys.stderr)
            continue
        lat, lon, label = result
        code = f"A{len(addresses) + 1}"
        print(f"  [{code}] {label}  ({lat:.6f}, {lon:.6f})")
        addresses.append(Address(code=code, raw=raw, label=label, lat=lat, lon=lon))
    return addresses


def write_csv(path, addresses, counts_by_addr, all_crime_types, radii_m):
    """
    counts_by_addr: dict of code -> dict of (radius_m, crime_type) -> int.

    CSV layout:
        Address legend : one row per address — [<full address>, <short code>]
        Column header  : "Radius_m", "Crime type", A1, A2, ...
        Then one row per (radius, crime type).
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for a in addresses:
            w.writerow([a.label, a.code])
        w.writerow(["Radius_m", "Crime type"] + [a.code for a in addresses])
        for radius_m in radii_m:
            for crime_type in all_crime_types:
                row = [radius_m, crime_type]
                for a in addresses:
                    row.append(counts_by_addr[a.code].get((radius_m, crime_type), 0))
                w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="crime_lookup_output.csv",
        help="CSV output path (default: crime_lookup_output.csv)",
    )
    parser.add_argument(
        "--input",
        help="Text file with one address per line. If given, no interactive prompting.",
    )
    parser.add_argument(
        "--radius",
        type=parse_radii,
        default=DEFAULT_RADII_M,
        help="Comma-separated list of radii in metres (default: 500).",
    )
    args = parser.parse_args()

    config = load_config()

    if args.input:
        raw_lines = file_address_lines(args.input)
    else:
        raw_lines = prompt_address_lines()
    addresses = geocode_addresses(raw_lines, config)
    if not addresses:
        print("No addresses entered.", file=sys.stderr)
        sys.exit(1)

    crime_df = load_crime_data(config["crime"]["crime_data_file"])
    if crime_df is None:
        print("Crime data file not available.", file=sys.stderr)
        sys.exit(4)

    all_crime_types = sorted(crime_df["Crime type"].dropna().unique())

    counts_by_addr = {}
    for a in addresses:
        print(f"\n=== [{a.code}] {a.label} ===")
        counts = {}
        for radius_m in args.radius:
            hits = crimes_within(crime_df, a.lat, a.lon, radius_m / 1000.0)
            print_aggregate(radius_m, hits, all_crime_types)
            by_type = (
                hits["Crime type"]
                .value_counts()
                .reindex(all_crime_types, fill_value=0)
            )
            for crime_type, count in by_type.items():
                counts[(radius_m, crime_type)] = int(count)
        counts_by_addr[a.code] = counts

    write_csv(args.output, addresses, counts_by_addr, all_crime_types, args.radius)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
