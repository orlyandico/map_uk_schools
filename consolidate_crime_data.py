import os
import json
import pandas as pd
import glob
import argparse
import gzip
import shutil


# Load configuration
def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default values.")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}. Using default values.")
        return get_default_config()


def get_default_config():
    """Return default configuration if config file is missing"""
    return {
        "crime": {
            "crime_data_file": "combined_crimes.csv.gz",
            "source_crime_data_dir": None,  # Must be provided via CLI or config
            "excluded_outcomes": True,  # Exclude outcome files by default
        },
        "crime_processing": {
            "columns": ["Month", "Longitude", "Latitude", "Crime type"],
            "compress_output": True,
        },
    }


# Load configuration at module level
config = load_config()


def process_crime_data(root_dir=None, output_file=None):
    """
    Consolidate crime data CSV files from a directory structure.

    Args:
        root_dir: Root directory containing crime CSV files. If None, uses config.
        output_file: Output filename. If None, uses config.
    """
    # Use config defaults if not specified
    if root_dir is None:
        root_dir = config["crime"].get("source_crime_data_dir")
        if root_dir is None:
            print("Error: Crime data directory not specified. Use --crime-data-dir or add 'source_crime_data_dir' to config.json")
            return

    if output_file is None:
        output_file = config["crime"]["crime_data_file"]

    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} does not exist")
        return

    dfs = []
    columns = config["crime_processing"]["columns"]
    exclude_outcomes = config["crime"].get("excluded_outcomes", True)

    print(f"Processing crime data from: {root_dir}")
    print(f"Columns to extract: {columns}")

    for dirpath, _, _ in os.walk(root_dir):
        csv_files = [f for f in glob.glob(os.path.join(dirpath, "*.csv"))]

        # Optionally filter out outcomes files
        if exclude_outcomes:
            csv_files = [f for f in csv_files if 'outcomes' not in f.lower()]

        for file in csv_files:
            try:
                df = pd.read_csv(file, usecols=columns)
                dfs.append(df)
                print(f"Processed {file} ({len(df)} records)")
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nCombined {len(dfs)} files into {len(combined_df)} total records")

        # Determine if we should compress
        compress_output = config["crime_processing"]["compress_output"]

        if compress_output and not output_file.endswith('.gz'):
            output_file = f"{output_file}.gz"

        if output_file.endswith('.gz'):
            # Write CSV then gzip it
            temp_csv = output_file.replace('.gz', '')
            combined_df.to_csv(temp_csv, index=False)

            # Gzip the file
            with open(temp_csv, 'rb') as f_in:
                with gzip.open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove the uncompressed file
            os.remove(temp_csv)
            print(f"Saved compressed output to: {output_file}")
        else:
            # Write directly without compression
            combined_df.to_csv(output_file, index=False)
            print(f"Saved output to: {output_file}")
    else:
        print("No valid CSV files found")


if __name__ == "__main__":
    default_output = config["crime"]["crime_data_file"]
    default_dir = config["crime"].get("source_crime_data_dir", "<not configured>")

    parser = argparse.ArgumentParser(
        description='Consolidate crime data CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
  This script uses config.json for default values. You can add:
  {
    "crime": {
      "source_crime_data_dir": "/path/to/crime/data",
      "crime_data_file": "combined_crimes.csv.gz"
    }
  }
        """
    )
    parser.add_argument(
        '--crime-data-dir',
        default=None,
        help=f'Directory containing crime CSV files (default from config: {default_dir})'
    )
    parser.add_argument(
        '--output',
        default=None,
        help=f'Output filename (default from config: {default_output})'
    )

    args = parser.parse_args()
    process_crime_data(args.crime_data_dir, args.output)
