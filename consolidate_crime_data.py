import os
import pandas as pd
import glob
import argparse
import gzip
import shutil

CRIME_DATA = '/Users/orly/iCloudDrive/Documents/UK/crime_data/'

def process_crime_data(root_dir, output_file):
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} does not exist")
        return
    
    dfs = []
    columns = ['Month', 'Longitude', 'Latitude', 'Crime type']

    for dirpath, _, _ in os.walk(root_dir):
        csv_files = [f for f in glob.glob(os.path.join(dirpath, "*.csv"))
                    if 'outcomes' not in f.lower()]

        for file in csv_files:
            try:
                df = pd.read_csv(file, usecols=columns)
                dfs.append(df)
                print(f"Processed {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Write CSV then gzip it
        temp_csv = output_file.replace('.gz', '') if output_file.endswith('.gz') else output_file
        combined_df.to_csv(temp_csv, index=False)
        
        # Gzip the file
        with open(temp_csv, 'rb') as f_in:
            with gzip.open(f"{temp_csv}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove the uncompressed file
        os.remove(temp_csv)
        
        print(f"Combined {len(dfs)} files into {temp_csv}.gz")
    else:
        print("No valid CSV files found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Consolidate crime data CSV files')
    parser.add_argument('--crime-data-dir', default=CRIME_DATA, help=f'Directory containing crime CSV files (default: {CRIME_DATA})')
    parser.add_argument('--output', default='combined_crimes.csv', help='Output filename (will be gzipped)')
    
    args = parser.parse_args()
    process_crime_data(args.crime_data_dir, args.output)
