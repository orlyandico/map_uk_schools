import os
import pandas as pd
import glob

CRIME_DATA = '/Users/orly/iCloudDrive/Documents/UK/crime_data/'

def process_crime_data(root_dir, output_file):
   dfs = []
   columns = ['Month', 'Longitude', 'Latitude', 'Crime type']

   for dirpath, _, _ in os.walk(root_dir):
       # Exclude files containing 'outcomes'
       csv_files = [f for f in glob.glob(os.path.join(dirpath, "*.csv"))
                   if 'outcomes' not in f.lower()]

       for file in csv_files:
           try:
               df = pd.read_csv(file, usecols=columns)
               dfs.append(df)
               print(f"Processed {file}")
           except ValueError as e:
               print(f"Error reading {file}: {e}")
               continue

   if dfs:
       combined_df = pd.concat(dfs, ignore_index=True)
       combined_df.to_csv(output_file, index=False)

# Usage
process_crime_data(CRIME_DATA, 'combined_crimes.csv')
