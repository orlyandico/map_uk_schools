# Map UK Schools

Use UK government data to filter and plot UK schools, filtering by GCSE results.

This Python script uses the CSV file obtained from this location:

```
https://www.compare-school-performance.service.gov.uk/download-data

- select 2022-2023
- select all of England
- select 16-18 results (final)
- select CSV format

The actual URL (you will need to download it with a browser) is
https://www.compare-school-performance.service.gov.uk/download-data?download=true&regions=0&filters=KS5&fileformat=csv&year=2022-2023&meta=false
```

The script requires several packages:

    pip install pandas
    pip install googlemaps
    pip install tqdm
    pip install folium
    
No longer uses Google Maps API, now uses Amazon Location Services.  AWS credentials must be set properly (via ```aws configure```)

Simply run the Python script, it will look for an input CSV file named `2022-2023_england_ks5final.csv` and produce two output files: `processed_school_data.csv` (the filtered school data) and `schools_map.html` (the actual map).

You can change the filtering of which schools to add (out of the 2973 total in England) at lines 79-89.
