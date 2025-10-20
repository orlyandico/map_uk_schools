"""
Generate processed school data from UK government CSV files

This script processes school performance CSVs, geocodes addresses, calculates
crime statistics, and saves processed data ready for web app generation.

Run this first, then use create_standalone_app.py to generate the web application.
"""

import logging

# Import shared library
import school_data_lib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plot_schools.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main orchestration function to process schools data

    This function coordinates the entire pipeline:
    1. Validate configuration
    2. Load data (crime data, school data)
    3. Process and consolidate schools
    4. Geocode and enrich with crime stats
    5. Save processed data and caches
    """
    # Load configuration
    config = school_data_lib.load_config()

    # Validate configuration first
    try:
        school_data_lib.validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        return

    # Extract configuration values
    crime_data_file = config["crime"]["crime_data_file"]
    school_crime_radius_km = config["crime"]["school_crime_radius_km"]
    percentile = config["filtering"]["percentile"]
    school_data_pattern = config["data"]["school_data_pattern"]
    output_csv = config["output"]["processed_csv"]

    # Step 1: Load caches
    school_data_lib.load_geocoding_cache(config)
    school_data_lib.load_crime_cache(config)

    # Step 2: Load crime data
    crime_df = school_data_lib.load_crime_data(crime_data_file)

    # Step 3: Load and combine school data files
    df = school_data_lib.load_school_data_files(school_data_pattern)
    if df is None:
        return

    # Step 4: Process and consolidate schools
    df_selected = school_data_lib.process_and_consolidate_schools(df, config)
    if df_selected is None:
        return

    # Step 5: Filter by percentile
    df_selected = school_data_lib.filter_schools_by_percentile(df_selected, percentile, config)

    # Step 6: Geocode schools and enrich with crime data
    df_selected = school_data_lib.geocode_and_enrich_schools(df_selected, crime_df, config)

    # Step 7: Validate we have schools to process
    if len(df_selected) == 0:
        logger.error("No schools with valid coordinates to display on map")
        return

    # Step 8: Save processed data
    df_selected.sort_values(by="TB3PTSE", ascending=False).to_csv(
        output_csv, index=False
    )
    logger.info(f"Saved processed data to {output_csv}")

    # Step 9: Extract and index crime data
    df_selected, school_crime_stats = school_data_lib.extract_and_index_crime_data(
        df_selected, school_crime_radius_km
    )

    # Print final statistics
    logger.info(f"\n{'='*50}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Total schools processed: {len(df_selected)}")
    logger.info(f"Schools with crime data: {sum(1 for s in school_crime_stats if s)}")

    if school_crime_stats:
        crime_counts = [s['total_crimes'] for s in school_crime_stats if s]
        if crime_counts:
            logger.info(f"\nCrime Statistics (3km radius):")
            logger.info(f"  Min: {min(crime_counts)} crimes")
            logger.info(f"  Max: {max(crime_counts)} crimes")
            logger.info(f"  Avg: {sum(crime_counts)/len(crime_counts):.1f} crimes")

    logger.info(f"\nGenerated files:")
    logger.info(f"  - {output_csv}")
    logger.info(f"  - geocoding_cache.json")
    logger.info(f"  - crime_cache.json")
    logger.info(f"\nNext step: Run 'python3 create_standalone_app.py' to generate web app")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
