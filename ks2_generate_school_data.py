"""
Generate processed KS2 primary school data.

Pipeline:
  1. Load KS2 performance CSVs (long format from EES)
  2. Load GIAS for school addresses
  3. Join, pivot, and consolidate across years
  4. Filter by percentile
  5. Geocode with AWS Location Services (cached)
  6. Calculate crime statistics (cached)
  7. Save processed CSV and caches

Run this before ks2_create_standalone_app.py.
"""

import logging

import school_data_lib
import ks2_school_data_lib


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ks2_generate.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    # Load and validate config
    config = ks2_school_data_lib.load_config()
    try:
        ks2_school_data_lib.validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    crime_data_file = config["crime"]["crime_data_file"]
    school_crime_radius_km = config["crime"]["school_crime_radius_km"]
    output_csv = config["output"]["processed_csv"]
    ks2_files = config["data"]["ks2_performance_files"]
    gias_file = config["data"]["gias_file"]

    # Step 1: Load caches
    school_data_lib.load_geocoding_cache(config)
    school_data_lib.load_crime_cache(config)

    # Step 2: Load crime data
    crime_df = school_data_lib.load_crime_data(crime_data_file)

    # Step 3: Load KS2 performance data
    perf_df = ks2_school_data_lib.load_ks2_performance_files(ks2_files, config)
    if perf_df is None:
        logger.error(
            "No KS2 performance data loaded. "
            "Run ks2_download_data.py or download files manually."
        )
        return

    # Step 4: Load GIAS
    gias_df = ks2_school_data_lib.load_gias(gias_file)
    if gias_df is None:
        logger.error(
            "GIAS data not loaded. "
            "Run ks2_download_data.py or download gias_establishments.csv manually."
        )
        return

    # Step 5: Consolidate (join + average across years)
    df = ks2_school_data_lib.consolidate_ks2_schools(perf_df, gias_df, config)
    if df is None or len(df) == 0:
        logger.error("No schools after consolidation.")
        return

    # Step 6: Filter by percentile
    df = ks2_school_data_lib.filter_by_percentile(df, config)
    if len(df) == 0:
        logger.error("No schools after percentile filter.")
        return

    # Step 7: Geocode and enrich with crime data
    df = ks2_school_data_lib.geocode_and_enrich(df, crime_df, config)
    if len(df) == 0:
        logger.error("No schools with valid coordinates.")
        return

    # Step 8: Save processed data
    df.sort_values("expected_pct", ascending=False).to_csv(output_csv, index=False)
    logger.info(f"Saved processed data to {output_csv}")

    # Step 9: Compute crime indices
    df, crime_stats_list = school_data_lib.extract_and_index_crime_data(
        df, school_crime_radius_km
    )

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("PROCESSING COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Total schools: {len(df):,}")
    logger.info(f"Schools with crime data: {sum(1 for s in crime_stats_list if s)}")
    logger.info(f"expected_pct — min: {df['expected_pct'].min():.1f}%  "
                f"max: {df['expected_pct'].max():.1f}%  "
                f"mean: {df['expected_pct'].mean():.1f}%")
    logger.info(f"\nGenerated files:")
    logger.info(f"  {output_csv}")
    logger.info(f"  {config['caching']['geocoding_cache_file']}")
    logger.info(f"  {config['caching']['crime_cache_file']}")
    logger.info(f"\nNext step: python3 ks2_create_standalone_app.py")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
