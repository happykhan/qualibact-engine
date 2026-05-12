import gzip
import re
import pandas as pd
import os
import logging


# Function to extract genus and species
def rename_species(name):
    pattern = r"^([A-Za-z]+)_?[A-Za-z]*\s([a-z]+)"
    match = re.search(pattern, name)
    if match:
        return f"{match.group(1)} {match.group(2)}"  # Genus species format
    return name  # Return unchanged if no match


def get_df(
    tsv_table_path,
    workdir,    
    min_genome_count=None,
    species_list=None,
    force=False,
):
    filtered_file = os.path.join(workdir, "metadata_filtered.parquet")
    pq_table_path = os.path.join(workdir, "metadata.parquet")
    if not os.path.exists(pq_table_path) or force or not os.path.exists(filtered_file):
        full_df = create_full_df(tsv_table_path)
        full_df, full_df_filtered = filter_assembly_data(
            full_df, min_genome_count, species_list
        )
        # Save the dataframe to a parquet file
        full_df.to_parquet(pq_table_path, index=False)
        logging.info("Parquet file created at %s", pq_table_path)
        full_df_filtered.to_parquet(
            os.path.join(workdir, "metadata_filtered.parquet"), index=False
        )
        # Create a table of species and their counts for full_df
        species_counts = full_df["species_sylph"].value_counts().reset_index()
        species_counts.columns = ["species_sylph", "count"]
        # Write to csv
        species_counts = species_counts.rename(columns={"species_sylph": "species"})
        species_counts.to_csv(os.path.join(workdir, "species_counts.csv"), index=False)
    else:
        full_df = pd.read_parquet(pq_table_path)
        logging.info("Parquet file loaded from %s", pq_table_path)
        full_df_filtered = pd.read_parquet(filtered_file)
    return full_df, full_df_filtered


def filter_assembly_data(assembly_stats, min_genome_count=None, species_list=None):
    # remove rows where species_sylph is NaN
    assembly_stats = assembly_stats[~assembly_stats["species_sylph"].isna()]
    # Get rid of species where is ends with _sp . So it's actually a grouped genus
    assembly_stats = assembly_stats[
        ~assembly_stats["species_sylph"].str.endswith("_sp")
    ]
    assembly_stats["species_sylph"] = assembly_stats["species_sylph"].apply(
        rename_species
    )
    # Rename Clostridioides difficile to Clostridium difficile
    assembly_stats["species_sylph"] = assembly_stats["species_sylph"].replace(
        "Clostridioides difficile", "Clostridium difficile"
    )
    # Species that dont work
    excluded_species = [
        "Salmonella diarizonae",
        "Sarcina perfringens",
        "Salmonella houtenae",
    ]
    assembly_stats = assembly_stats[
        ~assembly_stats["species_sylph"].isin(excluded_species)
    ]
    # filter on species list
    if species_list is not None:
        # Remove any species that are not in the list
        assembly_stats = assembly_stats[
            assembly_stats["species_sylph"].isin(species_list)
        ]
    assembly_stats = assembly_stats[~assembly_stats["species_sylph"].str.contains(";")]
    # filter species == unknown
    assembly_stats = assembly_stats[
        ~assembly_stats["species_sylph"].str.contains("unknown")
    ]
    if min_genome_count is not None:
        # Do not include rows where number per species < MIN_GENOME_COUNT
        assembly_stats_filtered = (
            assembly_stats.copy()
            .groupby("species_sylph")
            .filter(lambda x: len(x) >= min_genome_count)
        )
    else:
        assembly_stats_filtered = assembly_stats.copy()
    return assembly_stats, assembly_stats_filtered

from qualibact.species_run import METRICS_LIST

def create_full_df(
    tsv_table_path
):
    # Check the file exists
    if not os.path.exists(tsv_table_path):
        logging.error(
            "Required file not found: %s\n"
            "Please ensure that the merged table is present.",
            tsv_table_path,
        )
        raise FileNotFoundError(
            f"Required file not found: {tsv_table_path}"
        )

    # Load the TSV into a dataframe
    if tsv_table_path.endswith(".gz"):
        assembly_stats = pd.read_csv(tsv_table_path, sep="\t", compression="gzip")
    else:
        assembly_stats = pd.read_csv(tsv_table_path, sep="\t")

    # Check that all required columns are present
    missing_cols = [col for col in METRICS_LIST if col not in assembly_stats.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {tsv_table_path}: {missing_cols}. "
            f"Expected columns: {METRICS_LIST}"
        )

    return assembly_stats
