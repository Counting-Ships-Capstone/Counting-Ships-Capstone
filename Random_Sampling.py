# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:40:01 2025

@author: jylee and ChatGPT
"""
# %% Importing modules
import polars as pl
import numpy as np
from datetime import timedelta

# %% Defining Functions
def import_file(input_file: str = "ArcGISPro\\Capstone_20_March\\Year_Dataset.csv") -> pl.DataFrame: 

    """
    Imports the file and converts it to a polars DataFrame. 
    
    Parameter: input_file (string): The input .csv file. 
    
    Returns: Outputs a polars DataFrame. 
    """
    
    df = pl.read_csv(input_file, try_parse_dates=True)
    important_columns = ["MMSI", "BaseDateTime", "SOG", "COG", "Heading", "VesselType", "LAT", "LON"]
    df = df[important_columns]
    return df


def export_file(df: pl.DataFrame, output_file: str = "ArcGISPro\\Capstone_20_March\\Year_Dataset_Track_Number.csv"):
    """
    Exports the file to a .csv file. 
    
    Parameters: 
    - df (pl.DataFrame): The complete polars DataFrame. 
    - output_file (string): The output .csv file. 

    """
    
    df.write_csv(output_file)
    
    return print(f"Processed data saved to: {output_file}") 


def add_track_number(df: pl.DataFrame, time_between_tracks: int = 30) -> pl.DataFrame:
    """
    Adds a column with the track number to a dataset from MarineCadastre. 

    Parameters: 
    - df (pl.DataFrame): the polars DataFrame.
    - time_between_tracks (integer): The time in minutes between consecutive AIS pings from a 
      vessel for it to be considered a separate track. 
      
    Returns: Outputs a .csv of the same dataset but with a track number column. 
    """
        
    # Read CSV into a Polars DataFrame    
    # Define time gap threshold (default: 30 minutes)
    time_gap = timedelta(minutes=time_between_tracks)
    
    # Sort by MMSI and Timestamp to ensure correct track separation
    df = df.sort(["MMSI", "BaseDateTime"])
    
    # Compute the time difference between consecutive pings for each vessel
    df = df.with_columns(
        (pl.col("BaseDateTime") - pl.col("BaseDateTime").shift(1).over("MMSI")).alias("Time_Diff")
    )

    # Identify where a new track starts (if time difference > threshold OR new MMSI)
    df = df.with_columns(
        ((pl.col("MMSI") != pl.col("MMSI").shift(1)) | (pl.col("Time_Diff") > time_gap))
        .cast(pl.Int32)
        .alias("New_Track")
    )
    
    # Generate a cumulative track number per MMSI
    df = df.with_columns(
        pl.col("New_Track")
        .cum_sum()
        .over("MMSI")
        .alias("TrackNumber")
    )
    
    # Delete intermediary columns
    df = df.drop(["Time_Diff","New_Track"])
    
    return df 

def sample_tracks(df: pl.DataFrame, 
                  num_samples: int = 94437,
                  seed: int = 123
                  ): 
    """
    Randomly samples vessel tracks based on random timestamps.
    
    Parameters:
    - df (pl.DataFrame): the polars DataFrame.
    - num_samples (integer): Number of random timestamps to sample.
    - seed (integer): Random seed for reproducibility.
    
    Returns: output .csv file containing the reduced dataset. 
    """

    # Read CSV into a Polars DataFrame    
    np.random.seed(seed)  # Set seed for reproducibility
    
    unique_tracks = df["TrackNumber"].unique()

    random_tracks = np.random.choice(unique_tracks.to_numpy(), num_samples, replace=False)
        
    # Filter dataset to include all points from selected tracks
    reduced_df = df.filter(pl.col("TrackNumber").is_in(random_tracks))
    
    return reduced_df


def ratio(df: pl.DataFrame,
          fraction: float = 1/3) -> int: 
    """
    Gives the number of tracks needed to sample some percentage of the tracks. 
    
    Perameters: 
    - df (pl.DataFrame): the polars DataFrame.
    - fraction (float): The fraction of tracks you want to have in your new sample. 
    
    Returns: 
    - number (integer): The number of tracks you need to have in your sample.
    """
    
    total = df.n_unique(subset='TrackNumber')
    number = int(fraction*total)
    
    return number


# %% Implementation
year_dataset = "ArcGISPro\\Capstone_20_March\\Year_Dataset.csv"
year_dataset_track_number = "ArcGISPro\\Capstone_20_March\\Year_Dataset_Track_Number.csv"
year_dataset_reduced = "ArcGISPro\\Capstone_20_March\\Year_Dataset_Reduced.csv"

raw_df = import_file(year_dataset)
df = add_track_number(raw_df)
reduced_df = sample_tracks(df, ratio(df, 0.01))
export_file(reduced_df, year_dataset_reduced)

 
