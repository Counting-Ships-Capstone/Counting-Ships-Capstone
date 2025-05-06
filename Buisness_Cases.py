# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 08:54:06 2025

@author: jylee
"""
# %% Importing Modules
import polars as pl 
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta
from typing import Sequence

# %% Defining Functions

# Files must be in this directory and titled: AIS_clipped_F#
def import_files(filepath: str) -> pl.DataFrame: 
    """
    Imports and merges all .xls AIS files matching the pattern in `filepath`. Data should be from
    MarineCadastre and have been clipped to the fairway in ArcGIS Pro. 
    
    Parameters: 
    - filepath (str): The filepath with an asterisk in the place of the file number.
    
    Returns: 
    - df (pl.DataFrame): The combined and cleaned dataframe with the track number column added.
    """
    
    #df = pl.read_excel(filepath) # used for .xls files 
    df = pl.read_csv(filepath, glob=True) # used for .csv files
    df = df.with_columns(
        pl.col("BaseDateTime").str.strptime(pl.Datetime).alias("BaseDateTime")
    )
    important_columns = ["MMSI", "BaseDateTime", "SOG", "COG", "Heading", "VesselType", "LAT", "LON", "Fairway_Heading"]
    df = add_track_number(df[important_columns].drop_nulls()) 
    return df 

def add_track_number(df: pl.DataFrame, time_between_tracks: int = 30) -> pl.DataFrame:
    """
    Adds a column with the track number to the dataset. Track number increments every time consecutive pings
    for the same vessel exceeds the time_between_tracks argument. Track number resets with a new MMSI

    Parameters: 
    - df (pl.DataFrame): the polars DataFrame.
    - time_between_tracks (int): The time in minutes between consecutive AIS pings from a 
      vessel for it to be considered a separate track. 
      
    Returns:
    - df (pl.Dataframe) The same dataset as the input but with a track number column. 
    """
        
    # Read CSV into a Polars DataFrame.
    # Define time gap threshold (default: 30 minutes).
    time_gap = timedelta(minutes=time_between_tracks)
    
    # Sort by MMSI and Timestamp to ensure correct track separation.
    df = df.sort(["MMSI", "BaseDateTime"])
    
    # Compute the time difference between consecutive pings for each vessel.
    df = df.with_columns(
        (pl.col("BaseDateTime") - pl.col("BaseDateTime").shift(1).over("MMSI"))
        .alias("Time_Diff")
    )

    # Identify where a new track starts (if time difference > threshold OR new MMSI).
    df = df.with_columns(
        ((pl.col("MMSI") != pl.col("MMSI").shift(1)) | (pl.col("Time_Diff") > time_gap))
        .cast(pl.Int32)
        .alias("New_Track")
    )
    
    # Generate a cumulative track number per MMSI.
    df = df.with_columns(
        pl.col("New_Track")
        .cum_sum() # Add +1 to start track numbering at 1.
        .over("MMSI")
        .alias("TrackNumber")
    )
    
    # Delete intermediary columns.
    df = df.drop(["Time_Diff","New_Track"])
    return df 

def save_file(df: pl.DataFrame, method: str, cutoff: float, dev_tolerance: float = None) -> None: 
    output_folder = "Filtered Output"
    os.makedirs(output_folder, exist_ok=True)
    if method == 'fairway_alignment':
        filename = os.path.join(output_folder, f"Filtered_Alignment_{dev_tolerance}dev_{cutoff:.2f}pct.csv")
    else: 
        filename = os.path.join(output_folder, f"Filtered_{method}_cutoff_{cutoff:.2f}.csv")
    df.write_csv(filename)
    print(f"Filtered data saved to: {filename}") 
    return 

def calculate_average_sog(df: pl.DataFrame, cutoff: float) -> pl.DataFrame:
    """
    Filters tracks by average speed over ground (SOG).

    Parameters:
    - df (pl.DataFrame): The dataframe containing AIS data with track numbers.
    - cutoff (float): Minimum speed in knots.

    Returns:
    - pl.DataFrame: Filtered dataset with only tracks having Avg SOG > cutoff.
    """
    track_stats = df.group_by(["MMSI", "TrackNumber"]).agg([
        pl.col("BaseDateTime").min().alias("StartTime"),
        pl.col("BaseDateTime").max().alias("EndTime"),
        pl.col("SOG").mean().alias("AvgSOG")
    ])
    
    valid_tracks = track_stats.filter(pl.col("AvgSOG") > cutoff)
    
    return df.join(valid_tracks.select(["MMSI", "TrackNumber"]), on=["MMSI", "TrackNumber"], how="inner")

def calculate_sinuosity(df: pl.DataFrame, cutoff: float) -> pl.DataFrame:
    df = df.sort(["MMSI", "TrackNumber", "BaseDateTime"])

    df = df.with_columns([
        (pl.col("LAT") - pl.col("LAT").shift(1).over(["MMSI", "TrackNumber"])).alias("dLAT"),
        (pl.col("LON") - pl.col("LON").shift(1).over(["MMSI", "TrackNumber"])).alias("dLON"),
    ])
    
    df = df.with_columns(
        (pl.col("dLAT") ** 2 + pl.col("dLON") ** 2).sqrt().alias("SegmentDist")
    )


    track_stats = df.group_by(["MMSI", "TrackNumber"]).agg([
        pl.col("LAT").first().alias("LAT_start"),
        pl.col("LAT").last().alias("LAT_end"),
        pl.col("LON").first().alias("LON_start"),
        pl.col("LON").last().alias("LON_end"),
        pl.col("SegmentDist").sum().alias("ActualDist")
    ])

    track_stats = track_stats.with_columns([
        ((pl.col("LAT_end") - pl.col("LAT_start")) ** 2 +
         (pl.col("LON_end") - pl.col("LON_start")) ** 2).sqrt().alias("StraightDist")
    ])

    # Avoid division by zero
    track_stats = track_stats.filter(pl.col("StraightDist") > 0)

    track_stats = track_stats.with_columns(
        (pl.col("ActualDist") / pl.col("StraightDist")).alias("Sinuosity")
    )

    valid_tracks = track_stats.filter(pl.col("Sinuosity") < cutoff)

    return df.join(valid_tracks.select(["MMSI", "TrackNumber"]), on=["MMSI", "TrackNumber"], how="inner")

def filter_by_cutoff(df: pl.DataFrame, method: str, cutoff: float) -> tuple[pl.DataFrame, float]:
    """
    Filters AIS vessel tracks using a specified method and cutoff threshold.
    
    This function delegates to either the average speed (SOG) filter or 
    the sinuosity filter based on the selected method. It then calculates 
    the percentage of data points retained after filtering.
    
    Parameters:
    - df (pl.DataFrame): The Polars DataFrame containing AIS data. 
      Must include 'MMSI', 'TrackNumber', and relevant columns for filtering.
    - method (str): The filtering method to use. Must be one of:
        - "average_sog" (retains tracks with average SOG > cutoff)
        - "sinuosity" (retains tracks with sinuosity < cutoff)
    - cutoff (float): The threshold value used for filtering. Interpretation depends on method.
    
    Returns:
    - tuple:
        - filtered_df (pl.DataFrame): The filtered dataset including only valid tracks.
        - cleaned_pct (float): The percentage of original data points retained after filtering.
    """
    if method == "average_sog":
        filtered_df = calculate_average_sog(df, cutoff)
    elif method == "sinuosity":
        filtered_df = calculate_sinuosity(df, cutoff)
    else:
        raise ValueError("Method must be 'average_sog' or 'sinuosity'.")

    cleaned_pct = 100 * filtered_df.height / df.height if df.height > 0 else 100
    return filtered_df, cleaned_pct

def plot_cutoff_performance(results: list[tuple[float, float]], method: str, output_path: str):
    """
    Plots the effect of a cutoff value on the percentage of data retained.

    Parameters:
    - results (list): List of tuples (cutoff, percent_cleaned)
    - method (str): "sinuosity" or "average_sog"
    - output_path (str): Path to save the plot image
    """
    results_arr = np.array(results)
    plt.figure(figsize=(8, 5))
    plt.plot(results_arr[:, 0], results_arr[:, 1], marker='o')
    plt.xlabel(f"{method.replace('_', ' ').capitalize()} Cutoff")
    plt.ylabel("Percentage of Clean Data (%)")
    plt.title(f"{method.replace('_', ' ').capitalize()} Cutoff vs. Retained Data")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def run_cutoff_sweep(
    df: pl.DataFrame,
    method: str,
    cutoff_values: list[float],
    output_plot_path: str = "cutoff_performance_plot.png") -> list[tuple[float, float]]:
    """
    Iterates through a list of cutoff values and runs a specified filter method on the dataset.
    Collects and plots the percentage of retained data for each cutoff value.

    Parameters:
    - df (pl.DataFrame): The input AIS dataset with TrackNumber column.
    - method (str): One of ["average_sog", "sinuosity"].
    - cutoff_values (list of float): A list of threshold values to test.
    - output_plot_path (str): Path to save the output performance plot.

    Returns:
    - results (list of tuples): Each tuple contains (cutoff, percent_retained)
    """
    results = []

    for cutoff in cutoff_values:
        _, pct_retained = filter_by_cutoff(df, method, cutoff)
        results.append((cutoff, pct_retained))
        save_file(df, method, cutoff)

    plot_cutoff_performance(results, method, output_plot_path)
    return results

def calculate_percent_alignment(df: pl.DataFrame, cutoff: float, dev_tol: float) -> tuple[pl.DataFrame, float]:
    """
    Filters AIS tracks based on a single deviation tolerance and cutoff.

    Parameters:
    - df (pl.DataFrame): Input data with 'Heading', 'Fairway_Heading', 'MMSI', 'TrackNumber'
    - cutoff (float): Minimum percent of aligned points required per track (0–1)
    - dev_tol (float): Angular deviation tolerance (degrees)

    Returns:
    - filtered_df (pl.DataFrame): Filtered tracks that pass the cutoff
    - cleaned_pct (float): Percentage of points retained
    """
    df = df.with_columns([
        (((pl.col("Heading") - pl.col("Fairway_Heading")).abs() <= dev_tol) |
         ((pl.col("Heading") - ((pl.col("Fairway_Heading") + 180) % 360)).abs() <= dev_tol))
        .cast(pl.Int8)
        .alias("Aligned")
    ])

    alignment_stats = df.group_by(["MMSI", "TrackNumber"]).agg(
        pl.col("Aligned").mean().alias("PercentAligned")
    )

    valid_tracks = alignment_stats.filter(pl.col("PercentAligned") > cutoff)

    filtered_df = df.join(valid_tracks.select(["MMSI", "TrackNumber"]), on=["MMSI", "TrackNumber"], how="inner")
    cleaned_pct = 100 * filtered_df.height / df.height if df.height > 0 else 100

    return filtered_df, cleaned_pct

def process_percent_alignment(
    df: pl.DataFrame,
    cutoff_values: Sequence[float],
    dev_tolerance_range: Sequence[float],
    output_folder: str = "Filtered output"
):
    """
    Iterates over all cutoff and heading deviation tolerance values,
    calculates alignment, saves filtered CSVs, and plots results.
    """
    os.makedirs(output_folder, exist_ok=True)
    cleaned_percentages = []

    for dev_tol in dev_tolerance_range:
        for cutoff in cutoff_values:
            filtered_df, cleaned_pct = calculate_percent_alignment(df, cutoff, dev_tol)
            cleaned_percentages.append((cutoff, dev_tol, cleaned_pct))
            save_file(filtered_df, 'fairway_alignment', cutoff, dev_tol)
    
    plot_percent_alignment(cleaned_percentages, output_path=os.path.join(output_folder, "Alignment_Performance.png"))

def plot_percent_alignment(cleaned_percentages: list[tuple[float, float, float]], output_path: str = "alignment_results.png") -> None:
    """
    Plots percentage of clean data vs. cutoff for each heading deviation tolerance.

    Parameters:
    - cleaned_percentages (list): List of tuples (cutoff, dev_tolerance, percent_cleaned)
    - output_path (str): File path to save the plot image.

    Returns:
    - None (displays and saves the plot)
    """
    cleaned_arr = np.array(cleaned_percentages)
    plt.figure(figsize=(8, 5))
    for dev_tol in np.unique(cleaned_arr[:, 1]):
        subset = cleaned_arr[cleaned_arr[:, 1] == dev_tol]
        plt.plot(subset[:, 0], subset[:, 2], marker='o', linestyle='-', label=f'Dev Tol {int(dev_tol)}')
    plt.xlabel('Percent Alignment Cutoff')
    plt.ylabel('Percentage of Clean Data (%)')
    plt.title('Optimal Percent Alignment and Heading Deviation Tolerance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)  # ← Save to PNG
    plt.show()

# %% Usage
if __name__ == "__main__":
    filepath = "C:/Users/jylee/coding/CAPSTONE/athirdY_clipped_OrigF*.csv"
    output_dir = "C:/Users/jylee/coding/CAPSTONE"
    df = import_files(filepath)
    # Define your sweep parameters     
    sinuosity_cutoffs = np.arange(1, 30, 1) 
    avg_sog_cutoffs = np.arange(0, 5, 0.1) 
    perc_alignment_cutoffs = np.arange(0.5, 1.05, 0.05)
    deviations = range(0, 15)# Heading deviation tolerances (e.g., 0 to 14 degrees) 
    run_cutoff_sweep(df, "sinuosity", sinuosity_cutoffs)
    run_cutoff_sweep(df, 'average_sog', avg_sog_cutoffs)
    process_percent_alignment(df, perc_alignment_cutoffs, deviations)