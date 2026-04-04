#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyzes robot Action data in a given directory.

Recursively finds folders containing 'episode' subdirectories, then finds
all 'abs_action/*.npy' files within them. Loads them in parallel using
multiple threads, validates shape is (6,), and prints statistics
(min, max, 1st percentile, 99th percentile) across all valid action data.

Usage:
    python calculate_min_max_action.py /path/to/training_data

Example:
    python calculate_min_max_action.py my_data/training_pipeline/training_data

This file was generated from RynnVLA-002/rynnvla-002/data_lerobot/calculate_min_max_all_data_action.py
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import sys
import concurrent.futures
import argparse

# Adjust based on your CPU core count and I/O capacity.
# For I/O-bound tasks on NAS/NFS, more threads than CPU cores is often effective.
# Start with 16 or 32.
MAX_WORKERS = 32

def find_all_action_npy_files_fast(root_directories):
    """
    Uses glob patterns to quickly find all action .npy files in the given directories.
    Assumes structure: <root_directory>/<any_name>/abs_action/<chunk>/<step>.npy
    """
    npy_file_paths = []
    if not root_directories:
        return npy_file_paths

    print("Scanning for files using glob pattern...")
    glob_pattern = os.path.join('*', 'abs_action', '*', '0.npy')

    for root_dir in tqdm(root_directories, desc="Scanning directories", unit="dir"):
        if not os.path.isdir(root_dir):
            print(f"Warning: directory '{root_dir}' does not exist, skipping.", file=sys.stderr)
            continue

        search_path = os.path.join(root_dir, glob_pattern)
        matched_files = glob.glob(search_path, recursive=False)
        npy_file_paths.extend(matched_files)

    print(f"Scan complete. Found {len(npy_file_paths)} .npy files.")
    return npy_file_paths

def load_and_validate_action(file_path):
    """
    Worker function for a single thread: loads a .npy file and validates its shape.
    Returns the action data or None on failure.
    """
    try:
        action_data = np.load(file_path)
        if action_data.shape == (6,):
            return action_data
        else:
            print(f"\nWarning: file '{file_path}' has shape {action_data.shape}, expected (6,). Skipping.", file=sys.stderr)
            return None
    except Exception as e:
        print(f"\nError: could not load '{file_path}': {e}", file=sys.stderr)
        return None

def analyze_action_data_multithreaded(file_paths):
    """
    Loads all .npy files using multiple threads, then computes and prints statistics.
    """
    if not file_paths:
        print("No .npy files found to analyze.")
        return

    all_actions = []
    print(f"Loading and processing action data using up to {MAX_WORKERS} threads...")

    # Use ThreadPoolExecutor to load files in parallel.
    # executor.map passes each element of file_paths to load_and_validate_action.
    # We wrap the iterator with tqdm to show a progress bar.
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iterator = executor.map(load_and_validate_action, file_paths)
        all_actions = [r for r in tqdm(results_iterator, total=len(file_paths), desc="Loading .npy files", unit="file") if r is not None]

    if not all_actions:
        print("\nNo valid data loaded. Check file contents and shapes.")
        return

    # Stack all actions into a single (N, 6) NumPy array
    print(f"\nData loaded. Stacking arrays...")
    stacked_actions = np.array(all_actions)

    print(f"Successfully loaded and stacked {stacked_actions.shape[0]} actions, shape: {stacked_actions.shape}")
    print("Computing statistics...")

    min_vals = np.min(stacked_actions, axis=0)
    max_vals = np.max(stacked_actions, axis=0)
    q01_vals = np.percentile(stacked_actions, 1, axis=0)
    q99_vals = np.percentile(stacked_actions, 99, axis=0)

    print("\n--- Action Data Statistics ---")
    print("-" * 85)
    print(f"{'Dim':<10} | {'Min':<20} | {'Max':<20} | {'1st percentile':<20} | {'99th percentile':<20}")
    print("-" * 85)

    for i in range(6):
        print(f"Dim {i:<6} | {min_vals[i]:<20.8f} | {max_vals[i]:<20.8f} | {q01_vals[i]:<20.8f} | {q99_vals[i]:<20.8f}")

    print("-" * 85)

def find_episode_directories(root_dir):
    """
    Finds all 'task directories' under the given root.
    A directory is considered a task directory if all its immediate subdirectories
    start with 'episode'.
    """
    path_list = []
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.", file=sys.stderr)
        return path_list

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip folders with no subdirectories
        if not dirnames:
            continue

        # Check if all subdirectories start with 'episode'
        all_episodes = all(dirname.lower().startswith('episode') for dirname in dirnames)

        if all_episodes:
            path_list.append(dirpath)
            # Stop descending into this directory to avoid duplicates
            dirnames.clear()

    return path_list

def run_analysis_on_dirs(input_dirs):
    """
    Runs the full analysis pipeline on the given list of input directories.
    """
    print("--- Searching for task directories ---")
    all_episode_dirs = []
    for path in input_dirs:
        print(f"Searching in '{path}'...")
        found_dirs = find_episode_directories(path)
        if found_dirs:
            print(f"  -> Found {len(found_dirs)} task directories in '{path}'.")
            all_episode_dirs.extend(found_dirs)
        else:
            print(f"  -> No task directories found in '{path}'.")

    if not all_episode_dirs:
        print("\nNo task directories found in any of the provided paths. Exiting.")
        print("Tip: the script looks for a directory whose immediate subdirectories all start with 'episode'.")
        return

    print(f"\nFound {len(all_episode_dirs)} total task directories.")

    all_files = find_all_action_npy_files_fast(all_episode_dirs)
    analyze_action_data_multithreaded(all_files)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze robot action .npy files and compute statistics.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'input_dirs',
        nargs='+',
        metavar='INPUT_DIR',
        help='One or more root directories to search.'
    )

    args = parser.parse_args()
    run_analysis_on_dirs(args.input_dirs)

if __name__ == "__main__":
    main()
