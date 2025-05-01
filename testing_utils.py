#!/usr/bin/env python3
import os
import csv
import json
import shutil
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

# Ensure outputs directory exists
def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    return output_dir

# Clean up temporary folders
def cleanup_temp_folders(folders=None):
    """Clean up temporary folders after processing"""
    if folders is None:
        folders = ['static/data', 'static/audio', 'static/frames']
    
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Cleaned up {folder}")

# Save results to CSV file
def save_results_to_csv(results, output_path):
    """Save results to CSV file"""
    with open(output_path, 'w', newline='') as f:
        if not results:
            return
        
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_path}")

# Save results to JSON file
def save_results_to_json(results, output_path):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_path}")

# Print summary table
def print_summary_table(results, failed_videos=None):
    """Print summary table of results"""
    if not results:
        print("No results to display")
        return
    
    # Extract the keys for the table
    headers = list(results[0].keys())
    table_data = [[row.get(header, "") for header in headers] for row in results]
    
    print("\nResults summary:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print failed videos if any
    if failed_videos and len(failed_videos) > 0:
        print("\nFailed videos:")
        for video in failed_videos:
            print(f"- {video}")

# Generate output filename
def generate_output_filename(prefix='results', extension='csv'):
    """Generate unique output filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}" 