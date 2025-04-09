#----This script calculates the nearest 3D distance from antibody points to fibrillar structures, compares them to a random spatial distribution, and evaluates their significance using statistical analysis and visualization.----
#usage:python ab_fibril_distance.py --fibril Fibril.tbl --ab Ab.txt --output_prefix H21_analysis

import argparse
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze nearest distances between antibodies and fibrils.")
    parser.add_argument('--fibril', required=True, help='Path to Fibril .tbl file')
    parser.add_argument('--ab', required=True, help='Path to Antibody data file')
    parser.add_argument('--output_prefix', default='result', help='Prefix for output files')
    return parser.parse_args()

def extract_and_save_data(file_path):
    """
    Load fibril or antibody data from file and save extracted version.
    Returns a pandas DataFrame with columns x, y, z.
    """
    try:
        if 'Fibril' in os.path.basename(file_path):
            data = pd.read_csv(file_path, delim_whitespace=True, header=None, usecols=[19, 20, 23, 24, 25])
            data.columns = ['VolumeID', 'FilamentID', 'x', 'y', 'z']
        else:
            data = pd.read_csv(file_path, delimiter=r'\s+', engine='python')
            if data.shape[1] == 1:
                data = pd.read_csv(file_path, delimiter=',')
            if data.shape[1] != 3:
                raise ValueError("Unexpected number of columns in antibody data file.")
            data.columns = ['x', 'y', 'z']
        data.to_csv(file_path.replace('.tbl', '_extracted.txt').replace('.txt', '_extracted.txt'), index=False)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)

def simulate_filaments(data):
    """Group the input dataframe by FilamentID and return dict of filament coordinates."""
    filaments = {}
    for fid, group in data.groupby('FilamentID'):
        filaments[fid] = group[['x', 'y', 'z']].values
    return filaments

def calculate_and_convert_distances(ab_points, filaments):
    """Calculate nearest distance from each antibody point to the nearest fibril."""
    filament_coords = np.vstack([coords for coords in filaments.values()])
    if filament_coords.size == 0:
        raise ValueError("Fibril coordinates are empty.")
    tree = cKDTree(filament_coords)
    distances, _ = tree.query(ab_points[['x', 'y', 'z']], k=1)
    distances_nm = distances * 10.96 / 10  # Convert pixels to nanometers
    return distances_nm

def random_simulation_and_save(bounds, n_points, file_name):
    """Generate random 3D points within given bounds and save to file."""
    simulated_data = pd.DataFrame({
        'x': np.random.uniform(bounds[0][0], bounds[0][1], n_points),
        'y': np.random.uniform(bounds[1][0], bounds[1][1], n_points),
        'z': np.random.uniform(bounds[2][0], bounds[2][1], n_points),
    })
    simulated_data.to_csv(file_name, index=False)
    return simulated_data

def statistical_analysis_and_plot(experimental, simulated, output_prefix='result'):
    """Perform t-test and plot density distribution of distances."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(experimental, fill=True, common_norm=False, color='green', label='Experimental Data (nm)')
    sns.kdeplot(simulated, fill=True, common_norm=False, color='orange', label='Simulated Data (nm)')

    t_stat, p_value = ttest_ind(experimental, simulated)
    max_density = max(plt.gca().get_ylim())

    if p_value < 0.05:
        plt.text(x=np.mean(experimental), y=max_density * 0.9, s='*', fontsize=20, ha='center')
        plt.annotate('', xy=(min(experimental), max_density * 0.85), xytext=(max(simulated), max_density * 0.85),
                     arrowprops=dict(arrowstyle='<->', lw=2))
        plt.text(x=np.mean(experimental), y=max_density * 0.8, s=f'p < 0.05', fontsize=12, ha='center')

    plt.legend()
    plt.xlabel('Distance (nm)')
    plt.ylabel('Density')
    plt.title('Comparison of Nearest Distances in Nanometers')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison_plot_with_significance.png', dpi=300)
    plt.close()

def main():
    args = parse_args()

    # Step 1: Load and process fibril data
    filament_data = extract_and_save_data(args.fibril)
    filaments = simulate_filaments(filament_data)

    # Step 2: Load and process antibody data
    ab_data = extract_and_save_data(args.ab)
    ab_distances_nm = calculate_and_convert_distances(ab_data, filaments)
    ab_data['Nearest Distance (nm)'] = ab_distances_nm
    ab_data.to_csv(f'{args.output_prefix}_ab_nearest_distances.csv', index=False)

    # Step 3: Random simulation
    bounds = [(1, 510), (1, 720), (1, 150)]  # adjust based on your image volume
    random_file = f'{args.output_prefix}_random_simulated.txt'
    random_data = random_simulation_and_save(bounds, len(ab_data), random_file)
    random_distances_nm = calculate_and_convert_distances(random_data, filaments)
    random_data['Nearest Distance (nm)'] = random_distances_nm
    random_data.to_csv(f'{args.output_prefix}_random_nearest_distances.csv', index=False)

    # Step 4: Statistical test + plot
    statistical_analysis_and_plot(ab_distances_nm, random_distances_nm, args.output_prefix)

if __name__ == "__main__":
    main()

