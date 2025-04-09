"""
A script for analyzing filament structures and their associated antibodies.
Exports results in both TXT and RELION STAR formats.
"""
#usage:python script.py --id filamentID --distance 10.5 --fibril Filament.txt --ab Coords_Ab.txt

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


def extract_filament(fibril_data: pd.DataFrame, filament_id: int) -> pd.DataFrame:
    """
    Extract coordinates for a specific filament ID.
    
    Args:
        fibril_data: DataFrame containing all filament data
        filament_id: Target filament ID to extract
        
    Returns:
        DataFrame containing only the specified filament's coordinates
    """
    filament_data = fibril_data[fibril_data['FilamentID'] == filament_id]
    if filament_data.empty:
        raise ValueError(f"Filament ID {filament_id} not found in input data")
        
    output_path = Path(f'filamentID_{filament_id}.txt')
    filament_data[['x', 'y', 'z']].to_csv(output_path, index=False)
    return filament_data[['x', 'y', 'z']]


def filter_nearest_antibodies(
    antibody_data: pd.DataFrame,
    filament_coords: pd.DataFrame,
    distance_threshold: float
) -> pd.DataFrame:
    """
    Filter antibodies within specified distance of filament coordinates.
    
    Args:
        antibody_data: DataFrame containing antibody coordinates
        filament_coords: DataFrame containing filament coordinates
        distance_threshold: Maximum allowed distance (in same units as coordinates)
        
    Returns:
        Filtered DataFrame of nearby antibodies
    """
    kd_tree = cKDTree(filament_coords)
    distances, _ = kd_tree.query(antibody_data[['x', 'y', 'z']], k=1)
    return antibody_data[distances < distance_threshold]


def convert_to_star(input_path: Path, output_path: Path) -> None:
    """
    Convert coordinate file to RELION STAR format.
    
    Args:
        input_path: Path to input CSV/TXT file
        output_path: Path for output STAR file
    """
    data = pd.read_csv(input_path)
    
    star_header = (
        "# Created by filament analysis script\n\n"
        "data_\n\n"
        "loop_\n"
        "_rlnCoordinateX #1\n"
        "_rlnCoordinateY #2\n"
        "_rlnCoordinateZ #3\n"
    )
    
    with output_path.open('w') as f:
        f.write(star_header)
        data.to_csv(f, sep='\t', header=False, index=False)


def main(
    fibril_path: Path,
    antibody_path: Path,
    filament_id: int,
    distance_threshold: float
) -> None:
    """
    Main processing pipeline.
    
    Args:
        fibril_path: Path to filament data file
        antibody_path: Path to antibody data file
        filament_id: Target filament ID to analyze
        distance_threshold: Maximum antibody distance to include
    """
    # Load input data
    fibril_data = pd.read_csv(fibril_path)
    antibody_data = pd.read_csv(antibody_path)

    # Process filament data
    filament_coords = extract_filament(fibril_data, filament_id)
    convert_to_star(
        Path(f'filamentID_{filament_id}.txt'),
        Path(f'filamentID_{filament_id}.star')
    )

    # Process antibody data
    filtered_antibodies = filter_nearest_antibodies(
        antibody_data,
        filament_coords,
        distance_threshold
    )
    
    base_prefix = antibody_path.stem
    output_prefix = f"{base_prefix}_{filament_id}_{int(distance_threshold)}"
    
    filtered_antibodies.to_csv(f"{output_prefix}.txt", index=False)
    convert_to_star(
        Path(f"{output_prefix}.txt"),
        Path(f"{output_prefix}.star")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze filament structures and associated antibodies."
    )
    parser.add_argument(
        '--fibril',
        type=Path,
        default='Fibril_extracted.txt',
        help='Path to filament data file (default: Fibril_extracted.txt)'
    )
    parser.add_argument(
        '--ab',
        type=Path,
        default='H21.txt',
        help='Path to antibody data file (default: H21.txt)'
    )
    parser.add_argument(
        '--id',
        type=int,
        required=True,
        help='Filament ID to analyze'
    )
    parser.add_argument(
        '--distance',
        type=float,
        required=True,
        help='Maximum antibody distance from filament (in coordinate units)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.fibril.exists():
        raise FileNotFoundError(f"Fibril file not found: {args.fibril}")
    if not args.ab.exists():
        raise FileNotFoundError(f"Antibody file not found: {args.ab}")
    if args.distance <= 0:
        raise ValueError("Distance threshold must be positive")

    main(
        fibril_path=args.fibril,
        antibody_path=args.ab,
        filament_id=args.id,
        distance_threshold=args.distance
    )
