#!/usr/bin/env python
"""
Parse DOCK6 MOL2 files to extract docked poses and scores

Functions:
    parse_mol2_file(mol2_path) -> list of dicts with coordinates and scores
    parse_rec_box_pdb(box_path) -> dict with grid box parameters
"""

import numpy as np
from openmm.unit import angstrom, nanometer
import re


def parse_rec_box_pdb(box_path):
    """
    Parse DOCK6 rec_box.pdb file to get grid box parameters

    Parameters
    ----------
    box_path : str
        Path to rec_box.pdb file

    Returns
    -------
    dict
        Dictionary with keys:
        - 'center': tuple (x, y, z) in Angstroms
        - 'dimensions': tuple (x, y, z) in Angstroms
        - 'min_corner': tuple (x, y, z) in Angstroms
        - 'max_corner': tuple (x, y, z) in Angstroms
    """
    with open(box_path, 'r') as f:
        lines = f.readlines()

    center = None
    dimensions = None

    for line in lines:
        if line.startswith('REMARK') and 'CENTER' in line:
            # REMARK    CENTER (X Y Z)   23.000  23.000  23.000
            # Extract just the numbers after (X Y Z)
            match = re.search(r'CENTER.*?\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if match:
                center = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        elif line.startswith('REMARK') and 'DIMENSIONS' in line:
            # REMARK    DIMENSIONS (X Y Z)   46.000  46.000  46.000
            match = re.search(r'DIMENSIONS.*?\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if match:
                dimensions = (float(match.group(1)), float(match.group(2)), float(match.group(3)))

    if center is None or dimensions is None:
        raise ValueError(f"Could not parse center or dimensions from {box_path}")

    # Calculate min and max corners
    min_corner = tuple(c - d/2 for c, d in zip(center, dimensions))
    max_corner = tuple(c + d/2 for c, d in zip(center, dimensions))

    return {
        'center': center,
        'dimensions': dimensions,
        'min_corner': min_corner,
        'max_corner': max_corner
    }


def parse_mol2_file(mol2_path, max_poses=None):
    """
    Parse DOCK6 MOL2 file containing multiple docked poses

    Parameters
    ----------
    mol2_path : str
        Path to MOL2 file (e.g., anchor_and_grow_scored.mol2 or xtal_scored.mol2)
    max_poses : int, optional
        Maximum number of poses to read (default: all)

    Returns
    -------
    list of dict
        Each dict contains:
        - 'name': molecule name
        - 'coordinates': numpy array of shape (n_atoms, 3) in Angstroms
        - 'atom_names': list of atom names
        - 'atom_types': list of atom types
        - 'charges': numpy array of partial charges
        - 'scores': dict of DOCK6 scores (if present)
          - 'grid_score': total grid score
          - 'grid_vdw': vdW component
          - 'grid_es': electrostatic component
          - 'int_energy': internal energy
          - 'rmsd_heavy': heavy atom RMSD (if present)
          - 'rmsd_all': all atom RMSD (if present)
          - 'rmsd_min': minimum RMSD (if present)
    """
    with open(mol2_path, 'r') as f:
        content = f.read()

    # Split into individual molecules
    # Scores appear BEFORE @<TRIPOS>MOLECULE, so we need to pair them
    molecules = []
    mol_splits = content.split('@<TRIPOS>MOLECULE')

    for i in range(1, len(mol_splits)):  # Skip first (everything before first molecule)
        if max_poses is not None and len(molecules) >= max_poses:
            break

        # Scores are in mol_splits[i-1] (before the @<TRIPOS>MOLECULE marker)
        # Molecule data is in mol_splits[i] (after the marker)
        score_block = mol_splits[i-1]
        mol_block = mol_splits[i]

        mol_data = parse_single_mol2_block(mol_block, score_block)
        if mol_data is not None:
            molecules.append(mol_data)

    return molecules


def parse_single_mol2_block(mol_block, score_block=""):
    """Parse a single molecule block from MOL2 file

    Parameters
    ----------
    mol_block : str
        Text after @<TRIPOS>MOLECULE marker
    score_block : str
        Text before @<TRIPOS>MOLECULE marker (contains scores)
    """

    lines = mol_block.strip().split('\n')

    # Initialize data structure
    mol_data = {
        'name': None,
        'coordinates': None,
        'atom_names': [],
        'atom_types': [],
        'charges': [],
        'scores': {}
    }

    # Parse scores from the block BEFORE @<TRIPOS>MOLECULE
    score_lines = score_block.strip().split('\n')
    header_lines = [line for line in score_lines if line.startswith('##########')]

    atom_section_start = None
    for i, line in enumerate(lines):
        if line.startswith('@<TRIPOS>ATOM'):
            atom_section_start = i
            break

    # Extract scores from header
    for line in header_lines:
        if 'Grid Score:' in line or 'Grid_Score:' in line:
            match = re.search(r'Grid[_ ]Score:\s+([-\d.]+)', line)
            if match:
                mol_data['scores']['grid_score'] = float(match.group(1))
        elif 'Grid_vdw:' in line:
            match = re.search(r'Grid_vdw:\s+([-\d.]+)', line)
            if match:
                mol_data['scores']['grid_vdw'] = float(match.group(1))
        elif 'Grid_es:' in line:
            match = re.search(r'Grid_es:\s+([-\d.]+)', line)
            if match:
                mol_data['scores']['grid_es'] = float(match.group(1))
        elif 'Int_energy:' in line:
            match = re.search(r'Int_energy:\s+([-\d.]+)', line)
            if match:
                mol_data['scores']['int_energy'] = float(match.group(1))
        elif 'HA_RMSDs:' in line:
            match = re.search(r'HA_RMSDs:\s+([-\d.]+)', line)
            if match:
                mol_data['scores']['rmsd_heavy'] = float(match.group(1))
        elif 'HA_RMSDh:' in line:
            match = re.search(r'HA_RMSDh:\s+([-\d.]+)', line)
            if match:
                mol_data['scores']['rmsd_all'] = float(match.group(1))
        elif 'HA_RMSDm:' in line:
            match = re.search(r'HA_RMSDm:\s+([-\d.]+)', line)
            if match:
                mol_data['scores']['rmsd_min'] = float(match.group(1))

    # Parse molecule name (first line after @<TRIPOS>MOLECULE)
    if len(lines) > 0:
        mol_data['name'] = lines[0].strip()

    # Parse atoms section
    if atom_section_start is None:
        return None

    coords = []
    for line in lines[atom_section_start + 1:]:
        if line.startswith('@<TRIPOS>'):
            break

        parts = line.split()
        if len(parts) < 6:
            continue

        # MOL2 format: atom_id atom_name x y z atom_type [subst_id] [subst_name] [charge]
        atom_name = parts[1]
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        atom_type = parts[5]
        charge = float(parts[8]) if len(parts) > 8 else 0.0

        mol_data['atom_names'].append(atom_name)
        mol_data['atom_types'].append(atom_type)
        mol_data['charges'].append(charge)
        coords.append([x, y, z])

    if coords:
        mol_data['coordinates'] = np.array(coords)
        mol_data['charges'] = np.array(mol_data['charges'])
    else:
        return None

    return mol_data


def convert_mol2_to_openmm_positions(mol2_coords):
    """
    Convert MOL2 coordinates (Angstroms) to OpenMM Quantity with nanometers

    Parameters
    ----------
    mol2_coords : numpy.ndarray
        Coordinates in Angstroms, shape (n_atoms, 3)

    Returns
    -------
    openmm.unit.Quantity
        Positions in nanometers
    """
    # MOL2 uses Angstroms, OpenMM uses nanometers
    return mol2_coords * 0.1 * nanometer


def get_grid_bounds_for_openmm(box_params, spacing_angstrom, buffer_nm=0.0):
    """
    Convert DOCK6 box parameters to OpenMM grid parameters

    Parameters
    ----------
    box_params : dict
        Output from parse_rec_box_pdb()
    spacing_angstrom : float
        Grid spacing in Angstroms
    buffer_nm : float
        Buffer to add around grid in nanometers (added to each side)

    Returns
    -------
    dict
        Dictionary with:
        - 'origin_nm': tuple (x, y, z) in nanometers
        - 'dimensions_nm': tuple (x, y, z) in nanometers
        - 'spacing_nm': float in nanometers
        - 'grid_counts': tuple (nx, ny, nz)
    """
    # Convert to nanometers
    origin_nm = tuple(c * 0.1 - buffer_nm for c in box_params['min_corner'])
    dimensions_nm = tuple(d * 0.1 + 2 * buffer_nm for d in box_params['dimensions'])
    spacing_nm = spacing_angstrom * 0.1

    # Calculate grid counts
    # Need +1 because N grid points cover (N-1) cells
    # e.g., to cover 34A with 0.5A spacing: need 69 points (68 cells), not 68 points
    grid_counts = tuple(int(np.ceil(d / spacing_nm)) + 1 for d in dimensions_nm)

    return {
        'origin_nm': origin_nm,
        'dimensions_nm': dimensions_nm,
        'spacing_nm': spacing_nm,
        'grid_counts': grid_counts
    }


if __name__ == '__main__':
    """Test the parser on 1g9v system"""

    print("Testing MOL2 parser on 1g9v system")
    print("=" * 80)

    # Test box parser
    box_path = '/scratch/AstexDiv_comprehensive/4-UCSF_dock6/1g9v/rec_box.pdb'
    box_params = parse_rec_box_pdb(box_path)
    print("\nBox parameters:")
    print(f"  Center: {box_params['center']} Å")
    print(f"  Dimensions: {box_params['dimensions']} Å")
    print(f"  Min corner: {box_params['min_corner']} Å")
    print(f"  Max corner: {box_params['max_corner']} Å")

    # Test grid bounds conversion
    grid_bounds = get_grid_bounds_for_openmm(box_params, spacing_angstrom=0.125)
    print(f"\nGrid parameters for 0.125Å spacing:")
    print(f"  Origin: {grid_bounds['origin_nm']} nm")
    print(f"  Dimensions: {grid_bounds['dimensions_nm']} nm")
    print(f"  Grid counts: {grid_bounds['grid_counts']}")

    # Test crystal structure parser
    xtal_path = '/scratch/AstexDiv_comprehensive/4-UCSF_dock6/1g9v/xtal_scored.mol2'
    xtal_poses = parse_mol2_file(xtal_path)
    print(f"\nCrystal structure:")
    print(f"  Number of poses: {len(xtal_poses)}")
    if xtal_poses:
        pose = xtal_poses[0]
        print(f"  Name: {pose['name']}")
        print(f"  Atoms: {len(pose['coordinates'])}")
        print(f"  Grid Score: {pose['scores'].get('grid_score', 'N/A')}")
        print(f"  Grid vdW: {pose['scores'].get('grid_vdw', 'N/A')}")
        print(f"  Grid ES: {pose['scores'].get('grid_es', 'N/A')}")
        print(f"  First 3 coords:\n{pose['coordinates'][:3]}")

    # Test docked poses parser (only first 5)
    docked_path = '/scratch/AstexDiv_comprehensive/4-UCSF_dock6/1g9v/anchor_and_grow_scored.mol2'
    docked_poses = parse_mol2_file(docked_path, max_poses=5)
    print(f"\nDocked poses (first 5):")
    print(f"  Number of poses: {len(docked_poses)}")
    for i, pose in enumerate(docked_poses[:3]):
        print(f"\n  Pose {i+1}:")
        print(f"    Name: {pose['name']}")
        print(f"    Atoms: {len(pose['coordinates'])}")
        print(f"    Grid Score: {pose['scores'].get('grid_score', 'N/A')}")
        print(f"    RMSD (heavy): {pose['scores'].get('rmsd_heavy', 'N/A')}")
        print(f"    RMSD (min): {pose['scores'].get('rmsd_min', 'N/A')}")
