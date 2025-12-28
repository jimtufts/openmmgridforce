#!/usr/bin/env python
"""Test .dx format I/O"""

import os
import sys
import tempfile
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import grid_io


def test_dx_roundtrip():
    """Test writing and reading .dx format"""
    print("Testing .dx format I/O...")

    # Create test grid data
    counts = (10, 10, 10)
    spacing = (0.1, 0.1, 0.1)
    origin = (1.0, 2.0, 3.0)
    vals = np.arange(1000, dtype=float)  # 10*10*10 = 1000 points

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.dx', delete=False) as f:
        temp_file = f.name

    try:
        # Write
        grid_io.write_dx(temp_file, counts, spacing, vals, origin)
        print(f"  Wrote test grid to {temp_file}")

        # Read
        data = grid_io.read_dx(temp_file)
        print(f"  Read grid back from {temp_file}")

        # Verify - should be in Angstroms now
        assert tuple(data['counts']) == counts, "Counts mismatch"
        assert np.allclose(data['spacing'], np.array(spacing) * 10.0), "Spacing mismatch (should be in Angstroms)"
        assert np.allclose(data['origin'], np.array(origin) * 10.0), "Origin mismatch (should be in Angstroms)"
        assert np.allclose(data['vals'], vals), "Values mismatch"

        print("  [PASS] All data matches (converted to Angstroms)!")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_dx_gzip():
    """Test gzipped .dx format"""
    print("\nTesting gzipped .dx format...")

    counts = (5, 5, 5)
    spacing = (0.2, 0.2, 0.2)
    origin = (0.0, 0.0, 0.0)
    vals = np.random.randn(125)  # 5*5*5 = 125 points

    with tempfile.NamedTemporaryFile(suffix='.dx.gz', delete=False) as f:
        temp_file = f.name

    try:
        # Write
        grid_io.write_dx(temp_file, counts, spacing, vals, origin)
        print(f"  Wrote compressed grid to {temp_file}")

        # Check file is actually compressed
        assert temp_file.endswith('.gz'), "Should be .gz file"

        # Read - values in Angstroms
        data = grid_io.read_dx(temp_file)
        print(f"  Read compressed grid back")

        # Verify
        assert np.allclose(data['vals'], vals), "Values mismatch in gzipped file"

        print("  [PASS] Gzipped I/O works!")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == '__main__':
    print("=" * 60)
    print("Testing .dx format I/O")
    print("=" * 60)

    test_dx_roundtrip()
    test_dx_gzip()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
