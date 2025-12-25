#!/usr/bin/env python3
"""Split model_weights.bin into two parts for separate AppVars."""

import subprocess
import sys
from pathlib import Path

SPLIT_SIZE = 64000  # Bytes per AppVar (leaving room for headers)

def main():
    weights_file = Path("model_weights.bin")
    if not weights_file.exists():
        print("Error: model_weights.bin not found")
        return 1
    
    data = weights_file.read_bytes()
    total_size = len(data)
    
    print(f"Total weights: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    
    if total_size <= SPLIT_SIZE:
        print("Model fits in single AppVar, no split needed")
        return 0
    
    # Split into two parts
    part1 = data[:SPLIT_SIZE]
    part2 = data[SPLIT_SIZE:]
    
    print(f"Part 1: {len(part1):,} bytes")
    print(f"Part 2: {len(part2):,} bytes")
    
    # Write split files
    Path("model_weights_p1.bin").write_bytes(part1)
    Path("model_weights_p2.bin").write_bytes(part2)
    
    # Use convbin to create AppVars
    convbin = "PATH_TO_CONVBIN"
    
    print("\nCreating AppVars...")
    
    # Part 1: GRUMDL1
    result = subprocess.run([
        convbin, "-j", "bin", "-k", "8xv",
        "-i", "model_weights_p1.bin",
        "-o", "GRUMDL1.8xv",
        "-n", "GRUMDL1", "-r"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error creating part 1: {result.stderr}")
        return 1
    print(result.stdout.strip())
    
    # Part 2: GRUMDL2
    result = subprocess.run([
        convbin, "-j", "bin", "-k", "8xv",
        "-i", "model_weights_p2.bin",
        "-o", "GRUMDL2.8xv",
        "-n", "GRUMDL2", "-r"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error creating part 2: {result.stderr}")
        return 1
    print(result.stdout.strip())
    
    # Show file sizes
    print("\nCreated files:")
    for f in ["GRUMDL1.8xv", "GRUMDL2.8xv"]:
        p = Path(f)
        if p.exists():
            print(f"  {f}: {p.stat().st_size:,} bytes")
    
    print("\nDone! Transfer both GRUMDL1.8xv and GRUMDL2.8xv to calculator.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
